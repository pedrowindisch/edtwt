from __future__ import annotations

import csv
import os
import time
from threading import Event
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import requests

from extrator.models.Tweet import Tweet
from extrator.storage import ExtractionStorage


API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
MIN_REQUEST_INTERVAL_SECONDS = 0.0
RATE_LIMIT_RETRY_SECONDS = 10.0
MAX_RATE_LIMIT_RETRY_SECONDS = 30.0
IDLE_POLL_INTERVAL_SECONDS = 60.0
TOP_PAGE_BATCH_SIZE = 10


class ExtractionApiError(RuntimeError):
    pass


class PaginationOverlapError(RuntimeError):
    pass


SEARCH_BY_DATE = "date"
SEARCH_TOPS = "top"


@dataclass(slots=True, frozen=True)
class ExtractionConfig:
    api_key: str
    base_query: str
    lang_filter: str | None
    output_path: Path
    database_path: Path
    legacy_checkpoint_path: Path

    @classmethod
    def from_env(cls, default_output_path: str) -> "ExtractionConfig":
        load_dotenv()

        api_key = os.getenv("TWITTER_API_KEY")
        base_query = os.getenv("TWITTER_SEARCH_QUERY")
        lang_filter = os.getenv("TWITTER_LANG_FILTER") or "pt"
        output_path_raw = os.getenv("TWITTER_OUTPUT_CSV") or default_output_path
        database_path_raw = os.getenv("TWITTER_DATABASE_PATH") or "data/tweets.sqlite3"

        if not api_key:
            raise ValueError("Defina TWITTER_API_KEY no ambiente ou no arquivo .env.")

        if not base_query:
            raise ValueError("Defina TWITTER_SEARCH_QUERY no ambiente ou no arquivo .env.")

        output_path = Path(output_path_raw)
        database_path = Path(database_path_raw)
        checkpoint_raw = os.getenv("TWITTER_RESUME_FILE") or f"{output_path}.checkpoint"

        return cls(
            api_key=api_key,
            base_query=base_query.strip(),
            lang_filter=lang_filter.strip() if lang_filter else None,
            output_path=output_path,
            database_path=database_path,
            legacy_checkpoint_path=Path(checkpoint_raw),
        )


class Extrator:
    def __init__(
        self,
        caminho_arquivo: str,
        progress_callback: Callable[[int], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        top_batch_prompt_callback: Callable[[int], bool] | None = None,
    ):
        self.caminho_arquivo = caminho_arquivo
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.top_batch_prompt_callback = top_batch_prompt_callback
        self._last_request_at: float | None = None
        self._stop_requested = Event()

    def extrair(
        self,
        newest_date: date | None = None,
        oldest_date: date | None = None,
        search_mode: str = SEARCH_BY_DATE,
    ) -> dict[str, int | str | bool]:
        config = ExtractionConfig.from_env(self.caminho_arquivo)
        storage = ExtractionStorage(config.database_path, config.output_path)
        query_key = self._build_query_key(config, search_mode)
        if search_mode == SEARCH_BY_DATE:
            storage.ensure_query_initialized(query_key, config.legacy_checkpoint_path)
        total_rows = 0
        total_dates = 0
        first_date: str | None = None
        last_date: str | None = None

        self._emit_status("Rate limit de seguranca ativo: 1 request a cada 5s.")

        with requests.Session() as session:
            session.headers.update({"X-API-Key": config.api_key})

            if search_mode == SEARCH_TOPS:
                self._emit_progress(0)
                self._emit_status("Modo Top ativo. Ignorando datas e buscando tweets em destaque.")
                tweets_saved = self._fetch_top(session, config, storage)
                self._emit_progress(100)
                return {
                    "rows": tweets_saved,
                    "dates": 0,
                    "path": str(config.output_path),
                    "database_path": str(config.database_path),
                    "start_date": "top",
                    "end_date": "top",
                    "stopped": self._stop_requested.is_set(),
                    "search_mode": search_mode,
                }

            while True:
                current_newest_date, current_oldest_date, keep_watching = self._resolve_cycle_dates(
                    newest_date,
                    oldest_date,
                )
                dates_to_fetch = storage.get_pending_dates(
                    query_key,
                    current_oldest_date,
                    current_newest_date,
                )

                if first_date is None:
                    first_date = current_newest_date.isoformat()
                last_date = current_oldest_date.isoformat()

                self._emit_progress(0)

                if not dates_to_fetch:
                    self._emit_progress(100)
                    if self._stop_requested.is_set():
                        self._emit_status("Parada solicitada. Extração encerrada.")
                        break

                    self._emit_status(
                        f"Nenhuma data pendente entre {current_oldest_date.isoformat()} e {current_newest_date.isoformat()}."
                    )

                    if not keep_watching:
                        break

                    self._emit_status(
                        f"Nenhuma data nova para hoje ({current_newest_date.isoformat()}). Aguardando novos dias..."
                    )
                    if self._sleep_until_next_cycle(IDLE_POLL_INTERVAL_SECONDS):
                        self._emit_status("Parada solicitada. Extração encerrada.")
                        break
                    continue

                cycle_total_dates = len(dates_to_fetch)
                self._emit_status(
                    f"Iniciando extracao de {cycle_total_dates} dia(s), de {current_newest_date.isoformat()} ate {current_oldest_date.isoformat()}."
                )

                for index, current_date in enumerate(dates_to_fetch, start=1):
                    self._emit_status(
                        f"Iniciando raspagem do dia {current_date.isoformat()} ({index}/{cycle_total_dates})."
                    )
                    try:
                        tweets_saved = self._fetch_day(session, config, current_date, storage)
                    except PaginationOverlapError as exc:
                        self._emit_status(
                            f"Sobreposicao entre paginas detectada em {current_date.isoformat()}. Encerrando apenas este dia. Detalhes: {exc}"
                        )
                        tweets_saved = 0

                    if tweets_saved:
                        total_rows += tweets_saved
                    else:
                        self._emit_status(
                            f"Nenhum tweet encontrado em {current_date.isoformat()}. Pulando para o proximo dia."
                        )

                    storage.mark_date_completed(query_key, current_date)
                    total_dates += 1
                    self._emit_progress(int(index / cycle_total_dates * 100))
                    self._emit_status(
                        f"{current_date.isoformat()} concluido com {tweets_saved} tweet(s) salvos."
                    )

                    if self._stop_requested.is_set():
                        self._emit_status(
                            f"Parada solicitada. Encerrando apos concluir {current_date.isoformat()}."
                        )
                        break

                if self._stop_requested.is_set():
                    break

                if not keep_watching:
                    break

        return {
            "rows": total_rows,
            "dates": total_dates,
            "path": str(config.output_path),
            "database_path": str(config.database_path),
            "start_date": first_date or "",
            "end_date": last_date or "",
            "stopped": self._stop_requested.is_set(),
            "search_mode": search_mode,
        }

    def solicitar_parada(self) -> None:
        self._stop_requested.set()

    def _resolve_cycle_dates(
        self,
        newest_date: date | None,
        oldest_date: date | None,
    ) -> tuple[date, date, bool]:
        if newest_date or oldest_date:
            resolved_newest_date = newest_date or oldest_date or date.today()
            resolved_oldest_date = oldest_date or newest_date or date.today()

            if resolved_oldest_date > resolved_newest_date:
                raise ValueError("A data mais antiga nao pode ser maior que a data mais recente.")

            return resolved_newest_date, resolved_oldest_date, False

        today = date.today()
        return today, today, True

    def _build_query_key(self, config: ExtractionConfig, search_mode: str) -> str:
        lang_filter = config.lang_filter or ""
        return f"mode={search_mode}|query={config.base_query}|lang={lang_filter}"

    def _format_query_datetime(self, value: date) -> str:
        return f"{value.isoformat()}_00:00:00_UTC"

    def _fetch_day(
        self,
        session: requests.Session,
        config: ExtractionConfig,
        current_date: date,
        storage: ExtractionStorage,
    ) -> int:
        next_date = current_date + timedelta(days=1)
        query = (
            f"{config.base_query} "
            f"since:{self._format_query_datetime(current_date)} "
            f"until:{self._format_query_datetime(next_date)}"
        )
        if config.lang_filter:
            query = f"{query} lang:{config.lang_filter}"

        return self._fetch_query(
            session=session,
            config=config,
            storage=storage,
            query=query,
            query_type="Latest",
            label=current_date.isoformat(),
            search_date=current_date.isoformat(),
            paginate=False,
        )

    def _fetch_top(
        self,
        session: requests.Session,
        config: ExtractionConfig,
        storage: ExtractionStorage,
    ) -> int:
        query = config.base_query
        if config.lang_filter:
            query = f"{query} lang:{config.lang_filter}"

        self._emit_status(f"Executando busca Top com query: {query}")

        return self._fetch_query(
            session=session,
            config=config,
            storage=storage,
            query=query,
            query_type="Top",
            label="top",
            search_date="top",
            paginate=True,
        )

    def _fetch_query(
        self,
        session: requests.Session,
        config: ExtractionConfig,
        storage: ExtractionStorage,
        query: str,
        query_type: str,
        label: str,
        search_date: str,
        paginate: bool,
    ) -> int:

        page_number = 1
        rate_limit_hits = 0
        tweets_saved = 0
        cursor: str | None = None

        while True:
            params = {"queryType": query_type, "query": query}
            if paginate and cursor:
                params["cursor"] = cursor

            self._respect_rate_limit(config)
            self._emit_status(
                f"Enviando request para {label} [{query_type}] (pagina {page_number}, tentativa {rate_limit_hits + 1})..."
            )
            self._emit_status(
                f"Aguardando resposta da API para {label} [{query_type}] (pagina {page_number})..."
            )

            try:
                response = session.get(API_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                raise ExtractionApiError(f"Falha na requisicao para a API: {exc}") from exc

            self._last_request_at = time.monotonic()

            payload = self._decode_payload(response)

            if self._handle_rate_limit_response(response, payload, rate_limit_hits):
                rate_limit_hits += 1
                continue

            rate_limit_hits = 0

            page_tweets = payload.get("tweets", [])
            self._emit_status(
                f"Resposta recebida para {label} [{query_type}] (pagina {page_number}) com {len(page_tweets)} tweet(s)."
            )

            if response.status_code >= 400:
                raise ExtractionApiError(self._extract_api_error_message(payload, response))

            parsed_tweets = [Tweet.from_payload(item) for item in page_tweets]
            if parsed_tweets:
                self._emit_status(
                    f"Salvando {len(parsed_tweets)} tweet(s) da pagina {page_number} de {label} imediatamente."
                )
                storage.save_tweets(parsed_tweets, search_date)
                self._append_rows(config.output_path, parsed_tweets, search_date)
                tweets_saved += len(parsed_tweets)

            cursor = payload.get("next_cursor") if paginate else None
            has_next_page = bool(payload.get("has_next_page") and cursor)

            if paginate and has_next_page and page_number % TOP_PAGE_BATCH_SIZE == 0:
                self._emit_status(
                    f"Lote de {TOP_PAGE_BATCH_SIZE} paginas concluido em {label}."
                )
                if not self._should_continue_top_batch(page_number):
                    self._emit_status(
                        f"Paginacao Top encerrada apos {page_number} pagina(s) por decisao do usuario."
                    )
                    break

            if not has_next_page:
                self._emit_status(
                    f"Busca concluida para {label}. Total salvo: {tweets_saved} tweet(s)."
                )
                break

            page_number += 1
            self._emit_status(
                f"Proxima pagina detectada para {label}. Continuando para a pagina {page_number}."
            )

        return tweets_saved

    def _should_continue_top_batch(self, page_number: int) -> bool:
        if self.top_batch_prompt_callback is None:
            return False

        return self.top_batch_prompt_callback(page_number)

    def _handle_rate_limit_response(
        self,
        response: requests.Response,
        payload: dict,
        rate_limit_hits: int,
    ) -> bool:
        if not self._is_rate_limit_response(response, payload):
            return False

        retry_after = self._get_retry_after_seconds(response, payload, rate_limit_hits)
        self._emit_status(
            f"Limite da API atingido. Aguardando {retry_after:.1f}s para tentar novamente..."
        )
        self._sleep_until_next_cycle(retry_after)
        return True

    def _respect_rate_limit(self, config: ExtractionConfig) -> None:
        if self._last_request_at is None:
            return

        elapsed = time.monotonic() - self._last_request_at
        remaining = MIN_REQUEST_INTERVAL_SECONDS - elapsed

        if remaining <= 0:
            return

        self._emit_status(
            f"Aguardando {remaining:.1f}s antes da proxima request..."
        )
        self._sleep_until_next_cycle(remaining)

    def _sleep_until_next_cycle(self, total_seconds: float) -> bool:
        deadline = time.monotonic() + total_seconds

        while True:
            if self._stop_requested.is_set():
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            time.sleep(min(1.0, remaining))

    def _decode_payload(self, response: requests.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            if response.status_code >= 400:
                raise ExtractionApiError(
                    f"API retornou HTTP {response.status_code} sem JSON valido."
                ) from exc

            raise ExtractionApiError("API retornou uma resposta invalida.") from exc

        if not isinstance(payload, dict):
            raise ExtractionApiError("API retornou um payload em formato inesperado.")

        return payload

    def _is_rate_limit_response(self, response: requests.Response, payload: dict) -> bool:
        if response.status_code == 429:
            return True

        message = str(payload.get("message") or "").lower()
        return "too many requests" in message or "qps limit" in message

    def _get_retry_after_seconds(
        self,
        response: requests.Response,
        payload: dict,
        rate_limit_hits: int,
    ) -> float:
        minimum_retry = min(
            RATE_LIMIT_RETRY_SECONDS + rate_limit_hits * 5.0,
            MAX_RATE_LIMIT_RETRY_SECONDS,
        )

        retry_after_header = response.headers.get("Retry-After")
        if retry_after_header:
            try:
                return max(float(retry_after_header), minimum_retry)
            except ValueError:
                pass

        message = str(payload.get("message") or "").lower()
        if "5 seconds" in message:
            return minimum_retry

        return minimum_retry

    def _extract_api_error_message(
        self,
        payload: dict,
        response: requests.Response,
    ) -> str:
        error_code = payload.get("error")
        message = payload.get("message")

        if error_code is not None and message:
            return f"Erro da API ({error_code}): {message}"

        if message:
            return f"Erro da API: {message}"

        return f"API retornou HTTP {response.status_code}."

    def _append_rows(self, output_path: Path, tweets: list[Tweet], search_date: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_path.exists() and output_path.stat().st_size > 0

        with output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=Tweet.csv_headers())

            if not file_exists:
                writer.writeheader()

            for tweet in tweets:
                writer.writerow(tweet.to_csv_row(search_date))

    def _emit_progress(self, value: int) -> None:
        if self.progress_callback:
            self.progress_callback(value)

    def _emit_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)


def load_dotenv(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value
