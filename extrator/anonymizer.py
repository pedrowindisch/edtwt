from __future__ import annotations

import csv
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from extrator.extrator import load_dotenv


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class AnonymizerConfig:
    database_path: Path
    output_path: Path

    @classmethod
    def from_env(cls) -> "AnonymizerConfig":
        load_dotenv()

        database_path = Path(os.getenv("TWITTER_DATABASE_PATH") or "data/tweets.sqlite3")
        output_path = Path(
            os.getenv("TWITTER_ANONYMIZED_OUTPUT_CSV") or "exports/tweets_anonymized.csv"
        )
        return cls(database_path=database_path, output_path=output_path)


class Anonymizer:
    def __init__(
        self,
        progress_callback: Callable[[int], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ):
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def export(self) -> dict[str, int | str]:
        config = AnonymizerConfig.from_env()
        if not config.database_path.exists():
            raise ValueError(f"Banco SQLite nao encontrado em {config.database_path}.")

        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(config.database_path) as connection:
            connection.row_factory = sqlite3.Row
            total_rows = connection.execute("SELECT COUNT(*) AS total FROM tweets").fetchone()["total"]

            if total_rows == 0:
                raise ValueError("Nao ha tweets no banco para anonimizar.")

            self._emit_status(f"Exportando {total_rows} linha(s) anonimizadas...")
            self._emit_progress(0)

            cursor = connection.execute(
                """
                SELECT
                    search_date,
                    created_at,
                    lang,
                    text,
                    source,
                    retweet_count,
                    reply_count,
                    like_count,
                    quote_count,
                    view_count,
                    bookmark_count,
                    is_reply,
                    is_limited_reply,
                    author_followers,
                    author_following,
                    hashtags,
                    urls,
                    mentions
                FROM tweets
                ORDER BY row_id
                """
            )

            with config.output_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self._headers())
                writer.writeheader()

                processed = 0
                for row in cursor:
                    writer.writerow(self._anonymize_row(row))
                    processed += 1

                    if processed % 100 == 0 or processed == total_rows:
                        self._emit_progress(int(processed / total_rows * 100))
                        self._emit_status(
                            f"Anonimizadas {processed}/{total_rows} linha(s)..."
                        )

        return {"rows": total_rows, "path": str(config.output_path)}

    def _anonymize_row(self, row: sqlite3.Row) -> dict[str, str | int | bool]:
        hashtags = split_pipe_values(row["hashtags"])
        urls = split_pipe_values(row["urls"])
        mentions = split_pipe_values(row["mentions"])
        text = normalize_text(row["text"])

        return {
            "search_date": normalize_date(row["search_date"]),
            "created_date": normalize_date(row["created_at"]),
            "lang": normalize_text(row["lang"]),
            "source": normalize_text(row["source"]),
            "text_redacted": redact_text(text),
            "text_length": len(text),
            "retweet_count_bucket": bucketize_metric(row["retweet_count"]),
            "reply_count_bucket": bucketize_metric(row["reply_count"]),
            "like_count_bucket": bucketize_metric(row["like_count"]),
            "quote_count_bucket": bucketize_metric(row["quote_count"]),
            "view_count_bucket": bucketize_metric(row["view_count"]),
            "bookmark_count_bucket": bucketize_metric(row["bookmark_count"]),
            "author_followers_bucket": bucketize_metric(row["author_followers"]),
            "author_following_bucket": bucketize_metric(row["author_following"]),
            "is_reply": bool(row["is_reply"]),
            "is_limited_reply": bool(row["is_limited_reply"]),
            "hashtags_count": len(hashtags),
            "urls_count": len(urls),
            "mentions_count": len(mentions),
            "contains_recovery_term": contains_recovery_term(text),
        }

    def _headers(self) -> list[str]:
        return [
            "search_date",
            "created_date",
            "lang",
            "source",
            "text_redacted",
            "text_length",
            "retweet_count_bucket",
            "reply_count_bucket",
            "like_count_bucket",
            "quote_count_bucket",
            "view_count_bucket",
            "bookmark_count_bucket",
            "author_followers_bucket",
            "author_following_bucket",
            "is_reply",
            "is_limited_reply",
            "hashtags_count",
            "urls_count",
            "mentions_count",
            "contains_recovery_term",
        ]

    def _emit_progress(self, value: int) -> None:
        if self.progress_callback:
            self.progress_callback(value)

    def _emit_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_date(value: object) -> str:
    raw = normalize_text(value)
    if not raw:
        return ""
    return raw[:10]


def split_pipe_values(value: object) -> list[str]:
    raw = normalize_text(value)
    if not raw:
        return []
    return [item for item in raw.split("|") if item]


def redact_text(value: str) -> str:
    cleaned = URL_PATTERN.sub("[url]", value)
    cleaned = MENTION_PATTERN.sub("[mention]", cleaned)
    cleaned = HASHTAG_PATTERN.sub("[hashtag]", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def bucketize_metric(value: object) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0

    if number == 0:
        return "0"
    if number <= 10:
        return "1-10"
    if number <= 100:
        return "11-100"
    if number <= 1000:
        return "101-1000"
    if number <= 10000:
        return "1001-10000"
    return "10000+"


def contains_recovery_term(value: str) -> bool:
    lowered = value.lower()
    keywords = ("recovery", "edrecovery", "recuper", "treatment", "tratamento")
    return any(keyword in lowered for keyword in keywords)
