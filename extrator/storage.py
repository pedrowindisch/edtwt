from __future__ import annotations

import csv
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from extrator.models.Tweet import Tweet


LEGACY_CHECKPOINT_PREFIX = "last_completed_date="


class ExtractionStorage:
    def __init__(self, database_path: Path, csv_path: Path):
        self.database_path = database_path
        self.csv_path = csv_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_database()

    def ensure_query_initialized(
        self,
        query_key: str,
        legacy_checkpoint_path: Path | None = None,
    ) -> None:
        with self._connect() as connection:
            query_rows = connection.execute(
                "SELECT 1 FROM scraped_days WHERE query_key = ? LIMIT 1",
                (query_key,),
            ).fetchone()
            if query_rows:
                return

            has_any_scraped_day = connection.execute(
                "SELECT 1 FROM scraped_days LIMIT 1"
            ).fetchone()
            if has_any_scraped_day:
                return

            existing_dates = self._load_known_completed_dates(connection, legacy_checkpoint_path)
            connection.executemany(
                """
                INSERT OR IGNORE INTO scraped_days (query_key, scrape_date, completed_at)
                VALUES (?, ?, ?)
                """,
                [
                    (query_key, completed_date.isoformat(), self._now_iso())
                    for completed_date in existing_dates
                ],
            )
            connection.commit()

    def get_pending_dates(
        self,
        query_key: str,
        oldest_date: date,
        newest_date: date,
    ) -> list[date]:
        completed_dates = self.get_completed_dates(query_key, oldest_date, newest_date)
        pending_dates: list[date] = []
        current_date = newest_date

        while current_date >= oldest_date:
            if current_date not in completed_dates:
                pending_dates.append(current_date)
            current_date -= timedelta(days=1)

        return pending_dates

    def get_completed_dates(
        self,
        query_key: str,
        oldest_date: date,
        newest_date: date,
    ) -> set[date]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT scrape_date
                FROM scraped_days
                WHERE query_key = ?
                  AND scrape_date BETWEEN ? AND ?
                """,
                (query_key, oldest_date.isoformat(), newest_date.isoformat()),
            ).fetchall()

        completed_dates: set[date] = set()
        for row in rows:
            parsed = self._parse_date(row["scrape_date"])
            if parsed is not None:
                completed_dates.add(parsed)

        return completed_dates

    def save_tweets(self, tweets: list[Tweet], search_date: str) -> int:
        if not tweets:
            return 0

        columns = Tweet.csv_headers()
        placeholders = ", ".join("?" for _ in columns)
        records = [tweet.to_record(search_date) for tweet in tweets]

        with self._connect() as connection:
            connection.executemany(
                f"""
                INSERT INTO tweets ({", ".join(columns)})
                VALUES ({placeholders})
                """,
                [tuple(record[column] for column in columns) for record in records],
            )
            connection.commit()

        return len(records)

    def mark_date_completed(self, query_key: str, completed_date: date) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scraped_days (query_key, scrape_date, completed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(query_key, scrape_date) DO UPDATE SET
                    completed_at = excluded.completed_at
                """,
                (
                    query_key,
                    completed_date.isoformat(),
                    self._now_iso(),
                ),
            )
            connection.commit()

    def _ensure_database(self) -> None:
        with self._connect() as connection:
            self._migrate_tweets_table_if_needed(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scraped_days (
                    query_key TEXT NOT NULL,
                    scrape_date TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (query_key, scrape_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tweets (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_date TEXT NOT NULL,
                    id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    retweet_count INTEGER NOT NULL,
                    reply_count INTEGER NOT NULL,
                    like_count INTEGER NOT NULL,
                    quote_count INTEGER NOT NULL,
                    view_count INTEGER NOT NULL,
                    bookmark_count INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    in_reply_to_id TEXT NOT NULL,
                    in_reply_to_user_id TEXT NOT NULL,
                    in_reply_to_username TEXT NOT NULL,
                    is_reply INTEGER NOT NULL,
                    is_limited_reply INTEGER NOT NULL,
                    author_id TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    author_username TEXT NOT NULL,
                    author_url TEXT NOT NULL,
                    author_followers INTEGER NOT NULL,
                    author_following INTEGER NOT NULL,
                    hashtags TEXT NOT NULL,
                    urls TEXT NOT NULL,
                    mentions TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tweets_search_date ON tweets(search_date)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tweets_id ON tweets(id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_scraped_days_date ON scraped_days(scrape_date)"
            )
            connection.commit()

    def _migrate_tweets_table_if_needed(self, connection: sqlite3.Connection) -> None:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tweets'"
        ).fetchone()
        if not table_exists:
            return

        columns = connection.execute("PRAGMA table_info(tweets)").fetchall()
        has_row_id = any(column[1] == "row_id" for column in columns)
        if has_row_id:
            return

        connection.execute("DROP INDEX IF EXISTS idx_tweets_search_date")
        connection.execute("DROP INDEX IF EXISTS idx_tweets_id")
        connection.execute(
            """
            CREATE TABLE tweets_new (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_date TEXT NOT NULL,
                id TEXT NOT NULL,
                url TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                lang TEXT NOT NULL,
                retweet_count INTEGER NOT NULL,
                reply_count INTEGER NOT NULL,
                like_count INTEGER NOT NULL,
                quote_count INTEGER NOT NULL,
                view_count INTEGER NOT NULL,
                bookmark_count INTEGER NOT NULL,
                conversation_id TEXT NOT NULL,
                in_reply_to_id TEXT NOT NULL,
                in_reply_to_user_id TEXT NOT NULL,
                in_reply_to_username TEXT NOT NULL,
                is_reply INTEGER NOT NULL,
                is_limited_reply INTEGER NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_username TEXT NOT NULL,
                author_url TEXT NOT NULL,
                author_followers INTEGER NOT NULL,
                author_following INTEGER NOT NULL,
                hashtags TEXT NOT NULL,
                urls TEXT NOT NULL,
                mentions TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tweets_new (
                search_date,
                id,
                url,
                text,
                source,
                created_at,
                lang,
                retweet_count,
                reply_count,
                like_count,
                quote_count,
                view_count,
                bookmark_count,
                conversation_id,
                in_reply_to_id,
                in_reply_to_user_id,
                in_reply_to_username,
                is_reply,
                is_limited_reply,
                author_id,
                author_name,
                author_username,
                author_url,
                author_followers,
                author_following,
                hashtags,
                urls,
                mentions
            )
            SELECT
                search_date,
                id,
                url,
                text,
                source,
                created_at,
                lang,
                retweet_count,
                reply_count,
                like_count,
                quote_count,
                view_count,
                bookmark_count,
                conversation_id,
                in_reply_to_id,
                in_reply_to_user_id,
                in_reply_to_username,
                is_reply,
                is_limited_reply,
                author_id,
                author_name,
                author_username,
                author_url,
                author_followers,
                author_following,
                hashtags,
                urls,
                mentions
            FROM tweets
            """
        )
        connection.execute("DROP TABLE tweets")
        connection.execute("ALTER TABLE tweets_new RENAME TO tweets")

    def _load_known_completed_dates(
        self,
        connection: sqlite3.Connection,
        legacy_checkpoint_path: Path | None,
    ) -> set[date]:
        completed_dates: set[date] = set()

        rows = connection.execute("SELECT DISTINCT search_date FROM tweets").fetchall()
        for row in rows:
            parsed = self._parse_date(row["search_date"])
            if parsed is not None:
                completed_dates.add(parsed)

        completed_dates.update(self._load_csv_dates())

        checkpoint_last_date = self._read_legacy_checkpoint(legacy_checkpoint_path)
        if checkpoint_last_date:
            completed_dates.add(checkpoint_last_date)

        return completed_dates

    def _load_csv_dates(self) -> set[date]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return set()

        completed_dates: set[date] = set()
        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                parsed = self._parse_date(row.get("search_date", ""))
                if parsed is not None:
                    completed_dates.add(parsed)

        return completed_dates

    def _read_legacy_checkpoint(self, checkpoint_path: Path | None) -> date | None:
        if checkpoint_path is None or not checkpoint_path.exists():
            return None

        raw_value = checkpoint_path.read_text(encoding="utf-8").strip()
        if not raw_value:
            return None

        if raw_value.startswith(LEGACY_CHECKPOINT_PREFIX):
            raw_value = raw_value.split("=", maxsplit=1)[1]

        return self._parse_date(raw_value)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None

        return date.fromisoformat(value)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")
