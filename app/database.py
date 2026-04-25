from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS bot_users (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE NOT NULL,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    language_code TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_requests (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL,
    file_type TEXT NOT NULL,
    file_name TEXT,
    file_size BIGINT,
    verdict TEXT,
    faces_count INT,
    summary_text TEXT,
    error_text TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT fk_analysis_user
        FOREIGN KEY (telegram_id)
        REFERENCES bot_users (telegram_id)
);
"""


@dataclass(frozen=True)
class UserRecord:
    telegram_id: int
    username: str | None
    first_name: str | None
    last_name: str | None
    language_code: str | None


@dataclass(frozen=True)
class AnalysisRequestRecord:
    telegram_id: int
    file_type: str
    file_name: str | None = None
    file_size: int | None = None
    verdict: str | None = None
    faces_count: int | None = None
    summary_text: str | None = None
    error_text: str | None = None


class Database:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)

    def create_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(CREATE_TABLES_SQL))

    def upsert_user(self, user: UserRecord) -> None:
        query = text(
            """
            INSERT INTO bot_users (
                telegram_id,
                username,
                first_name,
                last_name,
                language_code
            )
            VALUES (
                :telegram_id,
                :username,
                :first_name,
                :last_name,
                :language_code
            )
            ON CONFLICT (telegram_id) DO UPDATE SET
                username = EXCLUDED.username,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                language_code = EXCLUDED.language_code,
                updated_at = NOW()
            """
        )
        with self.engine.begin() as conn:
            conn.execute(query, asdict(user))

    def insert_analysis_request(self, request: AnalysisRequestRecord) -> None:
        query = text(
            """
            INSERT INTO analysis_requests (
                telegram_id,
                file_type,
                file_name,
                file_size,
                verdict,
                faces_count,
                summary_text,
                error_text
            )
            VALUES (
                :telegram_id,
                :file_type,
                :file_name,
                :file_size,
                :verdict,
                :faces_count,
                :summary_text,
                :error_text
            )
            """
        )
        with self.engine.begin() as conn:
            conn.execute(query, asdict(request))


def safe_upsert_user(db: Database | None, user: UserRecord) -> None:
    if db is None:
        return

    try:
        db.upsert_user(user)
    except Exception:
        logger.exception("Не удалось записать пользователя в БД")


def safe_insert_analysis_request(db: Database | None, request: AnalysisRequestRecord) -> None:
    if db is None:
        return

    try:
        db.insert_analysis_request(request)
    except Exception:
        logger.exception("Не удалось записать запрос анализа в БД")
