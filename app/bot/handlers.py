from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol

from telegram import Update
from telegram.ext import ContextTypes

from app.bot.messages import (
    HELP_MESSAGE,
    NO_FACE_MESSAGE,
    START_MESSAGE,
    TOO_LARGE_MESSAGE,
    UNSUPPORTED_MESSAGE,
)
from app.config import Settings
from app.database import (
    AnalysisRequestRecord,
    Database,
    UserRecord,
    safe_insert_analysis_request,
    safe_upsert_user,
)
from app.services.deepfake_service import DeepfakeService
from app.utils.files import safe_suffix, unique_file_path

logger = logging.getLogger(__name__)


class TelegramFileCarrier(Protocol):
    file_size: int | None
    file_name: str | None

    async def get_file(self): ...


class BotHandlers:
    def __init__(self, settings: Settings, service: DeepfakeService, db: Database | None = None) -> None:
        self.settings = settings
        self.service = service
        self.db = db

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._store_user(update)
        if update.message:
            await update.message.reply_text(START_MESSAGE)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._store_user(update)
        if update.message:
            await update.message.reply_text(HELP_MESSAGE)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return

        photo = update.message.photo[-1]
        file_info = await photo.get_file()

        temp_path = unique_file_path(self.settings.temp_dir, ".jpg")
        await file_info.download_to_drive(custom_path=str(temp_path))

        await self._analyze_and_reply(
            update,
            temp_path,
            file_type="image",
            file_name=None,
            file_size=photo.file_size,
        )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document:
            return

        document = update.message.document
        await self._handle_media_file(update, document, default_suffix=".bin")

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.video:
            return

        await self._handle_media_file(update, update.message.video, default_suffix=".mp4")

    async def _handle_media_file(
        self,
        update: Update,
        media: TelegramFileCarrier,
        default_suffix: str,
    ) -> None:
        assert update.message is not None

        suffix = safe_suffix(media.file_name, default_suffix)
        file_type = self._file_type_from_suffix(suffix)

        max_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if media.file_size and media.file_size > max_bytes:
            await update.message.reply_text(TOO_LARGE_MESSAGE)
            await self._store_analysis_request(
                update,
                file_type=file_type,
                file_name=media.file_name,
                file_size=media.file_size,
                error_text=TOO_LARGE_MESSAGE,
            )
            return

        if suffix not in (self.service.IMAGE_EXTS | self.service.VIDEO_EXTS):
            await update.message.reply_text(UNSUPPORTED_MESSAGE)
            await self._store_analysis_request(
                update,
                file_type=file_type,
                file_name=media.file_name,
                file_size=media.file_size,
                error_text=UNSUPPORTED_MESSAGE,
            )
            return

        file_info = await media.get_file()
        temp_path = unique_file_path(self.settings.temp_dir, suffix)
        await file_info.download_to_drive(custom_path=str(temp_path))

        await self._analyze_and_reply(
            update,
            temp_path,
            file_type=file_type,
            file_name=media.file_name,
            file_size=media.file_size,
        )

    async def _analyze_and_reply(
        self,
        update: Update,
        input_path: Path,
        file_type: str,
        file_name: str | None,
        file_size: int | None,
    ) -> None:
        assert update.message is not None

        await self._store_user(update)
        wait_message = await update.message.reply_text("Файл получен. Начинаю анализ...")

        try:
            result = await asyncio.to_thread(self.service.analyze, input_path)
            await self._store_analysis_request(
                update,
                file_type=result.input_type or file_type,
                file_name=file_name,
                file_size=file_size,
                verdict=result.verdict,
                faces_count=result.faces_count,
                summary_text=result.summary_text,
            )

            if result.faces_count == 0:
                await wait_message.edit_text(NO_FACE_MESSAGE)
                return

            verdict_map = {
                "fake": "⚠️ Итог: вероятно deepfake",
                "real": "✅ Итог: вероятно настоящее",
                "no_faces": "Лица не обнаружены",
            }

            text = f"{verdict_map.get(result.verdict, 'Результат готов')}\n\n{result.summary_text}"

            await wait_message.edit_text(text)

            if result.output_path and result.output_path.exists():
                if result.input_type == "image":
                    with open(result.output_path, "rb") as f:
                        await update.message.reply_photo(photo=f)
                elif result.input_type == "video":
                    with open(result.output_path, "rb") as f:
                        await update.message.reply_video(video=f)

        except Exception as e:
            logger.exception("Ошибка при анализе файла")
            await self._store_analysis_request(
                update,
                file_type=file_type,
                file_name=file_name,
                file_size=file_size,
                error_text=str(e),
            )
            await wait_message.edit_text(f"Произошла ошибка при обработке файла:\n{e}")
        finally:
            try:
                if input_path.exists():
                    input_path.unlink()
            except Exception:
                logger.warning("Не удалось удалить временный файл: %s", input_path)

    async def _store_user(self, update: Update) -> None:
        user = update.effective_user
        if not user:
            return

        record = UserRecord(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            language_code=user.language_code,
        )
        await asyncio.to_thread(safe_upsert_user, self.db, record)

    async def _store_analysis_request(
        self,
        update: Update,
        file_type: str,
        file_name: str | None = None,
        file_size: int | None = None,
        verdict: str | None = None,
        faces_count: int | None = None,
        summary_text: str | None = None,
        error_text: str | None = None,
    ) -> None:
        user = update.effective_user
        if not user:
            return

        await self._store_user(update)
        record = AnalysisRequestRecord(
            telegram_id=user.id,
            file_type=file_type,
            file_name=file_name,
            file_size=file_size,
            verdict=verdict,
            faces_count=faces_count,
            summary_text=summary_text,
            error_text=error_text,
        )
        await asyncio.to_thread(safe_insert_analysis_request, self.db, record)

    def _file_type_from_suffix(self, suffix: str) -> str:
        if suffix in self.service.IMAGE_EXTS:
            return "image"
        if suffix in self.service.VIDEO_EXTS:
            return "video"
        return "unknown"
