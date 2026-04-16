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
from app.services.deepfake_service import DeepfakeService
from app.utils.files import safe_suffix, unique_file_path

logger = logging.getLogger(__name__)


class TelegramFileCarrier(Protocol):
    file_size: int | None
    file_name: str | None

    async def get_file(self): ...


class BotHandlers:
    def __init__(self, settings: Settings, service: DeepfakeService) -> None:
        self.settings = settings
        self.service = service

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text(START_MESSAGE)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text(HELP_MESSAGE)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return

        photo = update.message.photo[-1]
        file_info = await photo.get_file()

        temp_path = unique_file_path(self.settings.temp_dir, ".jpg")
        await file_info.download_to_drive(custom_path=str(temp_path))

        await self._analyze_and_reply(update, temp_path)

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

        max_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if media.file_size and media.file_size > max_bytes:
            await update.message.reply_text(TOO_LARGE_MESSAGE)
            return

        suffix = safe_suffix(media.file_name, default_suffix)
        if suffix not in (self.service.IMAGE_EXTS | self.service.VIDEO_EXTS):
            await update.message.reply_text(UNSUPPORTED_MESSAGE)
            return

        file_info = await media.get_file()
        temp_path = unique_file_path(self.settings.temp_dir, suffix)
        await file_info.download_to_drive(custom_path=str(temp_path))

        await self._analyze_and_reply(update, temp_path)

    async def _analyze_and_reply(self, update: Update, input_path: Path) -> None:
        assert update.message is not None

        wait_message = await update.message.reply_text("Файл получен. Начинаю анализ...")

        try:
            result = await asyncio.to_thread(self.service.analyze, input_path)

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
            await wait_message.edit_text(f"Произошла ошибка при обработке файла:\n{e}")
        finally:
            try:
                if input_path.exists():
                    input_path.unlink()
            except Exception:
                logger.warning("Не удалось удалить временный файл: %s", input_path)
