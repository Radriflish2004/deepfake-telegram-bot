from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from app.bot.handlers import BotHandlers
from app.config import get_settings
from app.services.deepfake_service import DeepfakeService


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level, logging.INFO),
    )


def run() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    service = DeepfakeService(
        face_model_path=settings.face_model_path,
        deepfake_model_path=settings.deepfake_model_path,
        deepfake_data_path=settings.deepfake_data_path,
        results_dir=settings.results_dir,
        frame_skip=settings.video_frame_skip,
    )

    handlers = BotHandlers(settings=settings, service=service)

    app = Application.builder().token(settings.bot_token).build()

    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(CommandHandler("help", handlers.help_command))

    app.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handlers.handle_video))
    app.add_handler(
        MessageHandler(
            filters.Document.IMAGE | filters.Document.VIDEO,
            handlers.handle_document,
        )
    )

    app.run_polling(allowed_updates=[])


if __name__ == "__main__":
    run()
