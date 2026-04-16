from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    bot_token: str
    temp_dir: Path
    results_dir: Path
    models_dir: Path
    log_level: str
    max_file_size_mb: int
    video_frame_skip: int

    @property
    def face_model_path(self) -> Path:
        return self.models_dir / "FaceDetector.onnx"

    @property
    def face_data_path(self) -> Path:
        return self.models_dir / "FaceDetector.data"

    @property
    def deepfake_model_path(self) -> Path:
        return self.models_dir / "deepfake.onnx"

    @property
    def deepfake_data_path(self) -> Path:
        return self.models_dir / "deepfake.data"


def get_settings() -> Settings:
    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise ValueError("Не найден BOT_TOKEN в .env")

    temp_dir = Path(os.getenv("TEMP_DIR", "./temp")).resolve()
    results_dir = Path(os.getenv("RESULTS_DIR", "./results")).resolve()
    models_dir = Path(os.getenv("MODELS_DIR", "./models")).resolve()

    temp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        bot_token=bot_token,
        temp_dir=temp_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "20")),
        video_frame_skip=int(os.getenv("VIDEO_FRAME_SKIP", "5")),
    )