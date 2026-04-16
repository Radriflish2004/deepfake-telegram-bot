# Deepfake Detection Telegram Bot

Telegram-бот для детекции дипфейков на фото и видео. Использует ONNX-модели для обнаружения лиц и их классификации (Real / Fake).

## Как это работает

1. Пользователь отправляет боту фото или видеофайл
2. Бот обнаруживает лица через `FaceDetector` (ONNX)
3. Каждое лицо классифицируется через `DeepfakeClassifier` (ONNX)
4. Бот возвращает аннотированное изображение/видео и текстовый отчёт

```
User → Telegram API → BotHandlers → DeepfakeService → FaceDetector + DeepfakeClassifier → Result
```

## Поддерживаемые форматы

- Изображения: `jpg`, `jpeg`, `png`, `bmp`, `tiff`, `webp`
- Видео: `mp4`, `avi`, `mov`, `mkv`, `wmv`, `flv`, `webm`

## Требования

- Python 3.11+
- Файлы ONNX-моделей в папке `models/`:
  - `FaceDetector.onnx` + `FaceDetector.data`
  - `deepfake.onnx` + `deepfake.onnx.data`

## Запуск локально

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Создать .env из примера и вписать токен
cp .env.example .env

# 4. Запустить
python run.py
```

## Деплой на сервер через Docker

```bash
# 1. Скопировать .env и вписать токен
cp .env.example .env

# 2. Собрать и запустить
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## Конфигурация (.env)

| Переменная         | По умолчанию | Описание                              |
|--------------------|--------------|---------------------------------------|
| `BOT_TOKEN`        | —            | Токен бота от @BotFather (обязателен) |
| `TEMP_DIR`         | `./temp`     | Папка для временных файлов            |
| `RESULTS_DIR`      | `./results`  | Папка для результатов                 |
| `MODELS_DIR`       | `./models`   | Папка с ONNX-моделями                 |
| `LOG_LEVEL`        | `INFO`       | Уровень логирования                   |
| `MAX_FILE_SIZE_MB` | `20`         | Максимальный размер файла (МБ)        |
| `VIDEO_FRAME_SKIP` | `5`          | Анализировать каждый N-й кадр видео   |

## Структура проекта

```
deepfake-telegram-bot/
├── app/
│   ├── bot/
│   │   ├── handlers.py          # Обработчики сообщений Telegram
│   │   └── messages.py          # Тексты сообщений бота
│   ├── services/
│   │   └── deepfake_service.py  # ML-пайплайн: детекция + классификация
│   ├── utils/
│   │   └── files.py             # Утилиты для работы с файлами
│   ├── config.py                # Конфигурация через .env
│   └── main.py                  # Инициализация и запуск бота
├── models/                      # ONNX-модели
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py                       # Точка входа
```
