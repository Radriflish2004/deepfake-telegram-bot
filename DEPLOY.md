# Deploy Guide

## Recommended Option

For this project, the simplest and most predictable deployment is a small Linux VPS with Docker.

Why this fits well:

- The bot is a long-running polling process, not a web app.
- It uses local ONNX model files from `models/`.
- It writes temporary and result files to disk.
- The repository already includes `Dockerfile` and `docker-compose.yml`.

Good VPS providers for this kind of workload:

- Hetzner Cloud
- DigitalOcean
- Vultr

Recommended baseline server:

- Ubuntu 22.04 or 24.04
- 2 vCPU
- 4 GB RAM
- 20+ GB SSD

## Deploy To A VPS

### 1. Prepare the server

Connect to the server:

```bash
ssh root@YOUR_SERVER_IP
```

Install Docker and Compose plugin:

```bash
apt update
apt install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable docker
systemctl start docker
```

### 2. Copy the project

Upload the project to the server in any convenient way.

If you use git:

```bash
git clone YOUR_REPOSITORY_URL deepfake-telegram-bot
cd deepfake-telegram-bot
```

If you do not use git yet, copy the folder with `scp` or via your IDE.

### 3. Configure environment variables

Create `.env`:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
BOT_TOKEN=your_real_bot_token
TEMP_DIR=./temp
RESULTS_DIR=./results
MODELS_DIR=./models
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
VIDEO_FRAME_SKIP=5
```

### 4. Verify model files

Make sure the server has:

- `models/FaceDetector.onnx`
- `models/FaceDetector.data`
- `models/deepfake.onnx`
- `models/deepfake.data`

### 5. Start the bot

```bash
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f
```

Stop the bot:

```bash
docker compose down
```

Restart after changes:

```bash
docker compose up -d --build
```

## Persistent Data

Current `docker-compose.yml` mounts:

- `./models` as read-only
- `./temp` for temporary files
- `./results` for generated output

This means your bot data stays on disk on the server between container restarts.

## Managed Platform Alternatives

### Railway

Works if you deploy as a long-running service using Docker.

Notes:

- Attach a volume if you want `temp/` or `results/` persisted.
- If you need persistent relative paths, mount the volume under `/app/...`.
- Ensure your model files are included in the image or mounted appropriately.

### Render

Prefer a Background Worker rather than a Web Service, because this bot uses polling and does not expose HTTP endpoints.

Notes:

- If you need files to persist, use a persistent disk.
- Make sure the start command runs the bot process continuously.

### Fly.io

Can work well with Docker plus a mounted volume, but it is a bit more infrastructure-heavy than a basic VPS.

## Important Notes

- Do not run the same polling bot instance twice with the same token.
- Keep `.env` private and never commit it.
- If you later want zero-downtime deploys or monitoring, we can add:
  - `watchtower` for automatic image updates
  - log rotation
  - a reverse proxy and webhook mode
  - GitHub Actions CI/CD
