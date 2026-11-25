# FoundIt — production image
#
# Two stages so the runtime image carries no build toolchain. ffmpeg is a
# runtime dependency (video decoding), not a build one, which is why it is
# installed in the final stage rather than copied across.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ffmpeg/ffprobe: required by the ingest path. curl: healthcheck only.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY schemas ./schemas
COPY fixtures ./fixtures
COPY tests ./tests
COPY tools ./tools

# The database lives on a mounted volume in production. Creating the
# directory here means the app works even before a volume is attached.
RUN mkdir -p /data

ENV DATABASE_PATH=/data/foundit.db \
    VENUE_TIMEZONE=Asia/Riyadh \
    PORT=8000

EXPOSE 8000

# Railway injects $PORT. Docker needs the shell form to expand it, so this
# uses exec via sh rather than the JSON array form.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

# --- optional: tests run in CI, not on boot -------------------------------
# Keep the test deps out of the runtime image by overriding:
#   docker build --target base -t foundit:test .
FROM base AS test
RUN pip install -r requirements-dev.txt
CMD ["python", "-m", "pytest", "tests/", "-q"]
