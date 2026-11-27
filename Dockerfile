# FoundIt — production image
#
# Stage order matters. `docker build` with no --target (which is what
# Railway runs) builds the LAST stage in this file. So the runtime stage
# must be last; the optional test stage sits above it and is skipped by
# BuildKit unless you ask for it with --target test.

# --- base: system deps, Python deps, app code -----------------------------
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
COPY tools ./tools
COPY prompts ./prompts

# The database lives on a mounted volume in production. Creating the
# directory here means the app works even before a volume is attached.
RUN mkdir -p /data

ENV DATABASE_PATH=/data/foundit.db \
    VENUE_TIMEZONE=Asia/Riyadh \
    PORT=8000

# --- test: optional, never deployed ----------------------------------------
#   docker build --target test -t foundit:test . && docker run --rm foundit:test
FROM base AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY tests ./tests
CMD ["python", "-m", "pytest", "tests/", "-q"]

# --- runtime: what Railway runs. Must stay the last stage. ----------------
FROM base AS runtime

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Shell form on purpose: Railway injects $PORT and only a shell expands it.
CMD uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
