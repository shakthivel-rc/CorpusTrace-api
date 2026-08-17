# CorpusTrace API image.
#
# Two stages because `mysqlclient` is a C extension: it needs a compiler and the MySQL
# client headers to build, and neither is worth carrying into the running image. The
# builder produces a virtualenv; the runtime stage copies it and installs only the shared
# library the extension actually links against.

# ---------------------------------------------------------------------------- builder ---
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        default-libmysqlclient-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied on its own so a source-only change does not reinstall 29 pinned packages.
COPY requirements.txt requirements-dev.txt ./
# --retries/--timeout are not decoration. pip defaults to 5 retries at a 15 s timeout, and
# one dropped connection anywhere in 31 packages fails the whole build with
#   ERROR: Could not find a version that satisfies the requirement passlib==1.7.4
#          (from versions: none)
# which names a package and a pin, reads exactly like a bad requirement, and sends the
# reader to PyPI to check a version that was never the problem. The index page simply did
# not arrive. Two minutes of build are thrown away for a blip that a retry would have
# absorbed, and on a proxied or corporate network that blip is not rare.
RUN pip install --no-cache-dir --upgrade pip --retries 10 --timeout 60 \
    && pip install --no-cache-dir --retries 10 --timeout 60 -r requirements.txt

# ---------------------------------------------------------------------------- runtime ---
FROM python:3.12-slim

# libmariadb3 is the runtime half of default-libmysqlclient-dev; curl is what the compose
# healthcheck uses to decide the API is actually serving rather than merely started.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmariadb3 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# WORKDIR is load-bearing, not cosmetic: `load_dotenv()` reads the *current working
# directory* and `get_settings()` runs at import time, so starting anywhere else makes the
# process abort before uvicorn binds. See CLAUDE.md §14.
WORKDIR /app

COPY . .

# Uploaded documents live here. Declared so a bind mount or named volume has something to
# attach to even on the first run, before anything has been uploaded.
RUN mkdir -p /app/uploads/rag

# Runs as a non-root user; the uploads directory is the only path the app writes to.
RUN useradd --create-home --uid 10001 corpustrace \
    && chown -R corpustrace:corpustrace /app/uploads
USER corpustrace

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/health/ready || exit 1

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
