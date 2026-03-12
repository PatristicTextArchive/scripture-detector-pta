# ── Scripture Detector ────────────────────────────────────────────────────────
#
# Build:
#   docker build -t scripture-detector .
#
# Run (ephemeral in-memory database — data resets on container restart):
#   docker run -p 5001:5001 scripture-detector
#
# Run (persistent database — mount a host directory):
#   docker run -p 5001:5001 \
#     -v "$(pwd)/data_volume:/app/db" \
#     -e SD_DB_DIR=/app/db \
#     -e GEMINI_API_KEY=your_key_here \
#     scripture-detector python app.py
#
# The default CMD uses --cache-db (in-memory SQLite) which is ideal for
# workshops and demos where persistence across restarts is not needed.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Install git (required by Hugging Face Spaces build system) and uv
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app


# Copy dependency manifest first to leverage Docker layer caching.
# Dependencies are only re-installed when pyproject.toml / uv.lock changes.
COPY pyproject.toml uv.lock* ./

# Sync dependencies (no dev extras)
RUN uv sync --no-dev --frozen || uv sync --no-dev

# Copy application source
COPY . .

# Bind to all interfaces inside the container
ENV SD_HOST=0.0.0.0
ENV SD_PORT=7860

# Expose the Flask port
EXPOSE 7860

# Default: run with --cache-db (in-memory database, no volume needed).
# To use a persistent database instead, override CMD and set SD_DB_DIR.
CMD ["uv", "run", "python", "app.py", "--cache-db"]
