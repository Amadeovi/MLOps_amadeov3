# syntax=docker/dockerfile:1

# 🔧 Base Python image
# Build stage: install dependencies and the churn package.
FROM python:3.15-slim AS build
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 👨‍💻 Set working directory
WORKDIR /app

# 📦 Copy project files
# Pinned, not `:latest` — a floating tag is invisible to Dependabot's docker ecosystem.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-editable



# Runtime stage: run the API as a non-root user.
FROM python:3.15-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# LightGBM requires the OpenMP runtime on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r -g 10001 app \
    && useradd -r -u 10001 -g app -m app

COPY --from=build --chown=10001:10001 /app/.venv /app/.venv

USER 10001:10001

# 🚀 Run API
EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]