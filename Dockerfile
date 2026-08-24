FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src ./src

EXPOSE 8000

CMD ["bash", "-c", \
    "uv run uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]