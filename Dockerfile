FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src ./src

EXPOSE 8000 8501

CMD ["bash", "-c", \
    "uv run uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 & \
     uv run streamlit run src/serve/live_dashboard.py --server.address 0.0.0.0 --server.port 8501"]