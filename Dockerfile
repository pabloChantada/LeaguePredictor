FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

# copy dependencies files
COPY pyproject.toml uv.lock /app/

# use cache and --no-dev depencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# copy the code to the root 
COPY /src /app/src

# uv will automatically install the dependencies from uv.lock
# start both the app and the live dashboard (dependent)
CMD ["uv", "run", "serve:app", "serve:live-dashboard"]