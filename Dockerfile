FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv export --no-dev --no-emit-project -o req.txt \
    && pip install --no-cache-dir -r req.txt

COPY arena ./arena
COPY results/arena.db ./results/arena.db
ENV ARENA_DB=/app/results/arena.db

EXPOSE 8080
# PORT is set by Render; Fly uses the 8080 default (fly.toml internal_port)
CMD uvicorn arena.web.app:app --host 0.0.0.0 --port ${PORT:-8080}
