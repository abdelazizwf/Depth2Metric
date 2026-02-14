FROM python:3.12.10-slim

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/apt \
    apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv

RUN mkdir /app && mkdir -p /app/models/torch
WORKDIR /app

COPY .python-version pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY static ./static

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

EXPOSE 80

CMD ["uv", "run", "uvicorn", "src.depth2metric.main:app", "--forwarded-allow-ips=*", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
