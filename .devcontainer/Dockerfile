FROM python:3.12-slim AS dev_containers_target_stage

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 curl procps git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras --dev

COPY . .

ENTRYPOINT ["uv", "run"]
CMD ["pytest"]
