# Reproducibility container (proposal Sec. 3.5). Build:
#   docker build -t eilj2 .
# Run the tests:
#   docker run --rm eilj2
# Regenerate figures (mount the repo to keep outputs):
#   docker run --rm -v %cd%:/work eilj2 python scripts/make_all.py --phase figures

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /work
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev || uv sync --no-dev

COPY . .
RUN uv sync --frozen || uv sync

CMD ["uv", "run", "pytest", "-q", "-m", "not slow"]
