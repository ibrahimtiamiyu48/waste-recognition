# SortVision waste-recognition API (Lab 07)
# Image = code + Python + packages. Model weights, config and evidence come from the VM at runtime:
#   models/  -> mounted read-only (trained by src/baseline_pipeline.py on the VM)
#   reports/ -> mounted read-write (JSONL evidence survives container replacement)
#   .env     -> passed with --env-file
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service files only (see .dockerignore)
COPY src/ src/
COPY config/ config/
COPY tests/ tests/
COPY examples/ examples/

# Non-root user; UID 1000 matches the first user on the Ubuntu VM, so it can write to the mounted reports/
ARG APP_UID=1000
RUN useradd --uid ${APP_UID} --create-home appuser \
 && mkdir -p reports models \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
