# One container: the Flask app served by gunicorn, with freshly trained
# model artifacts baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the models during the build so the running app never trains.
# The dataset is in data/raw/ (CC0 licence), so no network access is needed.
RUN python train_model.py --quiet

# Run as a non-root user. data/live is where live listings are written;
# on ECS an EFS volume is mounted there.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/live \
    && chown -R appuser:appuser /app/data/live
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

# One worker keeps memory low (TensorFlow is large) and makes the optional
# background refresh run once; threads handle concurrent visitors.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 60 --access-logfile - app:app"]
