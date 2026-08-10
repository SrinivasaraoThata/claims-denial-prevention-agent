FROM python:3.11-slim

WORKDIR /app

# xgboost's shared library needs libgomp (OpenMP runtime), not present in
# the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ agents/
COPY api/ api/
COPY data/ data/
COPY models/ models/

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
