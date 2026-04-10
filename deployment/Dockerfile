# --- Step 1 : Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installing in a specified folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Step 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Installation of curl for Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Taking back libraries
COPY --from=builder /install /usr/local

# Copy of code and artifacts
COPY ./app ./app
COPY ./src ./src
COPY ./config ./config
#COPY ./artifacts ./artifacts

# Sécurity: non root user
RUN adduser --disabled-password appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]