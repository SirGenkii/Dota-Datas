FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements_web.txt /app/requirements_web.txt
RUN pip install --no-cache-dir -r /app/requirements_web.txt

COPY . /app

ENV PYTHONPATH=/app

