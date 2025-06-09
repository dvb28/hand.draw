FROM python:3.12-slim

LABEL maintainer="Hand Draw Project"

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DISPLAY=:0

EXPOSE 6006

CMD ["python", "main.py"]
