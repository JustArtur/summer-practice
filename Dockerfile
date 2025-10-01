# Dockerfile для контейнера с обученной MNIST моделью
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY models/ ./models/

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Проверка наличия модели
RUN test -f models/model.joblib || (echo "Model file not found!" && exit 1)
RUN test -f models/metrics.json || (echo "Metrics file not found!" && exit 1)

# Переменные окружения
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.joblib
ENV METRICS_PATH=/app/models/metrics.json

# Порт для веб-сервиса (если понадобится)
EXPOSE 8000

# Команда по умолчанию - показать метрики модели
CMD ["python", "-c", "import json; print('Model metrics:'); print(json.dumps(json.load(open('models/metrics.json')), indent=2))"]
