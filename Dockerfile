# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY . .

# Открываем порт для FastAPI
EXPOSE 8002

# Команда для запуска FastAPI и Celery worker
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8002 & celery -A celery_config.celery_app worker --loglevel=info"]