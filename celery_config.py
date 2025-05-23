# celery_config.py
import os
from celery import Celery
from dotenv import load_dotenv
load_dotenv()

# Настройка Celery
celery_app = Celery(
    'recommendations',
    broker=f'redis://{os.getenv("REDIS_HOST", "localhost")}:{os.getenv("REDIS_PORT", 6379)}/{os.getenv("REDIS_DB", 0)}',
    backend=f'redis://{os.getenv("REDIS_HOST", "localhost")}:{os.getenv("REDIS_PORT", 6379)}/{os.getenv("REDIS_DB", 0)}'
)

# Регистрация задач из модуля main
celery_app.autodiscover_tasks(['main'], force=True)

# Настройки (опционально)
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)