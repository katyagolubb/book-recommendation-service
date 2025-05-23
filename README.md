Book Recommendation Service
This is a book recommendation service built with FastAPI, Celery, and Redis. It provides personalized book recommendations based on collaborative and content-based filtering techniques.
Features

Precompute user-book matrices and TF-IDF models for efficient recommendations.
Asynchronous API endpoints for fetching user books and computing recommendations.
Caching with Redis to improve performance.
Support for JWT authentication.

Prerequisites

Python 3.11+
Docker (optional, for containerized deployment)
Redis server

Installation
1. Clone the Repository
git clone https://github.com/yourusername/book-recommendation-service.git
cd book-recommendation-service

2. Create and Configure .env
Copy the .env.example file to .env and update the values:
cp .env.example .env

Edit .env with your configuration (e.g., JWT_SECRET_KEY, REDIS_HOST, etc.).
3. Install Dependencies
Install the required Python packages:
pip install -r requirements.txt

4. Run Redis
Ensure a Redis server is running locally or via Docker:
docker run -d -p 6379:6379 redis:7.0

5. Run the Application
Without Docker
Start the FastAPI server:
uvicorn main:app --host 0.0.0.0 --port 8002

Start the Celery worker in a separate terminal:
celery -A celery_config.celery_app worker --loglevel=info

With Docker
Build and run the application with docker-compose:
docker-compose up --build

6. Precompute Data
Run the precompute endpoint to generate the initial models:
curl -X GET "http://localhost:8002/api/precompute/" -H "Authorization: Bearer your_jwt_token"

Usage

Get recommendations for a user:curl -X GET "http://localhost:8002/api/recommendations/?user_id=1" -H "Authorization: Bearer your_jwt_token"



Environment Variables

JWT_SECRET_KEY: Secret key for JWT encoding/decoding.
JWT_ALGORITHM: Algorithm for JWT (default: HS256).
REDIS_HOST: Redis host (default: localhost).
REDIS_PORT: Redis port (default: 6379).
REDIS_DB: Redis database (default: 0).
BASE_API_URL: Base URL of the API (default: http://localhost:8000/api).
APP_PORT: Application port (default: 8002).

Contributing
Feel free to submit issues or pull requests. Please follow the existing code style.
License
MIT License (or specify your preferred license)
