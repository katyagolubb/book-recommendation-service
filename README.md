# Book Recommendation Service
This is a book recommendation service built with FastAPI, Celery, and Redis. It provides personalized book recommendations based on collaborative and content-based filtering techniques.
Features

## Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Redis server

## Installation
1. Clone the Repository
```
git clone https://github.com/yourusername/book-recommendation-service.git
cd book-recommendation-service
```
2. Create and Configure .env

3. Install Dependencies
Install the required Python packages:
```
pip install -r requirements.txt
```
4. Run Redis
Ensure a Redis server is running locally or via Docker:
```
docker run -d -p 6379:6379 redis:7.0
```
5. Run the Application
**Without Docker**
Start the FastAPI server:
```
uvicorn main:app --host 0.0.0.0 --port 8002
```
Start the Celery worker in a separate terminal:
```
celery -A celery_config.celery_app worker --loglevel=info
```
Start the Celery Beat scheduler for periodic tasks in another terminal:
```
celery -A celery_config.celery_app beat --loglevel=info
```
**With Docker**
Build and run the application with docker-compose:
```
docker-compose up --build
```
## Usage
1. Precompute Matrices
   - Method: GET  
   - URL: /api/precompute/  
   - Description: Triggers the precomputation of user-book and TF-IDF matrices. Requires a valid JWT token in the Authorization header.  
   - Headers: Authorization: Bearer your_jwt_token
   - Example Response (200):  
       ```json
       {
           "message": "Precomputation completed successfully"
       }
       ```
2. Get Recommendations for a User
   - Method: GET  
   - URL: /api/recommendations/?user_id=<user_id>  
   - Description: Fetches personalized book recommendations for a specified user. Requires a valid JWT token in the Authorization header. The user_id query parameter is required.  
   - Headers: Authorization: Bearer your_jwt_token
   - Example Response (200):  
       ```json
       [
           {
               "book_id": 1,
               "name": "The Hobbit",
               "author": "J.R.R. Tolkien",
               "genres": "Fantasy, Adventure"
           },
           {
               "book_id": 2,
               "name": "1984",
               "author": "George Orwell",
               "genres": "Dystopia, Fiction"
           }
       ]
       ```