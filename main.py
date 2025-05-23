import asyncio
import os
from dotenv import load_dotenv
import joblib
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from typing import List, Dict
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import defaultdict
import redis.asyncio as aioredis
import redis
import json
from joblib import dump, load
from celery_config import celery_app
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer

# Загружаем переменные окружения из .env
load_dotenv()

app = FastAPI()

# Получаем переменные окружения
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "mnfeu!dcp^(-khdb8!xt32&9_1f10n1rz=j!h64adrlwb8-_gg")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
BASE_API_URL = os.getenv("BASE_API_URL", "http://localhost:8000/api")
APP_PORT = int(os.getenv("APP_PORT", 8002))

# Асинхронный Redis-клиент для FastAPI
redis_client_async = aioredis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

# Синхронный Redis-клиент для Celery
redis_client_sync = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

# Зависимость для получения Bearer Token
security = HTTPBearer()

async def get_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials.scheme == "Bearer":
        raise HTTPException(status_code=401, detail="Bearer token is required")
    return credentials.credentials

async def get_current_user_id(token: str = Depends(get_token)) -> int:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token: user_id not found")
        return int(user_id)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Получение книг пользователя (асинхронная версия для FastAPI)
async def get_user_books(user_id: int, token: str) -> List[Dict]:
    cache_key = f"user_books:{user_id}"
    cached = await redis_client_async.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_API_URL}/books/list/?user_id={user_id}", headers=headers)
    if response.status_code != 200:
        return []
    books = response.json().get("results", [])
    await redis_client_async.setex(cache_key, 3600, json.dumps(books))
    return books

# Получение всех доступных книг (асинхронная версия для FastAPI)
async def get_all_books(token: str, user_id: int) -> List[Dict]:
    cache_key = f"all_books:{user_id}"
    cached = await redis_client_async.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"Bearer {token}"}
    all_books = []
    next_url = f"{BASE_API_URL}/books/search/?status=available"

    while next_url:
        response = requests.get(next_url, headers=headers)
        print(f"Response status: {response.status_code}, Text: {response.text}")
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch books")
        data = response.json()
        all_books.extend(data.get("results", []))
        next_url = data.get("next")

    await redis_client_async.setex(cache_key, 3600, json.dumps(all_books))
    return all_books

# Предварительный расчёт матриц
def precompute_matrices(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_API_URL}/books/list/", headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch all books")
    all_books = response.json().get("results", [])

    if not all_books:
        raise HTTPException(status_code=400, detail="No books available to precompute")

    print(f"All books received: {all_books}")
    users_books = defaultdict(list)
    user_book_ids = [book["user_book_id"] for book in all_books]
    print(f"User book IDs: {user_book_ids}")

    response = requests.post(f"{BASE_API_URL}/books/owners/", json={"user_book_ids": user_book_ids}, headers=headers)
    print(f"Owners response status: {response.status_code}, Text: {response.text}")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch book owners")

    owners = response.json()
    print(f"Owners: {owners}")
    for book in all_books:
        user_book_id = book["user_book_id"]
        book_id = book["book"]["book_id"]
        book_owner_id = owners.get(str(user_book_id))
        print(f"Processing book {user_book_id}, owner: {book_owner_id}")
        if book_owner_id:
            users_books[book_owner_id].append(book_id)

    user_ids = list(users_books.keys())
    book_ids = list(set(book_id for books in users_books.values() for book_id in books))
    print(f"User IDs: {user_ids}, Book IDs: {book_ids}")
    user_book_matrix = np.zeros((len(user_ids), len(book_ids)))

    for i, uid in enumerate(user_ids):
        for j, bid in enumerate(book_ids):
            if bid in users_books[uid]:
                user_book_matrix[i, j] = 1
    print(f"User book matrix:\n{user_book_matrix}")

    dump((user_ids, book_ids, user_book_matrix), "user_book_matrix.joblib")

    books_df = pd.DataFrame([book["book"] for book in all_books])
    print(f"Books DataFrame: {books_df}")
    books_df["genres_str"] = books_df["genres"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    books_df["text"] = books_df["genres_str"] + " " + books_df["overview"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books_df["text"])
    dump((books_df, tfidf, tfidf_matrix), "tfidf_matrix.joblib")

# Коллаборативная фильтрация
def collaborative_filtering(user_id: int, user_books: List[Dict]) -> List[int]:
    try:
        user_ids, book_ids, user_book_matrix = load("user_book_matrix.joblib")
        print(f"User IDs: {user_ids}")
        print(f"Book IDs: {book_ids}")
        print(f"User book matrix shape: {user_book_matrix.shape}")
    except FileNotFoundError:
        print("User book matrix file not found")
        return []

    if str(user_id) not in user_ids:
        print(f"User ID {user_id} not in user_ids")
        return []

    user_book_ids = {book["book"]["book_id"] for book in user_books}
    print(f"User book IDs in collaborative_filtering: {user_book_ids}")

    user_idx = user_ids.index(str(user_id))
    similarities = cosine_similarity([user_book_matrix[user_idx]], user_book_matrix)[0]
    print(f"Similarities: {similarities}")

    books_df = joblib.load("books_df.joblib")

    user_genres = set()
    for book in user_books:
        user_genres.update(book["book"].get("genres", ["Unknown"]))

    similar_users = []
    for idx, sim in enumerate(similarities):
        if user_ids[idx] == str(user_id):
            continue
        other_user_books = [book_ids[j] for j in range(len(book_ids)) if user_book_matrix[idx, j] == 1]
        other_user_genres = set()
        for book_id in other_user_books:
            book_row = books_df[books_df["book_id"] == book_id]
            if not book_row.empty:
                other_user_genres.update(book_row.iloc[0]["genres"])
        common_genres = user_genres.intersection(other_user_genres) - {"Unknown"}
        genre_bonus = 0.2 if common_genres else 0.0
        adjusted_sim = sim + genre_bonus
        similar_users.append((user_ids[idx], adjusted_sim))

    similar_users.sort(key=lambda x: x[1], reverse=True)
    print(f"Similar users: {similar_users}")

    recommended_book_ids = set()
    for similar_user_id, _ in similar_users[:3]:
        user_idx = user_ids.index(similar_user_id)
        user_books_set = set(book_ids[j] for j in range(len(book_ids)) if user_book_matrix[user_idx, j] == 1)
        user_books_set -= user_book_ids
        recommended_book_ids.update(user_books_set)

    print(f"Recommended book IDs: {recommended_book_ids}")
    return list(recommended_book_ids)

# Контентная фильтрация
def content_filtering(user_books: List[Dict], candidate_book_ids: List[int]) -> List[Dict]:
    tfidf_matrix = joblib.load("tfidf_matrix.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    books_df = joblib.load("books_df.joblib")

    candidate_book_ids = [book_id for book_id in candidate_book_ids if book_id in books_df["book_id"].values]
    if not candidate_book_ids:
        return []

    candidates = books_df[books_df["book_id"].isin(candidate_book_ids)]
    if candidates.empty:
        return []

    user_texts = [" ".join(book["book"].get("genres", ["Unknown"])) + " " + book["book"]["overview"] for book in user_books]
    user_tfidf = vectorizer.transform(user_texts)
    candidate_indices = [books_df.index[books_df["book_id"] == book_id].tolist()[0] for book_id in candidate_book_ids]
    candidate_tfidf = tfidf_matrix[candidate_indices]

    similarities = cosine_similarity(user_tfidf, candidate_tfidf).mean(axis=0)

    user_genres = set()
    for book in user_books:
        user_genres.update(book["book"].get("genres", ["Unknown"]))

    recommendations = []
    for idx, (book_id, similarity) in enumerate(zip(candidate_book_ids, similarities)):
        book_row = books_df[books_df["book_id"] == book_id].iloc[0]
        book_genres = set(book_row["genres"])
        common_genres = user_genres.intersection(book_genres)

        genre_penalty = 0.1 if "Unknown" in book_genres and user_genres - {"Unknown"} else 0.0
        adjusted_similarity = max(similarity - genre_penalty, 0.0)

        genre_bonus = 0.5 if common_genres and "Unknown" not in common_genres else 0.0
        adjusted_similarity = min(adjusted_similarity + genre_bonus, 1.0)

        reason = "Matches your interests"
        if common_genres:
            reason += f" (similarity: {adjusted_similarity:.2f}) - Genres: {', '.join(common_genres)}"
        elif "Unknown" in book_genres:
            reason += f" (similarity: {adjusted_similarity:.2f}) - Genres: Unknown"
        else:
            reason += f" (Explore a new genre: {', '.join(book_genres)})"

        recommendations.append({
            "book_id": int(book_id),
            "name": book_row["name"],
            "author": book_row["author"],
            "overview": book_row["overview"],
            "genres": book_row["genres"],
            "similarity": adjusted_similarity,
            "reason": reason,
        })

    recommendations.sort(key=lambda x: x["similarity"], reverse=True)
    return recommendations[:5]

# Добавление разнообразия
def add_diversity(recommendations: List[Dict], all_books: List[Dict], user_books: List[Dict]) -> List[Dict]:
    if len(recommendations) >= 5:
        return recommendations

    user_book_ids = {book["book"]["book_id"] for book in user_books}
    print(f"User book IDs in add_diversity: {user_book_ids}")

    unique_all_books = []
    seen_book_ids = set()
    for book in all_books:
        book_id = book["book"]["book_id"]
        if book_id not in seen_book_ids:
            seen_book_ids.add(book_id)
            unique_all_books.append(book)

    available_books = [book for book in unique_all_books if book["book"]["book_id"] not in user_book_ids and book["book"]["book_id"] not in [r["book_id"] for r in recommendations]]
    print(f"Available books after filtering: {[book['book']['book_id'] for book in available_books]}")
    if not available_books:
        return recommendations

    user_genres = set()
    for ub in user_books:
        user_genres.update(ub["book"].get("genres", ["Unknown"]))

    diverse_books = []
    for book in available_books:
        book_data = book["book"]
        book_genres = set(book_data.get("genres", ["Unknown"]))
        common_genres = book_genres & user_genres - {"Unknown"}
        if common_genres:
            diverse_books.append({
                **book_data,
                "reason": f"Because you like {', '.join(common_genres)}",
                "similarity": 0.0
            })

    if len(diverse_books) + len(recommendations) < 5:
        other_genres = set()
        for book in unique_all_books:
            book_genres = set(book["book"].get("genres", ["Unknown"]))
            other_genres.update(book_genres - user_genres - {"Unknown"})
        if other_genres:
            for book in available_books:
                if len(diverse_books) + len(recommendations) >= 5:
                    break
                book_data = book["book"]
                if book_data["book_id"] in [r["book_id"] for r in diverse_books + recommendations]:
                    continue
                book_genres = set(book_data.get("genres", ["Unknown"]))
                new_genres = book_genres & other_genres
                if new_genres:
                    diverse_books.append({
                        **book_data,
                        "reason": f"Explore a new genre: {', '.join(new_genres)}",
                        "similarity": 0.0
                    })

    if len(diverse_books) + len(recommendations) < 5:
        remaining_books = [book for book in available_books if book["book"]["book_id"] not in [r["book_id"] for r in diverse_books + recommendations]]
        prioritized_books = [book for book in remaining_books if "Unknown" not in book["book"].get("genres", ["Unknown"])]
        if prioritized_books:
            random.shuffle(prioritized_books)
            remaining_books = prioritized_books + [book for book in remaining_books if book not in prioritized_books]

        for book in remaining_books[:5 - len(recommendations) - len(diverse_books)]:
            book_data = book["book"]
            genres = book_data.get("genres", ["Unknown"])
            reason = "Explore a different book" if "Unknown" in genres else f"Explore a new genre: {', '.join(genres - {'Unknown'})}"
            diverse_books.append({
                **book_data,
                "reason": reason,
                "similarity": 0.0
            })

    all_recommendations = recommendations + diverse_books
    unique_recommendations = []
    seen_book_ids = set()
    for rec in all_recommendations:
        if rec["book_id"] not in seen_book_ids:
            seen_book_ids.add(rec["book_id"])
            unique_recommendations.append(rec)

    return unique_recommendations[:5]

# Фоновая задача для вычисления рекомендаций
@celery_app.task
def compute_recommendations_task(user_id: int, token: str):
    def sync_get_user_books(user_id: int, token: str) -> List[Dict]:
        cache_key = f"user_books:{user_id}"
        cached = redis_client_sync.get(cache_key)
        if cached:
            return json.loads(cached)

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_API_URL}/books/list/?user_id={user_id}", headers=headers)
        if response.status_code != 200:
            return []
        books = response.json().get("results", [])
        redis_client_sync.setex(cache_key, 3600, json.dumps(books))
        return books

    def sync_get_all_books(token: str, user_id: int) -> List[Dict]:
        cache_key = f"all_books:{user_id}"
        cached = redis_client_sync.get(cache_key)
        if cached:
            return json.loads(cached)

        headers = {"Authorization": f"Bearer {token}"}
        all_books = []
        next_url = f"{BASE_API_URL}/books/search/?status=available"

        while next_url:
            response = requests.get(next_url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch books")
            data = response.json()
            all_books.extend(data.get("results", []))
            next_url = data.get("next")

        redis_client_sync.setex(cache_key, 3600, json.dumps(all_books))
        return all_books

    user_books = sync_get_user_books(user_id, token)
    all_books = sync_get_all_books(token, user_id)

    if not user_books:
        if not all_books:
            return {"recommendations": []}
        random_books = random.sample(all_books, min(5, len(all_books)))
        return {
            "recommendations": [
                {**book["book"], "reason": "Popular book", "similarity": 0.0} for book in random_books
            ]
        }

    recommended_book_ids = collaborative_filtering(user_id, user_books)

    if not recommended_book_ids:
        user_book_ids = {book["book"]["book_id"] for book in user_books}
        recommended_book_ids = [book["book"]["book_id"] for book in all_books if book["book"]["book_id"] not in user_book_ids]

    recommendations = content_filtering(user_books, recommended_book_ids)

    user_genres = set()
    for ub in user_books:
        user_genres.update(ub["book"].get("genres", ["Unknown"]))

    for rec in recommendations:
        common_genres = user_genres.intersection(rec["genres"])
        rec["reason"] = f"Matches your interests (similarity: {rec['similarity']:.2f})"
        if common_genres:
            rec["reason"] += f" - Genres: {', '.join(common_genres)}"

    recommendations = add_diversity(recommendations, all_books, user_books)
    return {"recommendations": recommendations}

@app.get("/api/precompute/")
async def precompute(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header. Must start with 'Bearer '")
    token = authorization.split(" ")[1]

    async def get_all_user_books(token: str) -> List[Dict]:
        headers = {"Authorization": f"Bearer {token}"}
        all_books = []
        next_url = f"{BASE_API_URL}/books/all/"
        page = 1

        while next_url:
            response = requests.get(next_url, headers=headers)
            print(f"Page {page} - Response status: {response.status_code}, Text: {response.text}")
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to fetch all books: {response.text}")
            data = response.json()
            all_books.extend(data.get("results", []))
            next_url = data.get("next")
            page += 1
        print(f"Total books received: {len(all_books)}")
        print(f"All books received: {[book['book']['book_id'] for book in all_books]}")
        return all_books

    async def get_owners(books: List[Dict]) -> Dict[str, int]:
        user_book_ids = [str(book["user_book_id"]) for book in books]
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{BASE_API_URL}/books/owners/",
            json={"user_book_ids": user_book_ids},
            headers=headers
        )
        print(f"Owners response status: {response.status_code}, Text: {response.text}")
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch owners")
        owners = response.json()
        print(f"Owners: {owners}")
        return owners

    all_books = await get_all_user_books(token)
    if not all_books:
        raise HTTPException(status_code=404, detail="No books found")

    owners = await get_owners(all_books)

    user_ids = []
    book_ids = []
    for book in all_books:
        user_book_id = str(book["user_book_id"])
        owner_id = owners.get(user_book_id)
        if owner_id:
            user_ids.append(owner_id)
            book_ids.append(book["book"]["book_id"])

    user_ids = sorted(list(set(user_ids)))
    book_ids = sorted(list(set(book_ids)))
    print(f"User IDs: {user_ids}, Book IDs: {book_ids}")

    if not user_ids or not book_ids:
        raise HTTPException(status_code=404, detail="No valid users or books found")

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

    user_book_matrix = np.zeros((len(user_ids), len(book_ids)))
    for book in all_books:
        user_book_id = str(book["user_book_id"])
        owner_id = owners.get(user_book_id)
        if owner_id:
            user_idx = user_id_to_index[owner_id]
            book_idx = book_id_to_index[book["book"]["book_id"]]
            user_book_matrix[user_idx, book_idx] = 1

    print(f"User book matrix:\n{user_book_matrix}")

    books_data = []
    for book in all_books:
        books_data.append({
            "book_id": book["book"]["book_id"],
            "name": book["book"]["name"],
            "author": book["book"]["author"],
            "overview": book["book"]["overview"],
            "genres": book["book"]["genres"],
        })
    books_df = pd.DataFrame(books_data).drop_duplicates(subset=["book_id"])
    print(f"Books DataFrame: {books_df}")

    joblib.dump((user_ids, book_ids, user_book_matrix), "user_book_matrix.joblib")
    joblib.dump(books_df, "books_df.joblib")

    corpus = [
        " ".join(book["genres"]) + " " + book["overview"]
        for book in books_data
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    joblib.dump(tfidf_matrix, "tfidf_matrix.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    return {"message": "Precomputation completed"}

@app.get("/api/recommendations/")
async def get_recommendations(user_id: int, current_user_id: int = Depends(get_current_user_id),
                              token: str = Depends(get_token)):
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You can only access your own recommendations")

    cache_key = f"recommendations:{user_id}"
    cached = await redis_client_async.get(cache_key)
    print(f"Async Redis client type: {type(redis_client_async)}")
    print(f"Cached value type: {type(cached)}")
    print(f"Cached value: {cached}")
    if cached:
        return json.loads(cached)

    try:
        task = compute_recommendations_task.delay(user_id, token)
        result = task.get(timeout=10)
        print(f"Task result: {result}")
        await redis_client_async.setex(cache_key, 86400, json.dumps(result))
        return result
    except Exception as e:
        print(f"Task error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compute recommendations: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client_async.close()