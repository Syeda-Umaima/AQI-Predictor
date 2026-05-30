"""
Centralized MongoDB Atlas connection and resilience utilities.
"""
import os
import time
import logging
import certifi
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from functools import wraps

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env", override=False)

def mongo_retry(max_retries=3, delay=1):
    """Decorator to retry MongoDB operations on network failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"MongoDB operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            logger.error(f"MongoDB operation failed after {max_retries} attempts.")
            raise last_exception
        return wrapper
    return decorator

def get_mongo_client() -> MongoClient:
    """Standardized MongoClient with production resilience parameters."""
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        raise EnvironmentError("MONGO_URI is required in .env for database access.")
    
    ca = certifi.where()
    client = MongoClient(
        uri,
        retryWrites=True,
        retryReads=True,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=30000,
        socketTimeoutMS=45000,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True # Handling potential CA issues in restricted environments
    )
    return client

def get_database(db_name="aqi_predictor"):
    """Get a database instance with connectivity check."""
    client = get_mongo_client()
    db = client[db_name]
    # Ping to verify connectivity
    client.admin.command('ping')
    return db
