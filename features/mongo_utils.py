"""
Centralized MongoDB Atlas connection and resilience utilities.
Optimized for Streamlit: low-latency fail-fast and simplified retry.
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

def mongo_retry(max_retries=2, delay=0.5):
    """
    Simplified decorator to retry MongoDB operations on network failure.
    Optimized for UI responsiveness: fewer retries and shorter delays.
    """
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
                        time.sleep(delay)
            # Log error but let the caller handle the exception or fallback
            logger.error(f"MongoDB operation exhausted {max_retries} attempts.")
            raise last_exception
        return wrapper
    return decorator

def get_mongo_client() -> MongoClient:
    """Standardized MongoClient with aggressive fail-fast timeouts for UI."""
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        raise EnvironmentError("MONGO_URI is required in .env for database access.")
    
    ca = certifi.where()
    # Aggressive timeouts to prevent Streamlit UI freezes
    client = MongoClient(
        uri,
        retryWrites=True,
        retryReads=True,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=15000,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True 
    )
    return client

def get_database(db_name="aqi_predictor"):
    """Get a database instance with instant connectivity check."""
    client = get_mongo_client()
    db = client[db_name]
    # Quick ping to verify connectivity before proceeding
    client.admin.command('ping')
    return db
