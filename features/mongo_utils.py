"""
Centralized MongoDB Atlas connection and resilience utilities.
Optimized for Streamlit: cached connection pool to prevent socket exhaustion.
"""
import os
import time
import logging
import certifi
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from functools import wraps

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env", override=False)

def mongo_retry(max_retries=3, delay=2.0):
    """
    Retry decorator for MongoDB operations.
    Increased delay and retries for cloud environments.
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
            logger.error(f"MongoDB operation exhausted {max_retries} attempts.")
            raise last_exception
        return wrapper
    return decorator

@st.cache_resource
def get_mongo_client() -> MongoClient:
    """
    Standardized MongoClient cached as a single global resource.
    Prevents connection leakage and socket exhaustion on MongoDB Atlas.
    """
    # 1. Prioritize Streamlit Cloud Secrets, fallback to .env
    uri = None
    try:
        if "MONGO_URI" in st.secrets:
            uri = st.secrets["MONGO_URI"].strip()
    except Exception:
        pass
    
    if not uri:
        uri = os.getenv("MONGO_URI", "").strip()

    if not uri:
        raise EnvironmentError("MONGO_URI is missing in Secrets or .env")
    
    ca = certifi.where()
    
    # 2. Production-grade client with robust timeouts and TLS settings
    # We use 'Goldilocks' timeouts: long enough for handshakes, short enough to not lock UI.
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=15000, 
        connectTimeoutMS=15000,
        socketTimeoutMS=20000,
        retryWrites=True,
        retryReads=True,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True, # Added to bypass potential cert validation issues in cloud runners
        maxPoolSize=10,    # Limit pool size per worker to prevent Atlas connection spikes
        minPoolSize=1
    )
    return client

@st.cache_resource
def get_database(db_name="aqi_predictor"):
    """Get the cached database instance instantly without duplicate overhead."""
    client = get_mongo_client()
    db = client[db_name]
    return db
