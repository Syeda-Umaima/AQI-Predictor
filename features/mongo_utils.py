"""
Centralized MongoDB Atlas connection and resilience utilities.
Optimized for Streamlit: low-latency fail-fast and simplified retry.
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

def mongo_retry(max_retries=2, delay=1.0):
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
    # 1. Look for Streamlit Cloud Secret first; fall back to local .env if running locally
    if "MONGO_URI" in st.secrets:
        uri = st.secrets["MONGO_URI"].strip()
    else:
        uri = os.getenv("MONGO_URI", "").strip()

    if not uri:
        raise EnvironmentError("MONGO_URI is required in Streamlit Secrets or .env file.")
    
    ca = certifi.where()
    
    # 2. Reusable client configuration with robust safety timeouts
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=30000,  # Allow time for cloud environments to handshake
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        tlsCAFile=ca,                    # Uses secure certifi certificate chain
    )
    return client

def get_database(db_name="aqi_predictor"):
    """Get the cached database instance instantly without duplicate overhead."""
    client = get_mongo_client()
    return client[db_name]