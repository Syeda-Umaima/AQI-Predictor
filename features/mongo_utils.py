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
                        time.sleep(delay)
            logger.error(f"MongoDB operation exhausted {max_retries} attempts.")
            raise last_exception
        return wrapper
    return decorator

# Global client cache for non-Streamlit environments
_CLIENT_CACHE = None

def is_streamlit_running() -> bool:
    """Check if the code is running inside a Streamlit environment."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

def _get_uri() -> str:
    """Internal helper to resolve MONGO_URI from Secrets or Env."""
    uri = None
    try:
        if "MONGO_URI" in st.secrets:
            uri = st.secrets["MONGO_URI"].strip()
    except Exception:
        pass
    
    if not uri:
        uri = os.getenv("MONGO_URI", "").strip()

    if not uri:
        raise EnvironmentError("MONGO_URI is missing in Streamlit Secrets or .env")
    return uri

def create_mongo_client(uri: str) -> MongoClient:
    """Create a new MongoClient with production resilience parameters."""
    ca = certifi.where()
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=30000, 
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        retryWrites=True,
        retryReads=True,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True,
        maxPoolSize=10,
        minPoolSize=1
    )

def get_mongo_client() -> MongoClient:
    """
    Standardized MongoClient cached as a global resource.
    Environment-aware: uses st.cache_resource if in Streamlit, else a global var.
    """
    if is_streamlit_running():
        return _get_cached_client_streamlit()
    
    global _CLIENT_CACHE
    if _CLIENT_CACHE is None:
        uri = _get_uri()
        _CLIENT_CACHE = create_mongo_client(uri)
    return _CLIENT_CACHE

@st.cache_resource
def _get_cached_client_streamlit() -> MongoClient:
    """Streamlit-specific cached resource for MongoClient."""
    uri = _get_uri()
    return create_mongo_client(uri)

def get_database(db_name="aqi_predictor"):
    """Get the cached database instance instantly."""
    client = get_mongo_client()
    return client[db_name]
