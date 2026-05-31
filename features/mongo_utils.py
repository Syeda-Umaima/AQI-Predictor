"""
Centralized MongoDB Atlas connection and resilience utilities.
Refactored for dual-mode execution: Headless (CLI/Actions) and Streamlit.
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

def is_streamlit_running() -> bool:
    """Check if the code is running inside a Streamlit environment."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

# Global client for non-Streamlit (headless) environments
_GLOBAL_CLIENT = None

def create_mongo_client() -> MongoClient:
    """Create a new MongoClient with optimized production parameters."""
    # 1. Resolve URI
    uri = os.getenv("MONGO_URI", "").strip()
    if is_streamlit_running():
        import streamlit as st
        try:
            if "MONGO_URI" in st.secrets:
                uri = st.secrets["MONGO_URI"].strip()
        except Exception:
            pass
    
    if not uri:
        raise EnvironmentError("MONGO_URI is missing in Secrets or .env")
    
    ca = certifi.where()
    
    # 2. Configure Client
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=300000, 
        connectTimeoutMS=300000,
        socketTimeoutMS=300000,
        retryWrites=True,
        retryReads=True,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True,
        maxPoolSize=50,    # Increased for concurrent GitHub Actions + Streamlit
        minPoolSize=1,
        appName="AQI_Predictor_Pipeline",
        waitQueueTimeoutMS=120000, # Wait longer for a connection from the pool
        heartbeatFrequencyMS=10000, # More frequent heartbeats to keep connection alive
        compressors="snappy,zlib,zstd", # Added zstd as well
        zlibCompressionLevel=3
    )

def get_mongo_client() -> MongoClient:
    """Entry point for getting a cached MongoClient based on environment."""
    if is_streamlit_running():
        import streamlit as st
        # Lazy import to avoid streamlit dependency in CLI
        @st.cache_resource
        def _get_streamlit_client():
            return create_mongo_client()
        return _get_streamlit_client()
    
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = create_mongo_client()
    return _GLOBAL_CLIENT

def get_database(db_name="aqi_predictor"):
    """Get the database instance."""
    return get_mongo_client()[db_name]
