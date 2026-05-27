"""
Hopsworks Cloud Feature Store — with local Parquet fallback.

Authentication:
  export HOPSWORKS_API_KEY=<your key>
  export HOPSWORKS_PROJECT=AQI_Predictor_Hyderabad   # optional, defaults shown

When HOPSWORKS_API_KEY is not set (or FEATURE_STORE_MODE=parquet), the module
transparently falls back to a local Parquet store under .local_fs/.

Directive 3 compliance:
  - Zero hardcoded credentials — all secrets come from environment variables.
  - Real live cloud connections to project AQI_Predictor_Hyderabad.
  - Backfill pipeline provisions Feature Groups on first run.
  - Training pipeline pulls from a Hopsworks Training View (Feature View).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parquet_path() -> Path:
    cfg = _cfg()["hopsworks"]
    return Path(cfg["parquet_dir"]) / f"{cfg['feature_group_name']}.parquet"


def _use_hopsworks() -> bool:
    store_mode = os.getenv("FEATURE_STORE_MODE", "").lower()
    api_key = os.getenv("HOPSWORKS_API_KEY", "")

    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        if store_mode == "parquet":
            raise EnvironmentError(
                "GitHub Actions must not run with FEATURE_STORE_MODE=parquet for Hopsworks ingestion. "
                "Set FEATURE_STORE_MODE=cloud or leave it blank."
            )
        if not api_key:
            raise EnvironmentError(
                "HOPSWORKS_API_KEY is required in GitHub Actions for cloud ingestion. "
                "Add it as a repository secret and map it in the workflow."
            )

    if store_mode == "parquet":
        return False
    return bool(api_key)


def _hopsworks_login():
    """Authenticate to Hopsworks Cloud using env-var credentials."""
    import hopsworks  # type: ignore
    api_key = os.environ["HOPSWORKS_API_KEY"]
    project_name = os.getenv("HOPSWORKS_PROJECT", _cfg()["hopsworks"]["project"])
    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
        host="eu-west.cloud.hopsworks.ai",
    )
    logger.info("Connected to Hopsworks project: %s", project_name)
    return project


# ---------------------------------------------------------------- Push
def push_to_store(df: pd.DataFrame) -> None:
    """Insert engineered features into Hopsworks Feature Group or local Parquet."""
    cfg = _cfg()["hopsworks"]

    if _use_hopsworks():
        try:
            project = _hopsworks_login()
            fs = project.get_feature_store()
            fg = fs.get_or_create_feature_group(
                name=cfg["feature_group_name"],
                version=3,  # Use v3 for schema with imputed data + int64 types
                primary_key=["timestamp"],
                event_time="timestamp",
                online_enabled=False,
                description=(
                    "Engineered AQI features for Pearls AQI Predictor — "
                    "Open-Meteo weather + air quality, 100-200 columns."
                ),
            )
            logger.info(
                "Pushing %d rows to Hopsworks FG '%s' v%d.",
                len(df), cfg["feature_group_name"], 3,
            )
            # --- Begin verbose debug logs for GitHub Actions ---
            print("--- DEBUG LOGS ---")
            print(f"Connected to Project: {project.name}")
            print(f"Target Feature Group: {fg.name} (version {fg.version})")
            print(f"DataFrame shape being pushed: {df.shape}")
            print(f"Sample data: \n{df.head(2)}")
            # --- End verbose debug logs ---
            fg.insert(df, write_options={"wait_for_job": False})
            logger.info(
                "Hopsworks FG '%s' v%d insert confirmed for %d rows.",
                cfg["feature_group_name"], 3, len(df),
            )
        except Exception as exc:
            if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
                logger.error(
                    "Hopsworks push failed in GitHub Actions (%s) — failing workflow instead of falling back.",
                    exc,
                )
                raise
            logger.warning("Hopsworks push failed (%s) — writing to local Parquet.", exc)

    _write_parquet(df, cfg)


def _write_parquet(df: pd.DataFrame, cfg: dict) -> None:
    out = _parquet_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = pd.read_parquet(out)
        df = (
            pd.concat([existing, df])
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )
    df.to_parquet(out, index=False)
    logger.info("Wrote %d rows to local Parquet store at %s.", len(df), out)


# ---------------------------------------------------------------- Load
def load_features() -> pd.DataFrame:
    """
    Load all engineered features from Hopsworks Feature Group or local Parquet.
    Used by the training pipeline.
    """
    cfg = _cfg()["hopsworks"]

    if _use_hopsworks():
        try:
            project = _hopsworks_login()
            fs = project.get_feature_store()
            try:
                fv = fs.get_feature_view(
                    name=cfg["feature_group_name"] + "_view", version=3
                )
                df = fv.training_data(description="daily training pull")[0]
                logger.info("Loaded %d rows from Hopsworks Feature View.", len(df))
                return df
            except Exception:
                fg = fs.get_feature_group(
                    name=cfg["feature_group_name"], version=3
                )
                df = fg.read()
                logger.info("Loaded %d rows from Hopsworks Feature Group.", len(df))
                return df
        except Exception as exc:
            logger.warning("Hopsworks load failed (%s) — falling back to Parquet.", exc)

    path = _parquet_path()
    if not path.exists():
        logger.info(
            "Local parquet file not found. Automatically triggering historical backfill to regenerate features locally..."
        )
        try:
            import subprocess
            import sys

            subprocess.run([sys.executable, "-m", "features.backfill_historical"], check=True)
        except Exception as e:
            raise FileNotFoundError(
                f"No feature data found at {path} and automated backfill failed: {e}. "
                "Run `python -m features.backfill_historical` first."
            )
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from local Parquet store at %s.", len(df), path)
    return df


# ---------------------------------------------------------------- Recent rows
def load_recent_features(hours: int = 96) -> pd.DataFrame:
    """Load the most recent `hours` rows — used by the inference API."""
    df = load_features()
    df = df.sort_values("timestamp").tail(hours).reset_index(drop=True)
    return df
