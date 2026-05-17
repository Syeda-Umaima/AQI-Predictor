"""Local Hopsworks Feature Store with Parquet fallback (decoupled local architecture)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parquet_path() -> Path:
    cfg = _cfg()["hopsworks"]
    return Path(cfg["parquet_dir"]) / f"{cfg['feature_group_name']}.parquet"


def get_feature_store():
    """Return (feature_store, mode) where mode is 'hopsworks' or 'parquet'."""
    cfg = _cfg()["hopsworks"]
    if os.getenv("FEATURE_STORE_MODE", "").lower() == "parquet":
        Path(cfg["parquet_dir"]).mkdir(parents=True, exist_ok=True)
        return cfg["parquet_dir"], "parquet"
    try:
        import hopsworks  # type: ignore
        project = hopsworks.login(
            host=cfg["host"],
            project=cfg["project"],
            api_key_value=cfg["api_key_value"],
        )
        return project.get_feature_store(), "hopsworks"
    except Exception as e:
        if not cfg["fallback_to_parquet"]:
            raise
        logger.warning("Hopsworks unavailable (%s) — falling back to Parquet store.", e)
        Path(cfg["parquet_dir"]).mkdir(parents=True, exist_ok=True)
        return cfg["parquet_dir"], "parquet"


def push_to_store(df: pd.DataFrame) -> None:
    cfg = _cfg()["hopsworks"]
    fs, mode = get_feature_store()
    if mode == "hopsworks":
        fg = fs.get_or_create_feature_group(
            name=cfg["feature_group_name"],
            version=cfg["feature_group_version"],
            primary_key=["timestamp"],
            event_time="timestamp",
            description="Engineered AQI features for Pearls AQI Predictor.",
        )
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info("Inserted %d rows into Hopsworks FG '%s'.", len(df), cfg["feature_group_name"])
    else:
        out = Path(fs) / f"{cfg['feature_group_name']}.parquet"
        if out.exists():
            existing = pd.read_parquet(out)
            df = pd.concat([existing, df]).drop_duplicates("timestamp").sort_values("timestamp")
        df.to_parquet(out, index=False)
        logger.info("Wrote %d rows to local Parquet store at %s.", len(df), out)


def load_features() -> pd.DataFrame:
    """Load engineered features from Hopsworks or the local Parquet feature store."""
    cfg = _cfg()["hopsworks"]
    if os.getenv("FEATURE_STORE_MODE", "").lower() == "parquet":
        path = _parquet_path()
        if not path.exists():
            raise FileNotFoundError(
                "No Parquet feature store found. Run `python -m features.backfill_historical` first."
            )
        return pd.read_parquet(path)

    try:
        import hopsworks  # type: ignore
        project = hopsworks.login(
            host=cfg["host"],
            project=cfg["project"],
            api_key_value=cfg["api_key_value"],
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(cfg["feature_group_name"], version=cfg["feature_group_version"])
        return fg.read()
    except Exception as e:
        if not cfg["fallback_to_parquet"]:
            raise
        path = _parquet_path()
        if not path.exists():
            raise FileNotFoundError(
                "No feature data found. Run `python -m features.backfill_historical` first."
            ) from e
        logger.warning("Hopsworks unavailable (%s) — reading Parquet fallback.", e)
        return pd.read_parquet(path)
