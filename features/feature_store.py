"""Local Hopsworks Feature Store with Parquet fallback."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_store():
    """Return (feature_store, mode) where mode is 'hopsworks' or 'parquet'."""
    cfg = _cfg()["hopsworks"]
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
