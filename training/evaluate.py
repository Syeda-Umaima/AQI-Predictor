"""
Champion model evaluation — SHAP and LIME explainability.

Generates:
  artifacts/shap_summary.png   — beeswarm plot of SHAP values
  artifacts/shap_bar.png       — mean |SHAP| bar chart
  artifacts/lime_explanation.png — LIME local instance explanation
  artifacts/lime_weights.json    — top feature weights from LIME

Handles Ridge (LinearExplainer), tree models (TreeExplainer),
and LSTM (KernelExplainer with background sampling).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from training.train import load_training_frame, time_ordered_split, feature_columns

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TARGET = "target_aqi_next_1h"


# ---------------------------------------------------------------- Load model
def load_champion():
    meta_path = MODELS_DIR / "leaderboard.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"leaderboard.json not found at {meta_path}. Train models first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    name = meta["champion"]

    if name == "lstm":
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(MODELS_DIR / "champion_lstm.keras")
        scaler = joblib.load(MODELS_DIR / "champion_scaler.joblib")
        return name, model, scaler, meta

    artifact = joblib.load(MODELS_DIR / "champion.joblib")
    return name, artifact["model"], artifact.get("scaler"), meta


# ---------------------------------------------------------------- Predict fn
def _build_predict_fn(name: str, model, scaler, cols: list[str]):
    if name == "lstm":
        seq_len = model.input_shape[1]

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            arr_s = scaler.transform(arr)
            arr_seq = arr_s.reshape(arr_s.shape[0], 1, arr_s.shape[1]).repeat(seq_len, axis=1)
            return model.predict(arr_seq, verbose=0).ravel()

        return predict_fn

    if name == "ridge" and scaler is not None:
        def predict_fn(arr: np.ndarray) -> np.ndarray:
            return model.predict(scaler.transform(arr))
        return predict_fn

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        return model.predict(pd.DataFrame(arr, columns=cols))

    return predict_fn


# ------------------------------------------------------------------ SHAP
def run_shap(name: str, model, scaler, meta: dict, X_test: pd.DataFrame) -> None:
    cols = meta["feature_columns"]
    X_test = X_test[cols].copy()

    try:
        if name in ("random_forest", "xgboost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        elif name == "ridge":
            X_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols) if scaler else X_test
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            X_test = X_scaled
        else:
            predict_fn = _build_predict_fn(name, model, scaler, cols)
            n_bg = max(len(cols) + 10, min(200, len(X_test)))
            background = shap.sample(X_test, min(n_bg, len(X_test)))
            explainer = shap.KernelExplainer(predict_fn, background)
            X_shap = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_shap, nsamples=200)
            X_test = X_shap

        shap.summary_plot(shap_values, X_test, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "shap_summary.png", dpi=140, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "shap_bar.png", dpi=140, bbox_inches="tight")
        plt.close()

        logger.info("SHAP plots saved to %s/", ARTIFACTS)

    except Exception as exc:
        logger.warning("SHAP failed (%s) — skipping SHAP plots.", exc)


# ------------------------------------------------------------------ LIME
def run_lime(
    name: str, model, scaler, meta: dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
) -> None:
    cols = meta["feature_columns"]
    X_train_sub = X_train[cols]
    X_test_sub = X_test[cols]

    background = X_train_sub.sample(min(2000, len(X_train_sub)), random_state=42).values
    instance = X_test_sub.iloc[0].values
    predict_fn = _build_predict_fn(name, model, scaler, cols)

    try:
        explainer = LimeTabularExplainer(
            training_data=background,
            feature_names=cols,
            mode="regression",
            random_state=42,
        )
        explanation = explainer.explain_instance(
            instance, predict_fn, num_features=min(20, len(cols))
        )

        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME local explanation — {meta['champion']}")
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "lime_explanation.png", dpi=140, bbox_inches="tight")
        plt.close()

        weights = {feat: float(w) for feat, w in explanation.as_list()}
        (ARTIFACTS / "lime_weights.json").write_text(
            json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("LIME artefacts saved to %s/", ARTIFACTS)

    except Exception as exc:
        logger.warning("LIME failed (%s) — skipping LIME artefacts.", exc)


# -------------------------------------------------------------------- Driver
def run_explainability() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name, model, scaler, meta = load_champion()
    df = load_training_frame()
    cols = [c for c in meta["feature_columns"] if c in df.columns]
    meta["feature_columns"] = cols

    train_df, test_df = time_ordered_split(df, 0.2)
    X_train = train_df[cols]
    X_test = test_df[cols].sample(min(500, len(test_df)), random_state=42)

    run_shap(name, model, scaler, meta, X_test)
    run_lime(name, model, scaler, meta, X_train, X_test)

    print("\nModel Leaderboard:")
    print(pd.DataFrame(meta["leaderboard"]).T.round(4).sort_values("rmse"))
    print(f"\nChampion: {meta['champion']}")


if __name__ == "__main__":
    run_explainability()
