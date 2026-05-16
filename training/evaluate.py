"""
Champion evaluation + SHAP and LIME explainability.

Generates:
  * artifacts/shap_summary.png, artifacts/shap_bar.png
  * artifacts/lime_explanation.png, artifacts/lime_weights.json
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

from training.train import load_training_frame, time_ordered_split

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def load_champion():
    meta = json.loads((ARTIFACTS / "leaderboard.json").read_text())
    name = meta["champion"]
    if name == "lstm":
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(MODELS_DIR / "champion_lstm.keras")
        scaler = joblib.load(MODELS_DIR / "champion_scaler.joblib")
        return name, model, scaler, meta
    return name, joblib.load(MODELS_DIR / "champion.joblib"), None, meta


def _tabular_predict_fn(name: str, model, scaler, cols: list[str]):
    if name == "lstm":
        def predict_fn(arr: np.ndarray) -> np.ndarray:
            arr_s = scaler.transform(arr)
            seq = arr_s.reshape(arr_s.shape[0], 1, arr_s.shape[1]).repeat(
                model.input_shape[1], axis=1
            )
            return model.predict(seq, verbose=0).ravel()
        return predict_fn

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        return model.predict(pd.DataFrame(arr, columns=cols))
    return predict_fn


def run_shap(name: str, model, scaler, meta: dict, X_test: pd.DataFrame) -> None:
    if name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif name == "ridge":
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
    else:
        predict_fn = _tabular_predict_fn(name, model, scaler, meta["feature_columns"])
        background = shap.sample(X_test, min(50, len(X_test)))
        explainer = shap.KernelExplainer(predict_fn, background)
        X_shap = X_test.iloc[: min(100, len(X_test))]
        shap_values = explainer.shap_values(X_shap, nsamples=100)
        X_test = X_shap

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()
    logger.info("Wrote SHAP plots to %s/", ARTIFACTS)


def run_lime(
    name: str, model, scaler, meta: dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
) -> None:
    cols = meta["feature_columns"]
    background = X_train.sample(min(2000, len(X_train)), random_state=42).values
    instance = X_test.iloc[0].values
    predict_fn = _tabular_predict_fn(name, model, scaler, cols)

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=cols,
        mode="regression",
        random_state=42,
    )
    explanation = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=min(15, len(cols)),
    )

    fig = explanation.as_pyplot_figure()
    plt.title(f"LIME — local explanation ({meta['champion']})")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "lime_explanation.png", dpi=140, bbox_inches="tight")
    plt.close()

    weights = {feat: float(weight) for feat, weight in explanation.as_list()}
    (ARTIFACTS / "lime_weights.json").write_text(json.dumps(weights, indent=2))
    logger.info("Wrote LIME artefacts to %s/", ARTIFACTS)


def run_explainability() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name, model, scaler, meta = load_champion()
    df = load_training_frame()
    train_df, test_df = time_ordered_split(df, 0.2)
    cols = meta["feature_columns"]
    X_train = train_df[cols]
    X_test = test_df[cols].sample(min(500, len(test_df)), random_state=42)

    run_shap(name, model, scaler, meta, X_test)
    run_lime(name, model, scaler, meta, X_train, X_test)

    print("Leaderboard:")
    print(pd.DataFrame(meta["leaderboard"]).T.round(4))


if __name__ == "__main__":
    run_explainability()
