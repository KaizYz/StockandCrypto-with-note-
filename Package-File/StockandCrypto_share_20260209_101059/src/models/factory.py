from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _try_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        return LGBMClassifier, LGBMRegressor
    except Exception:
        return None, None


def _try_xgboost():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception:
        return None


def build_logistic_binary(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=seed,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )


def build_logistic_multiclass(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=seed,
                    class_weight="balanced",
                    max_iter=3000,
                    multi_class="auto",
                ),
            ),
        ]
    )


def build_mvp_classifier(
    cfg: Dict[str, Any],
    *,
    n_classes: int,
    seed: int,
) -> Tuple[Any, str]:
    backend = cfg.get("models", {}).get("mvp_backend", "lightgbm").lower()
    model_cfg = cfg.get("models", {}).get("classifier", {})

    LGBMClassifier, _ = _try_lightgbm()
    if backend == "lightgbm" and LGBMClassifier is not None:
        params = {
            "n_estimators": model_cfg.get("n_estimators", 500),
            "learning_rate": model_cfg.get("learning_rate", 0.03),
            "num_leaves": model_cfg.get("num_leaves", 63),
            "max_depth": model_cfg.get("max_depth", -1),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "reg_alpha": model_cfg.get("reg_alpha", 0.0),
            "reg_lambda": model_cfg.get("reg_lambda", 0.0),
            "random_state": seed,
            "verbosity": -1,
        }
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
            params["class_weight"] = "balanced"
        else:
            params["objective"] = "binary"
            params["class_weight"] = "balanced"
        return LGBMClassifier(**params), "lightgbm"

    XGBClassifier = _try_xgboost()
    if backend == "xgboost" and XGBClassifier is not None:
        obj = "multi:softprob" if n_classes > 2 else "binary:logistic"
        xgb = XGBClassifier(
            objective=obj,
            n_estimators=model_cfg.get("n_estimators", 500),
            learning_rate=model_cfg.get("learning_rate", 0.03),
            max_depth=model_cfg.get("max_depth", 6),
            subsample=model_cfg.get("subsample", 0.8),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
            reg_alpha=model_cfg.get("reg_alpha", 0.0),
            reg_lambda=model_cfg.get("reg_lambda", 1.0),
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=seed,
        )
        return xgb, "xgboost"

    # Pure sklearn fallback
    fallback = HistGradientBoostingClassifier(
        max_depth=model_cfg.get("max_depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        max_iter=model_cfg.get("n_estimators", 300),
        random_state=seed,
    )
    return fallback, "sklearn_hgb"


def build_quantile_regressor(
    cfg: Dict[str, Any], *, quantile: float, seed: int
) -> Tuple[Any, str]:
    backend = cfg.get("models", {}).get("mvp_backend", "lightgbm").lower()
    model_cfg = cfg.get("models", {}).get("regressor", {})

    _, LGBMRegressor = _try_lightgbm()
    if backend == "lightgbm" and LGBMRegressor is not None:
        reg = LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=model_cfg.get("n_estimators", 500),
            learning_rate=model_cfg.get("learning_rate", 0.03),
            num_leaves=model_cfg.get("num_leaves", 63),
            max_depth=model_cfg.get("max_depth", -1),
            subsample=model_cfg.get("subsample", 0.8),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
            reg_alpha=model_cfg.get("reg_alpha", 0.0),
            reg_lambda=model_cfg.get("reg_lambda", 0.0),
            random_state=seed,
            verbosity=-1,
        )
        return reg, "lightgbm_quantile"

    # sklearn fallback supports quantile loss directly.
    reg = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=model_cfg.get("n_estimators", 300),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        max_depth=model_cfg.get("max_depth", 3),
        random_state=seed,
    )
    return reg, "sklearn_gbr_quantile"
