from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import math, random
import copy
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ⬇⬇⬇  just add the optional argument and pass it to .fit(...)
def train_model(X_train, y_train, X_test, y_test,
                fold=None, sample_weight=None,n_jobs=-1, et_params=None):
    if fold is None:
        fold = random.randint(0, 10000)
    if et_params is not None:
        model = ExtraTreesRegressor(
            n_jobs=n_jobs,
            random_state=fold,
            **et_params,
        )
    else:
        model = ExtraTreesRegressor(
            n_jobs=n_jobs,
            max_features=1.0,
            random_state=fold,
        )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, y_pred), "rmse": rmse(y_test, y_pred)}
    return model, y_pred, metrics

def train_multiview_with_stacking(
    *,
    X_base,                # original training design (outer fold)
    y_base,
    X_test, y_test,        # outer test fold
    view_builders: List[Dict[str, Callable]],  # each: {"name": str, "build_aug": callable(indices, seed)->(X_aug,y_aug)}
    et_params=None,
    n_jobs: int = -1,
    random_state: int = 42,
    inner_splits: int = 5,
) -> Tuple[List[Dict], object, np.ndarray, Dict[str, float]]:
    """
    Train one ExtraTrees per view (perspective) and a RidgeCV combiner trained on
    out-of-fold (OOF) predictions from the outer training set.
    Returns: (per_view_records, combiner, y_pred_ensemble_on_test, ensemble_metrics).
    """
    rng = np.random.default_rng(random_state)
    n_train = X_base.shape[0]
    n_views = len(view_builders)

    # 1) Build OOF predictions matrix on the outer training fold
    P_oof = np.zeros((n_train, n_views), dtype=float)
    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_train)), 1):
        # for each view, build augmented train set for this inner fold and train a model
        for v, vb in enumerate(view_builders):
            # rebuild augmentation from *only* tr_idx
            X_aug, y_aug = vb["build_aug"](tr_idx, int(random_state + 1000*fold_id + v))
            # fit ET on augmented
            model, _, _ = train_model(
                X_aug, y_aug, X_base[va_idx], y_base[va_idx],
                fold=random_state + 10*fold_id + v, n_jobs=n_jobs, et_params=et_params
            )
            # predict OOF on va_idx
            P_oof[va_idx, v] = model.predict(X_base[va_idx])

    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    combiner = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=alphas, store_cv_values=False)
    )
    combiner.fit(P_oof, y_base)

    per_view = []
    P_test = np.zeros((X_test.shape[0], n_views), dtype=float)
    for v, vb in enumerate(view_builders):
        X_aug_full, y_aug_full = vb["build_aug"](np.arange(n_train), int(random_state + 9999 + v))
        model, y_pred_test, metrics = train_model(
            X_aug_full, y_aug_full, X_test, y_test,
            fold=random_state + 999 + v, n_jobs=n_jobs, et_params=et_params
        )
        per_view.append({
            "name": vb["name"],
            "model": model,
            "y_pred_test": y_pred_test,
            "metrics": metrics
        })
        P_test[:, v] = y_pred_test

    # 4) Ensemble prediction on test via the learned combiner
    y_pred_ens = combiner.predict(P_test)
    ens_metrics = {"r2": r2_score(y_test, y_pred_ens), "rmse": rmse(y_test, y_pred_ens)}
    return per_view, combiner, y_pred_ens, ens_metrics