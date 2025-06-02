from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math, random

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ⬇⬇⬇  just add the optional argument and pass it to .fit(...)
def train_model(X_train, y_train, X_test, y_test,
                fold=None, hyperparams=None, sample_weight=None):
    if fold is None:
        fold = random.randint(0, 10000)
    model = ExtraTreesRegressor(
        n_jobs=-1,
        max_features=1.0,
        random_state=fold,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, y_pred), "rmse": rmse(y_test, y_pred)}
    return model, y_pred, metrics
