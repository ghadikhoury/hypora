"""Tabular TEMPLATE. Learned preprocessing belongs inside fit(), never at module scope."""

import numpy as np
from runtime import Predictions, TrainData, TrialContext  # type: ignore[import-not-found]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


def _categorical_strings(X):
    """Normalize booleans/datetimes to strings while preserving missing entries."""
    # Format each scalar, so a timestamp does not change when other rows are added.
    # Column-wide datetime astype('string') chooses a format from the whole batch.
    return X.map(str).astype(object).where(X.notna(), np.nan)


def fit(train: TrainData, ctx: TrialContext):
    """Fit all learned transforms and the estimator using train.X/train.y only."""

    numeric = list(train.X.select_dtypes(include="number").columns)
    categorical = [name for name in train.X.columns if name not in numeric]
    # Imputation, scaling, and the category vocabulary are learned from training rows only.
    # Keeping these steps in the pipeline prevents fitting them on validation or test data.
    preparation = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        # Normalize pd.NA/NaT before the training-only imputer.
                        ("strings", FunctionTransformer(_categorical_strings)),
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )

    regression = ctx.task_type == "regression"
    estimator = (
        Ridge(alpha=1.0) if regression else LogisticRegression(max_iter=500, random_state=ctx.seed)
    )
    model = Pipeline([("preparation", preparation), ("model", estimator)])
    model.fit(train.X, train.y)
    return model


def predict(model, X) -> Predictions:
    """Reuse the fitted transforms; never refit preprocessing during prediction."""
    values = model.predict(X)
    estimator = model[-1]
    if hasattr(estimator, "classes_"):
        return Predictions(np.asarray(values), model.predict_proba(X), tuple(estimator.classes_))
    return Predictions(np.asarray(values))
