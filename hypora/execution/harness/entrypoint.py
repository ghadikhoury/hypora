"""Read-only sandbox entrypoint. Joblib is loaded ONLY here, never in the worker/control plane."""

import importlib.metadata
import importlib.util
import json
import platform
import random
import resource
import sys
from pathlib import Path

# python -I excludes /trial and user-controlled PYTHONPATH. Only this read-only path is added.
sys.path.insert(0, "/harness")

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from runtime import Predictions, TrainData, TrialContext  # type: ignore[import-not-found]


def main() -> None:
    """Fit or predict inside the sandbox and write data-only outputs for the worker."""
    action = sys.argv[1]
    context = TrialContext(**json.loads(Path("/data/context.json").read_text()))
    random.seed(context.seed)
    np.random.seed(context.seed)
    spec = importlib.util.spec_from_file_location("trial_pipeline", "/trial/pipeline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if action == "fit":
        X = pd.read_parquet("/data/train_features.parquet", engine="pyarrow")
        y = pd.read_parquet("/data/train_labels.parquet", engine="pyarrow")["target"]
        valid = Path("/data/validation_features.parquet")
        train = TrainData(
            X,
            y,
            pd.read_parquet(valid, engine="pyarrow") if valid.exists() else None,
            pd.read_parquet("/data/validation_labels.parquet", engine="pyarrow")["target"]
            if valid.exists()
            else None,
        )
        model = module.fit(train, context)
        joblib.dump(model, "/out/model.joblib")
    elif action == "predict":
        # Deserialization is untrusted execution and stays behind the same sandbox boundary.
        model = joblib.load("/data/model.joblib")
        X = pd.read_parquet("/data/features.parquet", engine="pyarrow")
        result = module.predict(model, X)
        if not isinstance(result, Predictions):
            raise TypeError("predict must return runtime.Predictions")
        frame = pd.DataFrame({"prediction": result.values})
        if result.probabilities is not None:
            for i in range(len(result.classes)):
                frame[f"p_{i}"] = result.probabilities[:, i]
        frame.to_parquet("/out/predictions.parquet", engine="pyarrow", index=False)
        Path("/out/classes.json").write_text(json.dumps(list(result.classes), default=_scalar))
    else:
        raise ValueError(f"unknown harness action: {action}")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    Path("/out/resources.json").write_text(
        json.dumps(
            {
                "python_version": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow")
                },
                "peak_rss_bytes": usage.ru_maxrss * 1024,
                "user_cpu_seconds": usage.ru_utime,
                "system_cpu_seconds": usage.ru_stime,
            }
        )
    )


def _scalar(value):
    return value.item()


if __name__ == "__main__":
    main()
