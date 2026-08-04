from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore
from mlfit.strategies import register_strategy


@register_strategy("sklearn")
class SklearnStrategy(BaseStrategy):
    """
    scikit-learn: CPU-only inference for classic machine learning models.

    Best for: RandomForest, SVM, LogisticRegression, XGBoost, LightGBM,
              and any sklearn-compatible estimator. Models are saved with
              joblib and loaded for low-latency, high-throughput CPU serving.
    No GPU required — sklearn models trivially fit on any machine.
    """

    def is_compatible(self, model, hw) -> bool:
        """Return True only for sklearn-type model descriptors."""
        return model.model_type == "sklearn"

    def estimate_feasibility(self, model, hw) -> FeasibilityScore:
        """
        Score sklearn suitability for the model.

        sklearn models always fit in memory and never need a GPU.
        The score is slightly lower for neural-family models (MLPClassifier
        etc.) because ONNX Runtime can outperform them significantly after
        export. All other families score at the top.

        Args:
            model: ModelProfile with model_type == "sklearn".
            hw: HardwareProfile (GPU presence is irrelevant here).

        Returns:
            FeasibilityScore in range 0.80–0.95.
        """
        architecture = model.architecture.lower()
        is_neural = any(x in architecture for x in ("mlp", "neural", "perceptron"))

        if is_neural:
            return FeasibilityScore(
                0.80,
                "Works but ONNX Runtime export gives ~2× speedup for MLP models",
                ["Consider: mlfit recommend --export onnx for better throughput"],
            )
        return FeasibilityScore(
            0.95,
            "Excellent — sklearn is the native runtime for this model type",
            [],
        )

    def generate_config(self, model, hw) -> BackendConfig:
        """
        Produce a sklearn joblib serving configuration.

        Sets n_jobs to the full CPU thread count for maximum parallelism
        during both fitting and prediction. The generated command is a
        Python code snippet rather than a CLI command.

        Args:
            model: ModelProfile with model_type == "sklearn".
            hw: HardwareProfile (cpu_threads used for n_jobs).

        Returns:
            BackendConfig with a Python snippet as the command field.
        """
        n_jobs = hw.cpu_threads
        class_name = model.architecture

        params = {
            "n_jobs": n_jobs,
            "backend": "loky",
        }

        command = self._build_code_snippet(class_name, n_jobs)

        return BackendConfig(
            backend="sklearn",
            model_id=model.model_id,
            params=params,
            command=command,
            estimated_vram_gb=0.0,
            estimated_tps=self._estimate_throughput(model, hw),
        )

    def format_key_settings(self, params: dict) -> str:
        """
        Format sklearn key parameters: n_jobs and parallelism backend.

        Args:
            params: BackendConfig.params dict.

        Returns:
            Compact string for the recommendations table.
        """
        parts = []
        if "n_jobs" in params:
            parts.append(f"n_jobs={params['n_jobs']}")
        if "backend" in params:
            parts.append(f"backend={params['backend']}")
        return ", ".join(parts) or "defaults"

    def _build_code_snippet(self, class_name: str, n_jobs: int) -> str:
        """
        Build the Python joblib code snippet for saving and loading the model.

        Args:
            class_name: sklearn estimator class name, e.g. "RandomForestClassifier".
            n_jobs: Number of parallel jobs to use.

        Returns:
            Multi-line Python string.
        """
        return (
            f"from joblib import dump, load\n\n"
            f"# After training (set n_jobs at fit time for parallelism):\n"
            f"# clf = {class_name}(n_jobs={n_jobs})\n"
            f"# clf.fit(X_train, y_train)\n"
            f"dump(clf, \"model.joblib\")\n\n"
            f"# Inference:\n"
            f"model = load(\"model.joblib\")\n"
            f"predictions = model.predict(X)"
        )

    def _estimate_throughput(self, model, hw) -> float:
        """
        Estimate inferences per second based on model family and CPU threads.

        Ensemble models (forests, boosting) have higher per-sample cost
        than linear models. We use the architecture field set by the analyzer.

        Args:
            model: ModelProfile with architecture set to the sklearn class name.
            hw: HardwareProfile (cpu_threads used for parallelism estimate).

        Returns:
            Estimated throughput as a float (inferences/second, batch of 1000).
        """
        architecture = model.architecture.lower()
        thread_scale = min(hw.cpu_threads / 8.0, 4.0)

        if any(x in architecture for x in ("forest", "tree", "gradient", "boost", "xgb", "lgbm")):
            base = 3000.0
        elif any(x in architecture for x in ("linear", "logistic", "ridge", "lasso")):
            base = 50000.0
        elif any(x in architecture for x in ("svm", "svr")):
            base = 5000.0
        elif any(x in architecture for x in ("mlp", "neural", "perceptron")):
            base = 8000.0
        else:
            base = 10000.0

        return round(base * thread_scale, 0)
