from mlfit.core.models import ModelProfile


def analyze_sklearn(model_id: str) -> ModelProfile:
    """
    Analyze a sklearn model descriptor.

    model_id format: "sklearn:ClassName"
    e.g., "sklearn:RandomForestClassifier", "sklearn:XGBClassifier"
    """
    class_name = model_id.split(":", 1)[1] if ":" in model_id else model_id

    # Classify the model family
    model_family = _classify_model(class_name)

    # sklearn models are tiny compared to deep learning models
    # They don't use GPUs and don't have "parameters" in the same sense
    profile = ModelProfile(
        model_id=model_id,
        model_type="sklearn",
        num_parameters=0,           # N/A for sklearn
        dtype="float64",            # sklearn default
        architecture=class_name,
        max_context_length=0,       # N/A
        num_hidden_layers=0,
        num_attention_heads=0,
        num_key_value_heads=0,
        hidden_size=0,
        quant_type=None,
    )
    profile.estimated_size_gb = 0.001  # <1MB for most sklearn models
    return profile


def _classify_model(class_name: str) -> str:
    """Classify the sklearn model family for backend selection."""
    name = class_name.lower()
    if any(x in name for x in ["forest", "tree", "gradient", "boost", "xgb", "lgbm"]):
        return "ensemble"
    if any(x in name for x in ["linear", "logistic", "ridge", "lasso", "svm", "svr"]):
        return "linear"
    if any(x in name for x in ["kmeans", "dbscan", "cluster"]):
        return "clustering"
    if any(x in name for x in ["mlp", "neural", "perceptron"]):
        return "neural"
    return "classical"
