from mlfit.core.models import ModelProfile


def analyze_model(model_id: str) -> ModelProfile:
    """
    Auto-detect model type and analyze it.

    Supported model ID formats:
      - "meta-llama/Llama-3-8B"           → HuggingFace LLM
      - "Qwen/Qwen2.5-7B-GGUF:Q4_K_M"    → GGUF on HuggingFace
      - "/path/to/model.gguf"             → Local GGUF file
      - "/path/to/model.onnx"             → Local ONNX file
      - "sklearn:RandomForestClassifier"  → sklearn model descriptor
    """
    # sklearn descriptor
    if model_id.startswith("sklearn:"):
        from mlfit.analyzers.sklearn_analyzer import analyze_sklearn
        return analyze_sklearn(model_id)

    # Local ONNX file
    if model_id.endswith(".onnx"):
        from mlfit.analyzers.onnx_analyzer import analyze_onnx
        return analyze_onnx(model_id)

    # Local GGUF file
    if model_id.endswith(".gguf"):
        from mlfit.analyzers.gguf_analyzer import analyze_gguf_file
        return analyze_gguf_file(model_id)

    # HuggingFace GGUF with quant specifier: "Repo/Model:Q4_K_M"
    if ":" in model_id and "/" in model_id:
        from mlfit.analyzers.gguf_analyzer import analyze_hf_gguf
        return analyze_hf_gguf(model_id)

    # Default: HuggingFace LLM
    from mlfit.analyzers.hf_analyzer import analyze_hf_model
    return analyze_hf_model(model_id)
