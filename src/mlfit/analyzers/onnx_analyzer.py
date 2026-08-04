from mlfit.core.models import ModelProfile


def analyze_onnx(path: str) -> ModelProfile:
    """Analyze a local .onnx file."""
    try:
        import onnx
        import numpy as np

        model = onnx.load(path)
        total_bytes = 0
        for init in model.graph.initializer:
            try:
                dtype_name = onnx.TensorProto.DataType.Name(init.data_type).lower()
                elem_size = np.dtype(dtype_name).itemsize
                total_bytes += int(np.prod(init.dims)) * elem_size
            except Exception:
                pass

        size_gb = total_bytes / 1e9

        profile = ModelProfile(
            model_id=path,
            model_type="onnx",
            num_parameters=total_bytes // 4,  # assume float32
            dtype="float32",
            architecture="ONNX",
            max_context_length=0,
            num_hidden_layers=0,
            num_attention_heads=0,
            num_key_value_heads=0,
            hidden_size=0,
            quant_type=None,
        )
        profile.estimated_size_gb = size_gb
        return profile

    except ImportError:
        raise ImportError("onnx is required: pip install onnx")
