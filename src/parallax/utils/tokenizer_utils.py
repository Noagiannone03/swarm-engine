"""Tokenizer loading helpers."""

try:
    # On Mac/Linux mlx_lm is present — use it directly.
    from mlx_lm.tokenizer_utils import load as _mlx_load_tokenizer
except ImportError:  # pragma: no cover - exercised on Windows (mlx absent)
    # mlx_lm transitively imports mlx.core, which has no Windows build. Fall back
    # to the vendored pure-Python copy.
    from parallax.utils._mlx_tokenizer_vendor import load as _mlx_load_tokenizer


def load_tokenizer(model_path, trust_remote_code=True, tokenizer_config_extra=None, **kwargs):
    """
    Wrapper function for MLX load_tokenizer that defaults trust_remote_code to True.
    This is needed for models like Kimi-K2 that contain custom code.
    """
    if tokenizer_config_extra is None:
        tokenizer_config_extra = {}

    if trust_remote_code:
        tokenizer_config_extra = tokenizer_config_extra.copy()
        tokenizer_config_extra["trust_remote_code"] = True

    return _mlx_load_tokenizer(model_path, tokenizer_config_extra=tokenizer_config_extra, **kwargs)
