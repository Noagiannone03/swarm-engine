"""Tokenizer loading helpers."""

try:
    from mlx_lm.tokenizer_utils import load as _mlx_load_tokenizer
except ImportError:  # MLX-LM is not available in native Windows CUDA runtimes.
    _mlx_load_tokenizer = None


def load_tokenizer(
    model_path,
    trust_remote_code=True,
    tokenizer_config_extra=None,
    eos_token_ids=None,
    **kwargs,
):
    """
    Wrapper function for MLX load_tokenizer that defaults trust_remote_code to True.
    This is needed for models like Kimi-K2 that contain custom code.
    """
    if tokenizer_config_extra is None:
        tokenizer_config_extra = {}

    if trust_remote_code:
        tokenizer_config_extra = tokenizer_config_extra.copy()
        tokenizer_config_extra["trust_remote_code"] = True

    if _mlx_load_tokenizer is not None:
        return _mlx_load_tokenizer(
            model_path,
            tokenizer_config_extra=tokenizer_config_extra,
            eos_token_ids=eos_token_ids,
            **kwargs,
        )

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra, **kwargs)
