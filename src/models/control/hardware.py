"""Fail before loading checkpoints when the allocated GPU cannot run the protocol."""


def validate_gpu(model_key, config, training=False):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for model execution')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The selected protocol requires BF16 support')
    key = 'training_min_vram_gib' if training else 'inference_min_vram_gib'
    minimum = config.get(key, {}).get(model_key, 0)
    if minimum < 0:
        raise ValueError('Minimum GPU memory must be nonnegative')
    available = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if available < minimum:
        raise RuntimeError(
            f'{model_key} requires an allocation with at least {minimum} GiB VRAM '
            f'for this configuration; received {available:.1f} GiB'
        )
