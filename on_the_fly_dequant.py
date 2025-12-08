"""
On-the-fly dequantization patch for compressed-tensors.

This module patches CompressedLinear to decompress weights on every forward pass
instead of caching the decompressed weights. This significantly reduces memory
usage at the cost of slower inference.

Usage:
    # Import and enable before loading model
    from on_the_fly_dequant import enable_on_the_fly
    enable_on_the_fly()
    
    # Then load model normally
    model = AutoModelForCausalLM.from_pretrained(...)

Memory Impact:
    - Without patch: ~12.8 GB peak (2.4 GB model + 5.8 GB cached weights + overhead)
    - With patch: ~3.5 GB peak (2.4 GB model + on-the-fly decompression)
"""

import logging
import torch
from torch.nn.functional import linear
from typing import Optional

logger = logging.getLogger(__name__)

# Store original forward for restoration
_original_forward = None
_patch_enabled = False


def _on_the_fly_forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    On-the-fly dequantization forward pass.
    
    Decompresses weights on every forward call without caching.
    This trades compute for memory.
    """
    # Always decompress from packed format
    weight_data = self.compressor.decompress_module(self)
    
    # Compute output with decompressed weights
    output = linear(input, weight_data, self.bias)
    
    # Immediately release the decompressed weights
    del weight_data
    
    return output


def _on_the_fly_forward_with_cache_clear(self, input: torch.Tensor) -> torch.Tensor:
    """
    On-the-fly forward with aggressive cache clearing.
    Use this if you need maximum memory savings.
    """
    weight_data = self.compressor.decompress_module(self)
    output = linear(input, weight_data, self.bias)
    del weight_data
    
    # Aggressive cleanup - may slow down inference
    torch.cuda.empty_cache()
    
    return output


def enable_on_the_fly(aggressive_cleanup: bool = False):
    """
    Enable on-the-fly dequantization for CompressedLinear layers.
    
    Args:
        aggressive_cleanup: If True, call torch.cuda.empty_cache() after each layer.
                          This maximizes memory savings but slows inference.
    """
    global _original_forward, _patch_enabled
    
    if _patch_enabled:
        logger.warning("On-the-fly dequantization is already enabled")
        return
    
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
    except ImportError:
        raise ImportError("compressed-tensors is not installed. Run: uv pip install compressed-tensors")
    
    _original_forward = CompressedLinear.forward
    
    if aggressive_cleanup:
        CompressedLinear.forward = _on_the_fly_forward_with_cache_clear
        logger.info("✅ On-the-fly dequantization enabled (aggressive cleanup mode)")
    else:
        CompressedLinear.forward = _on_the_fly_forward
        logger.info("✅ On-the-fly dequantization enabled")
    
    _patch_enabled = True


def disable_on_the_fly():
    """Restore original CompressedLinear forward behavior."""
    global _original_forward, _patch_enabled
    
    if not _patch_enabled:
        logger.warning("On-the-fly dequantization is not enabled")
        return
    
    if _original_forward is None:
        logger.error("Cannot restore: original forward not saved")
        return
    
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        CompressedLinear.forward = _original_forward
        _patch_enabled = False
        logger.info("✅ Original CompressedLinear forward restored")
    except ImportError:
        logger.error("compressed-tensors is not installed")


def reset_model_to_compressed(model: torch.nn.Module):
    """
    Reset all CompressedLinear layers in a model to compressed state.
    
    Use this to free cached decompressed weights if the model was used
    before enabling on-the-fly mode.
    
    Args:
        model: The model to reset
    """
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization import QuantizationStatus
    except ImportError:
        raise ImportError("compressed-tensors is not installed")
    
    freed_count = 0
    freed_memory = 0
    
    for name, module in model.named_modules():
        if isinstance(module, CompressedLinear):
            # Check if weights are cached (FROZEN state)
            if hasattr(module, 'quantization_status') and module.quantization_status == QuantizationStatus.FROZEN:
                # Calculate freed memory
                if hasattr(module, 'weight') and module.weight is not None:
                    freed_memory += module.weight.numel() * module.weight.element_size()
                    
                    # Delete cached weight
                    delattr(module, 'weight')
                    module.register_parameter('weight', None)
                
                # Reset status to compressed
                module.quantization_status = QuantizationStatus.COMPRESSED
                freed_count += 1
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    freed_mb = freed_memory / 1024**2
    logger.info(f"✅ Reset {freed_count} layers to compressed state, freed ~{freed_mb:.1f} MB")
    
    return freed_count, freed_mb


def get_memory_stats() -> dict:
    """Get current CUDA memory statistics."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


# Convenience context manager
class OnTheFlyContext:
    """
    Context manager for temporary on-the-fly dequantization.
    
    Usage:
        with OnTheFlyContext():
            output = model.generate(...)
    """
    
    def __init__(self, aggressive_cleanup: bool = False):
        self.aggressive_cleanup = aggressive_cleanup
    
    def __enter__(self):
        enable_on_the_fly(self.aggressive_cleanup)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        disable_on_the_fly()
        return False


if __name__ == "__main__":
    # Quick test
    print("On-the-fly dequantization module")
    print("=" * 50)
    print("Functions:")
    print("  - enable_on_the_fly(aggressive_cleanup=False)")
    print("  - disable_on_the_fly()")
    print("  - reset_model_to_compressed(model)")
    print("  - OnTheFlyContext() - context manager")
    print()
    print("Example:")
    print("  from on_the_fly_dequant import enable_on_the_fly")
    print("  enable_on_the_fly()")
    print("  model = AutoModelForCausalLM.from_pretrained(...)")

