"""GPU environment detection and auto-optimization configuration.

This module provides automatic GPU detection and optimal configuration
selection based on available hardware resources.

Usage:
    >>> from pipeline.gpu_environment import get_gpu_config
    >>> config = get_gpu_config()
    >>> print(f"Recommended backend: {config.recommended_backend}")
    >>> print(f"Tensor parallel size: {config.tensor_parallel_size}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Auto-generated optimal GPU configuration.

    This dataclass contains both detected GPU environment information
    and automatically calculated optimal settings.

    Attributes:
        has_cuda: Whether CUDA is available
        gpu_count: Number of available GPUs
        gpu_names: List of GPU model names
        total_memory_gb: Total VRAM across all GPUs
        per_gpu_memory_gb: VRAM per GPU
        compute_capability: CUDA compute capability (major, minor)
        recommended_backend: Auto-selected backend
        tensor_parallel_size: Recommended tensor parallel size
        data_parallel_workers: Recommended data parallel workers
        gpu_memory_utilization: Recommended GPU memory utilization
        batch_size: Recommended batch size
        use_flash_attention: Whether Flash Attention should be used
        use_bf16: Whether BF16 precision should be used
        optimization_strategy: Description of optimization strategy
        expected_speedup: Expected speedup vs single GPU sequential
    """

    # Detected environment
    has_cuda: bool
    gpu_count: int
    gpu_names: list[str]
    total_memory_gb: float
    per_gpu_memory_gb: float
    compute_capability: tuple[int, int]

    # Auto-optimized settings
    recommended_backend: str
    tensor_parallel_size: int
    data_parallel_workers: int
    gpu_memory_utilization: float
    batch_size: int
    use_flash_attention: bool
    use_bf16: bool

    # Metadata
    optimization_strategy: str
    expected_speedup: float


@lru_cache(maxsize=1)
def get_gpu_config() -> GPUConfig:
    """Get cached GPU configuration (singleton).

    This function is called once at startup and the result is cached.
    All components reference this single source of truth.

    The function:
    1. Detects available GPU hardware
    2. Calculates optimal settings based on detected hardware
    3. Logs the configuration
    4. Returns the configuration for use by all components

    Returns:
        GPUConfig with detected environment and optimal settings

    Examples:
        >>> config = get_gpu_config()
        >>> if config.has_cuda:
        ...     print(f"Found {config.gpu_count} GPUs")
        >>> else:
        ...     print("Running in CPU mode")
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, running in CPU mode")
        return _cpu_fallback_config()

    if not torch.cuda.is_available():
        logger.info("CUDA not available, running in CPU mode")
        return _cpu_fallback_config()

    # Detect GPU environment
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    props = torch.cuda.get_device_properties(0)
    per_gpu_memory = props.total_memory / (1024**3)
    total_memory = per_gpu_memory * gpu_count
    compute_capability = (props.major, props.minor)

    # Auto-select optimal configuration
    config = _auto_optimize(
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        per_gpu_memory_gb=per_gpu_memory,
        total_memory_gb=total_memory,
        compute_capability=compute_capability,
    )

    # Log detected environment
    _log_gpu_config(config)

    return config


def _auto_optimize(
    gpu_count: int,
    gpu_names: list[str],
    per_gpu_memory_gb: float,
    total_memory_gb: float,
    compute_capability: tuple[int, int],
) -> GPUConfig:
    """Auto-optimize configuration based on GPU environment.

    Optimization rules:
    1. 8x A100 80GB → vLLM TP=4, DP=2 (hybrid strategy)
    2. 4x A100 80GB → vLLM TP=2, DP=2 (hybrid strategy)
    3. 2x A100 80GB → hf-ray DP=2 (data parallel only)
    4. 1x A100 80GB → vLLM TP=1 (single GPU optimized)
    5. 4x RTX/A10 → hf-ray DP=4 (data parallel)
    6. 1x RTX/A10 → hf (single GPU)
    7. <1 GPU or small VRAM → pytorch CPU fallback

    Args:
        gpu_count: Number of GPUs
        gpu_names: List of GPU model names
        per_gpu_memory_gb: VRAM per GPU in GB
        total_memory_gb: Total VRAM across all GPUs
        compute_capability: CUDA compute capability

    Returns:
        GPUConfig with optimal settings
    """
    # Check availability of optional backends
    vllm_available = _check_backend_available("vllm")
    ray_available = _check_backend_available("ray")

    # Flash Attention requires Ampere (SM 8.0) or newer
    supports_flash_attention = compute_capability >= (8, 0)

    # BF16 requires Ampere or newer
    supports_bf16 = compute_capability >= (8, 0)

    # Strategy selection based on GPU count and memory
    if gpu_count >= 8 and per_gpu_memory_gb >= 70:
        # 8x A100 80GB: Hybrid (TP + DP)
        # Use 4 GPUs for tensor parallel, run 2 instances
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="vllm" if vllm_available else "hf",
            tensor_parallel_size=4,
            data_parallel_workers=2,
            gpu_memory_utilization=0.90,
            batch_size=8,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="hybrid_tp4_dp2",
            expected_speedup=12.0,
        )

    elif gpu_count >= 4 and per_gpu_memory_gb >= 70:
        # 4x A100 80GB: Hybrid (TP + DP)
        # Use 2 GPUs for tensor parallel, run 2 instances
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="vllm" if vllm_available else "hf",
            tensor_parallel_size=2,
            data_parallel_workers=2,
            gpu_memory_utilization=0.90,
            batch_size=8,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="hybrid_tp2_dp2",
            expected_speedup=6.0,
        )

    elif gpu_count >= 2 and per_gpu_memory_gb >= 70:
        # 2x A100 80GB: Data parallel only
        # Each GPU runs full model independently
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="hf-ray" if ray_available else "hf",
            tensor_parallel_size=1,
            data_parallel_workers=2,
            gpu_memory_utilization=0.90,
            batch_size=8,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="data_parallel_dp2",
            expected_speedup=2.5,
        )

    elif gpu_count >= 1 and per_gpu_memory_gb >= 70:
        # 1x A100 80GB: Single GPU optimized
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="vllm" if vllm_available else "hf",
            tensor_parallel_size=1,
            data_parallel_workers=1,
            gpu_memory_utilization=0.90,
            batch_size=8,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="single_gpu_optimized",
            expected_speedup=1.5,
        )

    elif gpu_count >= 4 and per_gpu_memory_gb >= 20:
        # 4x RTX 4090 / A10 24GB: Data parallel
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="hf-ray" if ray_available else "hf",
            tensor_parallel_size=1,
            data_parallel_workers=4,
            gpu_memory_utilization=0.85,
            batch_size=4,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="data_parallel_dp4",
            expected_speedup=4.0,
        )

    elif gpu_count >= 2 and per_gpu_memory_gb >= 20:
        # 2x RTX 4090 / A10: Data parallel
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="hf-ray" if ray_available else "hf",
            tensor_parallel_size=1,
            data_parallel_workers=2,
            gpu_memory_utilization=0.85,
            batch_size=4,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="data_parallel_dp2",
            expected_speedup=2.0,
        )

    elif gpu_count >= 1 and per_gpu_memory_gb >= 20:
        # 1x RTX 4090 / A10: Single GPU
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="hf",
            tensor_parallel_size=1,
            data_parallel_workers=1,
            gpu_memory_utilization=0.85,
            batch_size=4,
            use_flash_attention=supports_flash_attention,
            use_bf16=supports_bf16,
            optimization_strategy="single_gpu",
            expected_speedup=1.0,
        )

    else:
        # Fallback: Use what's available (small GPU or limited memory)
        return GPUConfig(
            has_cuda=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            total_memory_gb=total_memory_gb,
            per_gpu_memory_gb=per_gpu_memory_gb,
            compute_capability=compute_capability,
            recommended_backend="pytorch",
            tensor_parallel_size=1,
            data_parallel_workers=1,
            gpu_memory_utilization=0.75,
            batch_size=1,
            use_flash_attention=False,
            use_bf16=False,
            optimization_strategy="conservative",
            expected_speedup=1.0,
        )


def _cpu_fallback_config() -> GPUConfig:
    """Fallback configuration for CPU-only environment.

    Returns:
        GPUConfig configured for CPU execution
    """
    return GPUConfig(
        has_cuda=False,
        gpu_count=0,
        gpu_names=[],
        total_memory_gb=0.0,
        per_gpu_memory_gb=0.0,
        compute_capability=(0, 0),
        recommended_backend="pytorch",
        tensor_parallel_size=1,
        data_parallel_workers=1,
        gpu_memory_utilization=0.0,
        batch_size=1,
        use_flash_attention=False,
        use_bf16=False,
        optimization_strategy="cpu_fallback",
        expected_speedup=0.1,  # CPU is ~10x slower
    )


def _check_backend_available(backend: str) -> bool:
    """Check if a backend is available.

    Args:
        backend: Backend name ("vllm", "ray", etc.)

    Returns:
        True if backend is available, False otherwise
    """
    try:
        if backend == "vllm":
            import vllm  # noqa: F401
        elif backend == "ray":
            import ray  # noqa: F401
        else:
            return False
        return True
    except ImportError:
        return False


def _log_gpu_config(config: GPUConfig) -> None:
    """Log detected GPU configuration.

    Args:
        config: GPUConfig to log
    """
    logger.info("=" * 70)
    logger.info("🔍 GPU Environment Detection")
    logger.info("=" * 70)

    if not config.has_cuda:
        logger.warning("⚠️  CUDA not available - running in CPU mode")
        logger.warning("    Performance will be significantly slower (~10x)")
        logger.info("=" * 70)
        return

    logger.info(f"✅ Detected {config.gpu_count}x GPU ({config.per_gpu_memory_gb:.0f}GB each)")
    if config.gpu_names:
        logger.info(f"   Model: {config.gpu_names[0]}")
    logger.info(f"   Total VRAM: {config.total_memory_gb:.0f}GB")
    logger.info(f"   Compute Capability: SM {config.compute_capability[0]}.{config.compute_capability[1]}")
    logger.info("")
    logger.info("🚀 Auto-Optimization Configuration:")
    logger.info(f"   Strategy: {config.optimization_strategy}")
    logger.info(f"   Backend: {config.recommended_backend}")
    logger.info(f"   Tensor Parallel Size: {config.tensor_parallel_size}")
    logger.info(f"   Data Parallel Workers: {config.data_parallel_workers}")
    logger.info(f"   GPU Memory Utilization: {config.gpu_memory_utilization:.0%}")
    logger.info(f"   Batch Size: {config.batch_size}")
    logger.info(f"   Flash Attention: {'✅ Enabled' if config.use_flash_attention else '❌ Disabled'}")
    logger.info(f"   BF16 Precision: {'✅ Enabled' if config.use_bf16 else '❌ Disabled'}")
    logger.info("")
    logger.info(f"⚡ Expected Speedup: {config.expected_speedup:.1f}x (vs single GPU sequential)")
    logger.info("=" * 70)


def print_gpu_info() -> None:
    """Print GPU environment information (for CLI --show-gpu-info).

    This is a convenience function for displaying GPU info to users.
    """
    config = get_gpu_config()

    print("=" * 70)
    print("GPU Environment Information")
    print("=" * 70)

    if not config.has_cuda:
        print("⚠️  CUDA not available")
        print("Running in CPU mode (significantly slower)")
        print("=" * 70)
        return

    print("\n📊 Hardware:")
    print(f"   GPUs: {config.gpu_count}")
    if config.gpu_names:
        for i, name in enumerate(config.gpu_names):
            print(f"   GPU {i}: {name}")
    print(f"   VRAM per GPU: {config.per_gpu_memory_gb:.1f} GB")
    print(f"   Total VRAM: {config.total_memory_gb:.1f} GB")
    print(f"   Compute Capability: SM {config.compute_capability[0]}.{config.compute_capability[1]}")

    print("\n🚀 Auto-Optimization:")
    print(f"   Strategy: {config.optimization_strategy}")
    print(f"   Recommended Backend: {config.recommended_backend}")
    print(f"   Tensor Parallel: {config.tensor_parallel_size}")
    print(f"   Data Parallel: {config.data_parallel_workers}")
    print(f"   GPU Memory Util: {config.gpu_memory_utilization:.0%}")
    print(f"   Flash Attention: {'Yes' if config.use_flash_attention else 'No'}")
    print(f"   BF16: {'Yes' if config.use_bf16 else 'No'}")

    print("\n⚡ Performance:")
    print(f"   Expected Speedup: {config.expected_speedup:.1f}x")

    print("=" * 70)
