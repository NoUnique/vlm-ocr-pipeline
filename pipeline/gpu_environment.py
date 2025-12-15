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


@dataclass
class _GPUProfile:
    """GPU optimization profile for auto-configuration."""

    min_gpus: int
    min_memory_gb: float
    backend_with_vllm: str
    backend_without_vllm: str
    tensor_parallel: int
    data_parallel: int
    memory_util: float
    batch_size: int
    strategy: str
    speedup: float


# GPU profiles ordered by priority (most capable first)
_GPU_PROFILES: list[_GPUProfile] = [
    # 8x A100 80GB: Hybrid (TP=4, DP=2)
    _GPUProfile(8, 70, "vllm", "hf", 4, 2, 0.90, 8, "hybrid_tp4_dp2", 12.0),
    # 4x A100 80GB: Hybrid (TP=2, DP=2)
    _GPUProfile(4, 70, "vllm", "hf", 2, 2, 0.90, 8, "hybrid_tp2_dp2", 6.0),
    # 2x A100 80GB: Data parallel only
    _GPUProfile(2, 70, "hf-ray", "hf", 1, 2, 0.90, 8, "data_parallel_dp2", 2.5),
    # 1x A100 80GB: Single GPU optimized
    _GPUProfile(1, 70, "vllm", "hf", 1, 1, 0.90, 8, "single_gpu_optimized", 1.5),
    # 4x RTX 4090 / A10 24GB: Data parallel
    _GPUProfile(4, 20, "hf-ray", "hf", 1, 4, 0.85, 4, "data_parallel_dp4", 4.0),
    # 2x RTX 4090 / A10: Data parallel
    _GPUProfile(2, 20, "hf-ray", "hf", 1, 2, 0.85, 4, "data_parallel_dp2", 2.0),
    # 1x RTX 4090 / A10: Single GPU
    _GPUProfile(1, 20, "hf", "hf", 1, 1, 0.85, 4, "single_gpu", 1.0),
]


def _find_matching_profile(gpu_count: int, per_gpu_memory_gb: float) -> _GPUProfile | None:
    """Find the best matching GPU profile.

    Args:
        gpu_count: Number of GPUs available
        per_gpu_memory_gb: Memory per GPU in GB

    Returns:
        Matching profile or None if no match
    """
    for profile in _GPU_PROFILES:
        if gpu_count >= profile.min_gpus and per_gpu_memory_gb >= profile.min_memory_gb:
            return profile
    return None


def _build_gpu_config(
    profile: _GPUProfile | None,
    gpu_count: int,
    gpu_names: list[str],
    per_gpu_memory_gb: float,
    total_memory_gb: float,
    compute_capability: tuple[int, int],
    vllm_available: bool,
    ray_available: bool,
) -> GPUConfig:
    """Build GPUConfig from profile and environment.

    Args:
        profile: Matched GPU profile (None for fallback)
        gpu_count: Number of GPUs
        gpu_names: GPU model names
        per_gpu_memory_gb: Memory per GPU
        total_memory_gb: Total GPU memory
        compute_capability: CUDA compute capability
        vllm_available: Whether vLLM is available
        ray_available: Whether Ray is available

    Returns:
        Configured GPUConfig instance
    """
    # Hardware capabilities
    supports_flash_attention = compute_capability >= (8, 0)  # Ampere or newer
    supports_bf16 = compute_capability >= (8, 0)

    # Fallback for no matching profile
    if profile is None:
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

    # Select backend based on availability
    backend = profile.backend_with_vllm
    if backend == "vllm" and not vllm_available:
        backend = profile.backend_without_vllm
    elif backend == "hf-ray" and not ray_available:
        backend = "hf"

    return GPUConfig(
        has_cuda=True,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        total_memory_gb=total_memory_gb,
        per_gpu_memory_gb=per_gpu_memory_gb,
        compute_capability=compute_capability,
        recommended_backend=backend,
        tensor_parallel_size=profile.tensor_parallel,
        data_parallel_workers=profile.data_parallel,
        gpu_memory_utilization=profile.memory_util,
        batch_size=profile.batch_size,
        use_flash_attention=supports_flash_attention,
        use_bf16=supports_bf16,
        optimization_strategy=profile.strategy,
        expected_speedup=profile.speedup,
    )


def _auto_optimize(
    gpu_count: int,
    gpu_names: list[str],
    per_gpu_memory_gb: float,
    total_memory_gb: float,
    compute_capability: tuple[int, int],
) -> GPUConfig:
    """Auto-optimize configuration based on GPU environment.

    Selects the best optimization profile based on available hardware:
    - High-memory multi-GPU: Hybrid tensor + data parallelism
    - Mid-memory multi-GPU: Data parallelism only
    - Single GPU: Optimized single-GPU settings
    - Low memory: Conservative fallback

    Args:
        gpu_count: Number of GPUs
        gpu_names: List of GPU model names
        per_gpu_memory_gb: VRAM per GPU in GB
        total_memory_gb: Total VRAM across all GPUs
        compute_capability: CUDA compute capability

    Returns:
        GPUConfig with optimal settings
    """
    # Check backend availability
    vllm_available = _check_backend_available("vllm")
    ray_available = _check_backend_available("ray")

    # Find matching profile
    profile = _find_matching_profile(gpu_count, per_gpu_memory_gb)

    # Build configuration
    return _build_gpu_config(
        profile=profile,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        per_gpu_memory_gb=per_gpu_memory_gb,
        total_memory_gb=total_memory_gb,
        compute_capability=compute_capability,
        vllm_available=vllm_available,
        ray_available=ray_available,
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
            import vllm  # type: ignore[import-not-found]  # noqa: F401
        elif backend == "ray":
            import ray  # type: ignore[import-not-found]  # noqa: F401
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
