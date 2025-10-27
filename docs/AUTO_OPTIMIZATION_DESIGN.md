# GPU Auto-Optimization Design

**Core Principle**: Zero-configuration optimization - the system automatically applies optimal settings based on detected GPU environment without any user intervention.

---

## 1. Problem Statement

### Manual Configuration Required (Before)

```bash
# Too many manual settings required
python main.py --input doc.pdf \
    --recognizer deepseek-ocr \
    --recognizer-backend vllm \           # Manual 1
    --tensor-parallel-size 4 \            # Manual 2
    --data-parallel-workers 2 \           # Manual 3
    --gpu-memory-utilization 0.90 \       # Manual 4
    --batch-size 8                        # Manual 5
```

**Problem**: Users must understand GPU environment and know optimal settings.

---

## 2. Goal: Complete Automation

### Ideal User Experience

```bash
# Optimal performance without any configuration
python main.py --input doc.pdf --recognizer deepseek-ocr

# Output:
# 🔍 Detecting GPU environment...
# ✅ Found 8x NVIDIA A100-SXM4-80GB (640GB total)
# 🚀 Auto-optimizing for maximum performance...
#    Backend: vLLM (tensor parallel)
#    Tensor Parallel Size: 4
#    Data Parallel Workers: 2
#    GPU Memory Utilization: 90%
#    Flash Attention: Enabled
# ⚡ Expected speedup: 12x (vs single GPU)
```

**User specifies nothing - system automatically optimizes**

---

## 3. Implementation Design

### 3.1 Architecture

```
[User Input] → [Auto-Optimizer] → [Optimized Execution]
                      ↓
          [GPU Environment Detector]
```

### 3.2 Core Components

#### Component 1: GPU Environment Detector (Singleton)

**File**: `pipeline/gpu_environment.py`

```python
@dataclass
class GPUConfig:
    """Auto-generated optimal GPU configuration."""

    # Detected environment
    has_cuda: bool
    gpu_count: int
    total_memory_gb: float
    compute_capability: tuple[int, int]

    # Auto-optimized settings
    recommended_backend: str  # "vllm", "hf", "pytorch"
    tensor_parallel_size: int
    data_parallel_workers: int
    gpu_memory_utilization: float
    use_flash_attention: bool
    use_bf16: bool

    optimization_strategy: str  # "hybrid_tp4_dp2", etc.
    expected_speedup: float  # vs single GPU sequential


@lru_cache(maxsize=1)
def get_gpu_config() -> GPUConfig:
    """Get cached GPU configuration (singleton).

    Called once at startup and cached.
    All components reference this single source of truth.
    """
    # Detect GPU environment
    # Calculate optimal settings
    # Return configuration
```

**Detection Rules**:
- 8x A100 80GB → vLLM TP=4, DP=2 (hybrid, 12x speedup)
- 4x A100 80GB → vLLM TP=2, DP=2 (hybrid, 6x speedup)
- 2x A100 80GB → hf-ray DP=2 (data parallel, 2.5x speedup)
- 1x A100 80GB → vLLM TP=1 (single GPU, 1.5x speedup)
- No CUDA → pytorch CPU fallback (0.1x speedup)

#### Component 2: Recognizer Auto-Optimization

**File**: `pipeline/recognition/__init__.py`

```python
def create_recognizer(
    name: str,
    backend: str | None = None,  # None = auto-select
    use_auto_optimization: bool = True,
    **kwargs: Any,
) -> Recognizer:
    """Create recognizer with auto-optimization.

    Auto-optimization (when enabled):
    - Auto-select backend based on GPU environment
    - Auto-configure tensor_parallel_size
    - Auto-configure gpu_memory_utilization
    - Auto-select device placement
    """
    if use_auto_optimization:
        gpu_config = get_gpu_config()

        # Auto-select backend if not specified
        if backend is None:
            backend = gpu_config.recommended_backend

        # Inject auto-optimized settings (if not manually overridden)
        if "tensor_parallel_size" not in kwargs:
            kwargs["tensor_parallel_size"] = gpu_config.tensor_parallel_size

        if "gpu_memory_utilization" not in kwargs:
            kwargs["gpu_memory_utilization"] = gpu_config.gpu_memory_utilization

    return _RECOGNIZER_REGISTRY[name](**kwargs)
```

---

## 4. User Scenarios

### Scenario 1: Fully Automatic (Recommended)

```bash
python main.py --input doc.pdf --recognizer deepseek-ocr
```

**Internal Flow**:
1. Detect GPU environment: 8x A100 80GB
2. Auto-select backend: vLLM
3. Auto-calculate settings: TP=4, DP=2
4. Expected speedup: 12x

### Scenario 2: Partial Override

```bash
# Specify backend only, rest auto-optimized
python main.py --input doc.pdf \
    --recognizer deepseek-ocr \
    --recognizer-backend hf
```

**Internal Flow**:
1. Detect GPU environment: 8x A100 80GB
2. Backend: hf (user-specified)
3. Auto-calculate settings: device_map="auto", TP=4

### Scenario 3: Advanced Users (Fully Manual)

```bash
# Disable auto-optimization
python main.py --input doc.pdf \
    --recognizer deepseek-ocr \
    --recognizer-backend vllm \
    --tensor-parallel-size 2 \
    --use-auto-optimization=False
```

### Scenario 4: CPU Environment (Auto Fallback)

```bash
python main.py --input doc.pdf --recognizer deepseek-ocr
```

**Internal Flow**:
1. Detect GPU environment: No CUDA
2. Auto-select backend: pytorch (CPU)
3. Log warning: Performance will be 10x slower

---

## 5. Override Priority

```
1. Explicit CLI arguments (highest priority)
   └─> --tensor-parallel-size 8

2. Environment variables
   └─> VLLM_TENSOR_PARALLEL_SIZE=4

3. Configuration files
   └─> config/gpu_optimization.yaml

4. Auto-detection (default)
   └─> gpu_config.recommended_tensor_parallel_size
```

---

## 6. Key Differences

### Before (Problem)

```python
# User must know everything
llm = LLM(
    model=model_name,
    tensor_parallel_size=???,  # User must calculate?
    gpu_memory_utilization=???,  # How much is safe?
)
```

### After (Automatic)

```python
# Auto-optimized based on GPU environment
gpu_config = get_gpu_config()  # Automatic detection

llm = LLM(
    model=model_name,
    tensor_parallel_size=gpu_config.tensor_parallel_size,  # Automatic
    gpu_memory_utilization=gpu_config.gpu_memory_utilization,  # Automatic
)
```

---

## 7. Expected Performance (8x A100 80GB)

| User Input | Auto Settings | Processing Time | Speedup |
|-----------|--------------|----------------|---------|
| `python main.py --input doc.pdf` | vLLM TP=4, DP=2 | **4.2 min** | **12x** |
| (no manual configuration) | Auto Ray parallelization | | |

---

## 8. Design Philosophy

1. **Zero-configuration by default** - Users should not need to understand GPU architecture
2. **Expert override available** - Advanced users can still manually configure
3. **Fail-safe fallbacks** - Always degrade gracefully (GPU → CPU)
4. **Single source of truth** - One singleton `GPUConfig` referenced everywhere
5. **Logged transparency** - Users see what the system auto-selected

---

**Core Concept**: Users type `python main.py --input doc.pdf` and the system automatically finds and applies optimal settings.
