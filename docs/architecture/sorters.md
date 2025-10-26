# Sorters

Reading order analysis algorithms for document layout.

## Supported Sorters

| Sorter | Algorithm | Multi-Column | Speed | Quality |
|--------|-----------|--------------|-------|---------|
| `pymupdf` | Font analysis | ✅ | ⚡⚡⚡ | ⭐⭐⭐ |
| `mineru-xycut` | XY-Cut | ❌ | ⚡⚡⚡ | ⭐⭐ |
| `mineru-layoutreader` | LayoutLMv3 | ✅ | ⚡⚡ | ⭐⭐⭐⭐ |
| `mineru-vlm` | VLM reasoning | ✅ | ⚡ | ⭐⭐⭐⭐ |
| `olmocr-vlm` | VLM reasoning | ✅ | ⚡ | ⭐⭐⭐⭐ |
| `paddleocr-doclayout-v2` | Pointer network | ✅ | ⚡⚡ | ⭐⭐⭐⭐ |

!!! note
    Detailed sorter documentation coming soon. See [Architecture Overview](overview.md) for more information.
