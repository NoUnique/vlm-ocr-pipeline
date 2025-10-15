# Region Type System Documentation

This document describes the standardized region type system used throughout the VLM OCR Pipeline.

## Overview

All detectors in this pipeline use a **unified type system based on MinerU 2.5 VLM**, which provides the most comprehensive set of region types (25+ types). Detector-specific types are automatically mapped to standardized types for consistent processing throughout the pipeline.

### Implementation

- **Type definitions:** [`pipeline/types.py`](pipeline/types.py) - `RegionType` class
- **Type mapper:** [`pipeline/types.py`](pipeline/types.py) - `RegionTypeMapper` class
- **Detector integration:** Applied in all detectors during region creation
- **Output conversion:** [`pipeline/conversion/output/markdown/__init__.py`](pipeline/conversion/output/markdown/__init__.py)

### Type Mapping Process

1. Detector returns regions with native type names (e.g., `"plain text"`, `"abandon"`, `"figure_caption"`)
2. `RegionTypeMapper.map_type(type, detector_name)` converts to standardized type:
   - `"plain text"` → `"text"`
   - `"abandon"` → `"discarded"`
   - `"figure_caption"` → `"image_caption"`
3. All downstream pipeline stages use standardized types
4. Markdown conversion handles all 25+ standardized types

---

## Standardized Region Types

Based on MinerU 2.5 VLM (`external/MinerU/mineru/utils/enum_class.py`):

### Content Types
- `text` - Body text paragraphs
- `title` - Document/section titles

### Figure Types
- `image` - Image regions
- `image_body` - Image content area
- `image_caption` - Image captions
- `image_footnote` - Image footnotes

### Table Types
- `table` - Table regions
- `table_body` - Table content area
- `table_caption` - Table captions
- `table_footnote` - Table footnotes

### Equation Types
- `interline_equation` - Display equations (block-level)
- `inline_equation` - Inline equations

### Code Types
- `code` - Code blocks
- `code_body` - Code content area
- `code_caption` - Code captions
- `algorithm` - Algorithm pseudocode

### List Types
- `list` - List items

### Page Elements
- `header` - Page headers (not content headings)
- `footer` - Page footers
- `page_number` - Page numbers
- `page_footnote` - Page-level footnotes

### Reference Types
- `ref_text` - Reference text
- `phonetic` - Phonetic annotations
- `aside_text` - Aside/sidebar text
- `index` - Index entries

### Special Types
- `discarded` - Discarded/invalid content

---

## Detector Support Matrix

| Region Type | DocLayoutYOLO | MinerU DocLayoutYOLO | MinerU VLM 2.5 | Standardized Type |
|-------------|:-------------:|:--------------------:|:--------------:|:------------------|
| **Content** |
| `text` | ✓ | - | ✓ | `text` |
| `plain text` | ✓ | ✓ | - | `text` |
| `title` | ✓ | ✓ | ✓ | `title` |
| **Figures** |
| `image` | ✓ | - | ✓ | `image` |
| `figure` | ✓ | ✓ | - | `image` |
| `image_body` | - | - | ✓ | `image_body` |
| `image_caption` | - | - | ✓ | `image_caption` |
| `figure_caption` | - | ✓ | - | `image_caption` |
| `image_footnote` | - | - | ✓ | `image_footnote` |
| **Tables** |
| `table` | ✓ | ✓ | ✓ | `table` |
| `table_body` | - | - | ✓ | `table_body` |
| `table_caption` | - | ✓ | ✓ | `table_caption` |
| `table_footnote` | - | ✓ | ✓ | `table_footnote` |
| **Equations** |
| `equation` | ✓ | - | - | `interline_equation` |
| `interline_equation` | - | - | ✓ | `interline_equation` |
| `inline_equation` | - | - | ✓ | `inline_equation` |
| `isolate_formula` | - | ✓ | - | `interline_equation` |
| `formula_caption` | - | ✓ | - | `image_caption` |
| **Code** |
| `code` | - | - | ✓ | `code` |
| `code_body` | - | - | ✓ | `code_body` |
| `code_caption` | - | - | ✓ | `code_caption` |
| `algorithm` | - | - | ✓ | `algorithm` |
| **Lists** |
| `list` | ✓ | - | ✓ | `list` |
| `list_item` | ✓ | - | - | `list` |
| **Page Elements** |
| `header` | - | - | ✓ | `header` |
| `footer` | - | - | ✓ | `footer` |
| `page_number` | - | - | ✓ | `page_number` |
| `page_footnote` | - | - | ✓ | `page_footnote` |
| **References** |
| `ref_text` | - | - | ✓ | `ref_text` |
| `phonetic` | - | - | ✓ | `phonetic` |
| `aside_text` | - | - | ✓ | `aside_text` |
| `index` | - | - | ✓ | `index` |
| **Special** |
| `abandon` | - | ✓ | - | `discarded` |
| `discarded` | - | - | ✓ | `discarded` |

**Legend:**
- ✓ = Natively supported by detector
- `-` = Not supported

---

## Detector Specifications

### DocLayout-YOLO (Project)

**Implementation:** [`pipeline/layout/detection/doclayout_yolo.py`](pipeline/layout/detection/doclayout_yolo.py)

**Type System:** Model-dependent (loaded from YOLO model weights)

**Common Types:**
- `title`, `plain text`, `figure`, `table`, `equation`, `list`, `list_item`

**Notes:**
- Actual types depend on model's `class_names` (determined at runtime)
- DocStructBench model typically includes 6-8 types

### MinerU DocLayout-YOLO

**Implementation:** [`pipeline/layout/detection/mineru/doclayout_yolo.py`](pipeline/layout/detection/mineru/doclayout_yolo.py)

**Type System:** Fixed 10-type system

**Supported Types:**
```python
{
    0: "title",
    1: "plain text",
    2: "abandon",           # → discarded
    3: "figure",            # → image
    4: "figure_caption",    # → image_caption
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",   # → interline_equation
    9: "formula_caption",   # → image_caption
}
```

### MinerU VLM 2.5

**Implementation:** [`pipeline/layout/detection/mineru/vlm.py`](pipeline/layout/detection/mineru/vlm.py)

**Type System:** 25 standardized types (canonical reference)

**Features:**
- Most comprehensive type coverage
- Includes code, algorithm, and specialized academic types
- Distinguishes between body/caption/footnote for images, tables, and code
- Separate page elements (header, footer, page_number)
- Already uses standardized types (identity mapping)

### olmOCR VLM

**Implementation:** Used via sorters in [`pipeline/layout/ordering/olmocr/`](pipeline/layout/ordering/olmocr/)

**Type System:** Single type (`text`)

**Special Behavior:**
- VLM generates complete Markdown directly
- Output includes YAML front matter with metadata
- No region-level type distinction
- All content marked as `text` type

**Example Output:**
```yaml
---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
---

# Chapter 1

Content with **formatting**, $$equations$$, and tables.
```

---

## Markdown Conversion Rules

Defined in [`pipeline/conversion/output/markdown/__init__.py`](pipeline/conversion/output/markdown/__init__.py):

| Type | Markdown Format | Example |
|------|----------------|---------|
| `title` | `# {text}` | `# Introduction` |
| `text` | `{text}` | Plain text |
| `image`, `image_body` | `**Figure:** {text}` | `**Figure:** Chart showing...` |
| `image_caption` | `**Figure:** {text}` | `**Figure:** Monthly sales` |
| `image_footnote` | `*{text}*` | `*Source: 2024 report*` |
| `table`, `table_body` | `{text}` or `**Table:**\n\n{text}` | Markdown table or formatted |
| `table_caption` | `**Table:** {text}` | `**Table:** Q1 Results` |
| `table_footnote` | `*{text}*` | `*n=100*` |
| `interline_equation` | `$${text}$$` | `$$E = mc^2$$` |
| `inline_equation` | `${text}$` | `$x^2$` |
| `code`, `code_body`, `algorithm` | ` ```\n{text}\n``` ` | Code block |
| `code_caption` | `**Code:** {text}` | `**Code:** Algorithm 1` |
| `list` | `- {text}` | `- Item` |
| `header`, `footer`, `page_number` | *(skipped)* | - |
| `ref_text` | `{text}` | Plain text |
| `phonetic`, `aside_text` | `*{text}*` | `*pronunciation*` |
| `page_footnote` | `*{text}*` | `*footnote*` |
| `index` | `{text}` | Plain text |
| `discarded` | *(skipped)* | - |

---

## Adding New Detectors

To integrate a new detector with the type system:

1. **Implement the Detector protocol** ([`pipeline/types.py`](pipeline/types.py)):
   ```python
   class MyDetector:
       def detect(self, image: np.ndarray) -> list[Region]:
           # Return regions with native types
           pass
   ```

2. **Add type mapping** to `RegionTypeMapper`:
   ```python
   MY_DETECTOR_MAP: dict[str, str] = {
       "native_type_1": RegionType.TEXT,
       "native_type_2": RegionType.IMAGE,
       # ...
   }
   ```

3. **Apply mapping** in detector's `_to_region()` method:
   ```python
   standardized_type = RegionTypeMapper.map_type(
       original_type,
       "my-detector"
   )
   ```

4. **Update mapping method** in `RegionTypeMapper.map_type()`:
   ```python
   elif detector_name == "my-detector":
       mapping_dict = cls.MY_DETECTOR_MAP
   ```

That's it! The rest of the pipeline will automatically handle your detector's output using standardized types.
