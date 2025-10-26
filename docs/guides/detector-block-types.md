# Block Type System Documentation

This document describes the standardized block type system used throughout the VLM OCR Pipeline.

## Overview

All detectors in this pipeline use a **unified type system** that supports 25+ block types. The type system is based on comprehensive document element categories, with detector-specific types automatically mapped to standardized types for consistent processing throughout the pipeline.

### Implementation

- **Type definitions:** `pipeline/types.py` - `Block` dataclass with `type: str` field
- **Detector integration:** Applied in all detectors during block creation
- **Output conversion:** `pipeline/conversion/output/markdown/__init__.py`

### Type Mapping Process

1. Detector returns blocks with native type names (e.g., `"plain text"`, `"abandon"`, `"figure_caption"`)
2. Type normalization converts to standardized type:
   - `"plain text"` → `"text"`
   - `"abandon"` → `"discarded"`
   - `"figure_caption"` → `"image_caption"`
3. All downstream pipeline stages use standardized types
4. Markdown conversion handles all 25+ standardized types

---

## Standardized Block Types

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

| Block Type | DocLayoutYOLO | MinerU DocLayoutYOLO | MinerU VLM 2.5 | PaddleOCR PP-DocLayoutV2 |
|------------|---------------|---------------------|----------------|--------------------------|
| **Content Types** |
| `text` | `plain text` | `plain text` | `text` | `text`<br>`vertical_text`<br>`abstract`<br>`contents` |
| `title` | `title` | `title` | `title` | `doc_title`<br>`paragraph_title` |
| **Figure Types** |
| `image` | `figure` | `figure` | `image` | `image`<br>`chart`<br>`seal` |
| `image_body` | - | - | `image_body` | - |
| `image_caption` | - | `figure_caption`<br>`formula_caption` | `image_caption` | `figure_title` |
| `image_footnote` | - | - | `image_footnote` | - |
| **Table Types** |
| `table` | `table` | `table` | `table` | `table` |
| `table_body` | - | - | `table_body` | - |
| `table_caption` | - | `table_caption` | `table_caption` | `figure_title` |
| `table_footnote` | - | `table_footnote` | `table_footnote` | - |
| **Equation Types** |
| `interline_equation` | `equation` | `isolate_formula` | `interline_equation` | `display_formula`<br>`formula_number` |
| `inline_equation` | - | - | `inline_equation` | `inline_formula` |
| **Code Types** |
| `code` | - | - | `code` | - |
| `code_body` | - | - | `code_body` | - |
| `code_caption` | - | - | `code_caption` | - |
| `algorithm` | - | - | `algorithm` | `algorithm` |
| **List Types** |
| `list` | `list`<br>`list_item` | - | `list` | - |
| **Page Elements** |
| `header` | - | - | `header` | `header`<br>`header_image` |
| `footer` | - | - | `footer` | `footer`<br>`footer_image` |
| `page_number` | - | - | `page_number` | `page_number` |
| `page_footnote` | - | - | `page_footnote` | `footnote` |
| **Reference Types** |
| `ref_text` | - | - | `ref_text` | `reference`<br>`reference_content` |
| `phonetic` | - | - | `phonetic` | - |
| `aside_text` | - | - | `aside_text` | `aside_text` |
| `index` | - | - | `index` | - |
| **Special Types** |
| `discarded` | - | `abandon` | `discarded` | - |

**Legend:**
- Detector-specific type names shown in backticks
- `-` = Not supported by detector
- Multiple types per cell separated by line breaks

---

## Detector Specifications

### DocLayout-YOLO (Project)

**Implementation:** `pipeline/layout/detection/doclayout_yolo.py`

**Type System:** Model-dependent (loaded from YOLO model weights)

**Common Types:**
- `title`, `plain text`, `figure`, `table`, `equation`, `list`, `list_item`

**Notes:**
- Actual types depend on model's `class_names` (determined at runtime)
- DocStructBench model typically includes 6-8 types

### PaddleOCR PP-DocLayoutV2

**Implementation:** `pipeline/layout/detection/paddleocr/doclayout_v2.py`

**Type System:** Fixed 25-type system with integrated reading order

**Supported Types:**
```python
{
    # Titles and text
    "doc_title": "title",
    "paragraph_title": "title",
    "text": "text",
    "vertical_text": "text",
    "aside_text": "aside_text",

    # Page elements
    "page_number": "page_number",
    "header": "header",
    "footer": "footer",
    "header_image": "header",
    "footer_image": "footer",

    # Structural elements
    "abstract": "text",
    "contents": "text",
    "reference": "ref_text",
    "reference_content": "ref_text",
    "footnote": "page_footnote",

    # Math and formulas
    "inline_formula": "inline_equation",
    "display_formula": "interline_equation",
    "formula_number": "interline_equation",
    "algorithm": "algorithm",

    # Visual elements
    "image": "image",
    "table": "table",
    "chart": "image",
    "seal": "image",

    # Captions (unified as figure_title)
    "figure_title": "image_caption",  # Includes figure/table/chart captions
}
```

**Features:**
- **Unified model**: RT-DETR-L based PP-DocLayout_plus-L (81.4 mAP)
- **Reading order**: Built-in pointer network (6 Transformer layers)
- **Output is pre-sorted**: Blocks already have `order` field set
- **25 categories**: Most comprehensive category coverage
- **Model size**: 203.8 MB
- **Supports**: Chinese, English, Japanese, and vertical text documents
- **Trained on**: Papers, magazines, PPTs, contracts, and diverse document types

**Notes:**
- No additional sorter needed - reading order is built-in
- `figure_title` applies to all captions (images, tables, charts)
- Distinguishes between doc-level and paragraph-level titles

### MinerU DocLayout-YOLO

**Implementation:** `pipeline/layout/detection/mineru/doclayout_yolo.py`

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

**Implementation:** `pipeline/layout/detection/mineru/vlm.py`

**Type System:** 25 standardized types (canonical reference)

**Features:**
- Most comprehensive type coverage
- Includes code, algorithm, and specialized academic types
- Distinguishes between body/caption/footnote for images, tables, and code
- Separate page elements (header, footer, page_number)
- Already uses standardized types (identity mapping)

### olmOCR VLM

**Implementation:** Used via sorters in `pipeline/layout/ordering/olmocr/`

**Type System:** Single type (`text`)

**Special Behavior:**
- VLM generates complete Markdown directly
- Output includes YAML front matter with metadata
- No block-level type distinction
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

Defined in `pipeline/conversion/output/markdown/__init__.py`:

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

1. **Implement the Detector protocol** (see `pipeline/types.py`):
   ```python
   class MyDetector:
       def detect(self, image: np.ndarray) -> list[Block]:
           # Return blocks with native type names
           pass
   ```

2. **Create type mapping dictionary** (if detector uses non-standard types):
   ```python
   _TYPE_MAP = {
       "native_type_1": "text",
       "native_type_2": "image",
       "native_caption": "image_caption",
       # Map all detector-specific types to standardized types
   }
   ```

3. **Apply mapping** in detector's block creation:
   ```python
   def detect(self, image: np.ndarray) -> list[Block]:
       # Get detections from model
       raw_detections = self.model.predict(image)

       # Convert to Block objects with standardized types
       blocks = []
       for det in raw_detections:
           # Map type using _TYPE_MAP
           standardized_type = self._TYPE_MAP.get(det.type, det.type)

           block = Block(
               type=standardized_type,
               bbox=BBox.from_xyxy(det.x0, det.y0, det.x1, det.y1),
               detection_confidence=det.confidence,
               order=None,  # Set by sorter
               column_index=None,  # Set by sorter if multi-column
               text=None,  # Set by recognizer
               corrected_text=None,
               correction_ratio=None,
               source="my-detector",
           )
           blocks.append(block)

       return blocks
   ```

4. **Register in factory** (see `pipeline/layout/detection/__init__.py`):
   ```python
   def create_detector(name: str, **kwargs) -> Detector:
       if name == "my-detector":
           from .my_detector import MyDetector  # noqa: PLC0415
           return MyDetector(**kwargs)
       # ...
   ```

That's it! The rest of the pipeline will automatically handle your detector's output using standardized types.
