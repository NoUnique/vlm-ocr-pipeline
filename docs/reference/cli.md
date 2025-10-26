# CLI Reference

Complete command-line interface documentation.

## Main Command

```bash
python main.py [OPTIONS]
```

## Options

For complete CLI documentation, see [Basic Usage](../getting-started/basic-usage.md).

### Input

- `--input PATH`: Input file or directory
- `--max-pages N`: Process first N pages
- `--page-range START-END`: Process page range
- `--pages LIST`: Process specific pages (comma-separated)

### Backend

- `--backend {openai,gemini}`: VLM backend
- `--model MODEL`: Model name
- `--gemini-tier {free,tier1,tier2,tier3}`: Gemini API tier

### Components

- `--detector NAME`: Layout detector
- `--sorter NAME`: Reading order sorter
- `--recognizer NAME`: Text recognizer

### Output

- `--output DIR`: Output directory
- `--cache-dir DIR`: Cache directory
- `--temp-dir DIR`: Temporary files directory
- `--no-cache`: Disable caching

### Other

- `--dpi N`: PDF rendering DPI (default: 200)
- `-v, --verbose`: Verbose output
- `-vv`: Very verbose (debug)
- `--rate-limit-status`: Check Gemini rate limits

!!! note
    Run `python main.py --help` for complete option list.
