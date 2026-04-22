# AI Research Copilot Package

This directory contains the runnable Python modules for AI Research Copilot.
For the full project overview, setup instructions, and workflow examples, see the repository-level [README](../README.md).

## Quick Start

```bash
pip install -r requirements.txt
python cli.py fetch "transformer attention"
python cli.py pdf --input output/results.json --output output/papers_with_pdf.json --pdf-dir output/pdfs --max-pages 8
python cli.py summarize --input output/papers_with_pdf.json --output output/summaries.json --provider ollama --model llama3.2
python cli.py report --input output/summaries.json --output output/report.md --topic "transformer attention"
```
