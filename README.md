# AI Research Copilot

AI Research Copilot is a Python CLI tool that helps users go from a research topic to a structured understanding of relevant papers. It fetches papers, extracts structured summaries, and generates a Markdown research report. The project is designed as the MVP foundation for a larger research copilot that will later detect trends, contradictions, and research directions.

## Features

- Search arXiv by research topic
- Fetch top relevant papers with metadata
- Download paper PDFs and extract full text locally
- Generate structured summaries using Ollama by default, with OpenAI as an optional cloud provider
- Export enriched results to JSON
- Generate a Markdown research report
- Modular CLI workflow: `fetch -> pdf -> summarize -> report`

## Project Structure

```text
.
├── README.md                    # GitHub-facing project documentation
├── .env.example                 # Example environment variables for local setup
├── .gitignore                   # Repository hygiene for Python, outputs, and secrets
├── output/                      # Generated artifacts created from CLI runs
└── research_copilot/
    ├── __init__.py              # Package marker for the project modules
    ├── cli.py                   # Command-line entry point for fetch, summarize, and report
    ├── fetcher.py               # arXiv search and paper-fetching logic
    ├── pdf_parser.py            # PDF download and text extraction helpers
    ├── summarizer.py            # Structured per-paper summary generation
    ├── reporter.py              # Markdown research report generation
    ├── models.py                # Dataclasses for papers and summaries
    ├── utils.py                 # JSON I/O, report saving, and terminal display helpers
    ├── config.py                # Provider and client configuration for LLM backends
    ├── requirements.txt         # Python dependencies for the CLI
    └── output/                  # Default output directory inside the app workspace
```

## Installation

```bash
git clone <your-repo-url>
cd <repo-folder>/research_copilot
pip install -r requirements.txt
```

## Free Local Mode (Recommended)

1. Install [Ollama](https://ollama.com/download).
2. Pull a local model:

```bash
ollama pull llama3.2
```

3. Make sure Ollama is running locally.
4. Use the CLI with the local-first defaults:

```bash
python cli.py summarize \
  --input output/results.json \
  --output output/summaries.json \
  --provider ollama \
  --model llama3.2 \
  --verbose
```

The project now defaults to `ollama` with the `llama3.2` model, so you can also omit those flags when you want the recommended local setup.

## Cloud Mode (Optional)

If you want to use OpenAI instead of a local model, create a local environment file from the template and export your API key:

```bash
cp ../.env.example ../.env
export OPENAI_API_KEY=your_api_key_here
```

Then run summarization with the OpenAI provider explicitly:

```bash
python cli.py summarize \
  --input output/results.json \
  --output output/summaries.json \
  --provider openai \
  --model gpt-4o-mini \
  --verbose
```

## Usage

Fetch relevant papers for a topic:

```bash
python cli.py fetch "transformer attention" --max 10 --output output/results.json
```

Download PDFs and extract text from the first pages of each paper:

```bash
python cli.py pdf \
  --input output/results.json \
  --output output/papers_with_pdf.json \
  --pdf-dir output/pdfs \
  --max-pages 8 \
  --verbose
```

Generate structured summaries from the fetched papers:

```bash
python cli.py summarize \
  --input output/papers_with_pdf.json \
  --output output/summaries.json \
  --provider ollama \
  --model llama3.2 \
  --verbose
```

Generate a Markdown report from summarized papers:

```bash
python cli.py report \
  --input output/summaries.json \
  --output output/report.md \
  --topic "transformer attention" \
  --print
```

## Example Workflow

1. Start with a broad research topic such as `"retrieval augmented generation"`.
2. Fetch the most relevant arXiv papers into a local JSON file.
3. Download the corresponding PDFs and extract local text for deeper downstream analysis.
4. Enrich those papers with structured summaries using Ollama by default, or OpenAI when needed.
5. Generate a clean Markdown report that is easy to review, share, or extend.

## Why This Project

This repository is intentionally scoped as a clean MVP for a larger research workflow. The current version focuses on paper collection, structured extraction, and readable reporting so future phases can build on a reliable, modular foundation.

## Roadmap

- Add cross-paper trend detection
- Surface agreement and contradiction patterns between papers
- Suggest open problems and research directions
- Improve ranking and filtering for deeper literature reviews
