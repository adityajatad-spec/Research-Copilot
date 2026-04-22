"""Command-line interface for the AI Research Copilot."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

try:
    from .fetcher import fetch_papers
    from .models import Paper
    from .reporter import generate_report
    from .utils import display_papers, load_from_json, save_json_data, save_report, save_to_json
except ImportError:  # pragma: no cover - fallback for direct script execution
    from fetcher import fetch_papers
    from models import Paper
    from reporter import generate_report
    from utils import display_papers, load_from_json, save_json_data, save_report, save_to_json


APP_NAME = "AI Research Copilot"
DEFAULT_MAX_RESULTS = 10
DEFAULT_FETCH_OUTPUT_PATH = "output/results.json"
DEFAULT_FETCH_SOURCE = "arxiv"
DEFAULT_SUMMARIZE_INPUT_PATH = "output/results.json"
DEFAULT_SUMMARIZE_OUTPUT_PATH = "output/summaries.json"
DEFAULT_REPORT_INPUT_PATH = "output/summaries.json"
DEFAULT_REPORT_OUTPUT_PATH = "output/report.md"
DEFAULT_INSIGHTS_INPUT_PATH = "output/summaries.json"
DEFAULT_INSIGHTS_OUTPUT_PATH = "output/insights.json"
DEFAULT_GAPS_INPUT_PATH = "output/summaries.json"
DEFAULT_GAPS_OUTPUT_PATH = "output/gaps.json"
DEFAULT_PDF_INPUT_PATH = "output/results.json"
DEFAULT_PDF_OUTPUT_PATH = "output/papers_with_pdf.json"
DEFAULT_PDF_DIR = "output/pdfs"
DEFAULT_PDF_MAX_PAGES = 10
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3.2"
EMPTY_TOPIC_MESSAGE = "Please provide a non-empty research topic."
INVALID_MAX_MESSAGE = "--max must be greater than 0."
NO_RESULTS_MESSAGE = "No papers found for that topic."
NO_SAVE_MESSAGE = "Skipping JSON export because --no-save was provided."
SAVE_SUCCESS_MESSAGE = "Saved results to {filepath}"
INPUT_LOAD_ERROR = "Could not load input file: {error}"
SUMMARIZE_CONFIG_ERROR = "Configuration error: {error}"
SUMMARY_FAILED = "[extraction failed]"
TITLE_LIMIT = 60
UNKNOWN_TOPIC = "Unspecified Topic"
PROVIDER_HELP_TEXT = "LLM provider to use for summarization (ollama = free local model, openai = paid cloud API)"
MODEL_HELP_TEXT = "Model name to use for summarization (default: llama3.2 for Ollama)"
PDF_MAX_PAGES_ERROR = "--max-pages must be greater than 0."
FETCH_SOURCE_HELP_TEXT = "Paper source to query: arxiv, semanticscholar, or hybrid."
FETCH_SOURCE_IMPORT_ERROR = "Semantic Scholar support is unavailable: {error}"
INSIGHTS_PRINT_ERROR = "Could not render insights preview: {error}"
GAPS_PRINT_ERROR = "Could not render gap analysis preview: {error}"


def _truncate(text: str, limit: int = TITLE_LIMIT) -> str:
    """Trim long text for compact terminal output."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Fetch, enrich, summarize, extract insights from, and report on research papers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch relevant papers for a topic")
    fetch_parser.add_argument("topic", help="The research topic to search for")
    fetch_parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Number of papers to fetch",
    )
    fetch_parser.add_argument(
        "--output",
        default=DEFAULT_FETCH_OUTPUT_PATH,
        help="Path to save the JSON results",
    )
    fetch_parser.add_argument(
        "--source",
        default=DEFAULT_FETCH_SOURCE,
        choices=["arxiv", "semanticscholar", "hybrid"],
        help=FETCH_SOURCE_HELP_TEXT,
    )
    fetch_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display results without saving them to disk",
    )

    summarize_parser = subparsers.add_parser("summarize", help="Generate structured summaries for fetched papers")
    summarize_parser.add_argument(
        "--input",
        default=DEFAULT_SUMMARIZE_INPUT_PATH,
        help="JSON file containing fetched papers",
    )
    summarize_parser.add_argument(
        "--output",
        default=DEFAULT_SUMMARIZE_OUTPUT_PATH,
        help="Path to save enriched papers with summaries",
    )
    summarize_parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help=PROVIDER_HELP_TEXT,
    )
    summarize_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=MODEL_HELP_TEXT,
    )
    summarize_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-paper progress details while summarizing",
    )

    report_parser = subparsers.add_parser("report", help="Generate a markdown report from summarized papers")
    report_parser.add_argument(
        "--input",
        default=DEFAULT_REPORT_INPUT_PATH,
        help="JSON file containing summarized papers",
    )
    report_parser.add_argument(
        "--output",
        default=DEFAULT_REPORT_OUTPUT_PATH,
        help="Path to save the markdown report",
    )
    report_parser.add_argument(
        "--topic",
        default="",
        help="Research topic label for the report header",
    )
    report_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_report",
        help="Render the generated markdown report in the terminal",
    )

    insights_parser = subparsers.add_parser("insights", help="Extract cross-paper insights from enriched papers")
    insights_parser.add_argument(
        "--input",
        default=DEFAULT_INSIGHTS_INPUT_PATH,
        help="JSON file containing summarized and optionally PDF-enriched papers",
    )
    insights_parser.add_argument(
        "--output",
        default=DEFAULT_INSIGHTS_OUTPUT_PATH,
        help="Path to save the extracted insights as JSON",
    )
    insights_parser.add_argument(
        "--topic",
        default="",
        help="Research topic label for cross-paper insights",
    )
    insights_parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help=PROVIDER_HELP_TEXT,
    )
    insights_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=MODEL_HELP_TEXT,
    )
    insights_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_insights",
        help="Print the extracted insights in the terminal",
    )

    gaps_parser = subparsers.add_parser("gaps", help="Detect contradictions and research gaps across papers")
    gaps_parser.add_argument(
        "--input",
        default=DEFAULT_GAPS_INPUT_PATH,
        help="JSON file containing summarized papers and optional PDF text",
    )
    gaps_parser.add_argument(
        "--output",
        default=DEFAULT_GAPS_OUTPUT_PATH,
        help="Path to save the extracted gap report as JSON",
    )
    gaps_parser.add_argument(
        "--topic",
        default="",
        help="Research topic label for gap detection",
    )
    gaps_parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help=PROVIDER_HELP_TEXT,
    )
    gaps_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=MODEL_HELP_TEXT,
    )
    gaps_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_gaps",
        help="Print the extracted gap analysis in the terminal",
    )

    pdf_parser = subparsers.add_parser("pdf", help="Download paper PDFs and extract full text")
    pdf_parser.add_argument(
        "--input",
        default=DEFAULT_PDF_INPUT_PATH,
        help="JSON file containing fetched papers",
    )
    pdf_parser.add_argument(
        "--output",
        default=DEFAULT_PDF_OUTPUT_PATH,
        help="Path to save papers enriched with local PDF data",
    )
    pdf_parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
        help="Directory to store downloaded PDF files",
    )
    pdf_parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_PDF_MAX_PAGES,
        help="Maximum number of pages to parse from each PDF",
    )
    pdf_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-paper progress details while parsing PDFs",
    )

    return parser


def _run_fetch(args: argparse.Namespace, console: Console) -> None:
    """Run the fetch subcommand."""
    topic = args.topic.strip()
    max_results = args.max
    source = args.source
    if not topic:
        console.print(f"[yellow]{EMPTY_TOPIC_MESSAGE}[/yellow]")
        return

    if max_results <= 0:
        console.print(f"[yellow]{INVALID_MAX_MESSAGE}[/yellow]")
        return

    banner = Panel.fit(
        f"[bold]Topic:[/bold] {topic}\n[bold]Max Results:[/bold] {max_results}\n[bold]Source:[/bold] {source}",
        title=APP_NAME,
        border_style="blue",
    )
    console.print(banner)

    if source == "arxiv":
        papers = fetch_papers(topic=topic, max_results=max_results)
    else:
        try:
            from .scholar_fetcher import fetch_hybrid_papers, fetch_semantic_scholar_papers
        except ImportError as error:  # pragma: no cover - fallback for direct script execution
            try:
                from scholar_fetcher import fetch_hybrid_papers, fetch_semantic_scholar_papers
            except ImportError as import_error:
                console.print(f"[yellow]{FETCH_SOURCE_IMPORT_ERROR.format(error=import_error)}[/yellow]")
                return

        if source == "semanticscholar":
            papers = fetch_semantic_scholar_papers(topic=topic, max_results=max_results)
        else:
            papers = fetch_hybrid_papers(topic=topic, max_results=max_results)

    if not papers:
        console.print(f"[yellow]{NO_RESULTS_MESSAGE}[/yellow]")
        return

    display_papers(papers, query=topic)

    if args.no_save:
        console.print(f"[cyan]{NO_SAVE_MESSAGE}[/cyan]")
        return

    save_to_json(papers, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")


def _display_summary_table(papers: list[Paper], console: Console) -> None:
    """Display a summary table for summarized papers."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Title", overflow="fold")
    table.add_column("Core Contribution", overflow="fold")

    for paper in papers:
        core_contribution = SUMMARY_FAILED
        if paper.summary is not None:
            core_contribution = paper.summary.core_contribution

        table.add_row(_truncate(paper.title), core_contribution)

    console.print(table)


def _display_pdf_stats(stats: dict[str, int], console: Console) -> None:
    """Display PDF enrichment statistics in a Rich table."""
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for key, value in stats.items():
        label = key.replace("_", " ").title()
        table.add_row(label, str(value))

    console.print(table)


def _display_insight_report(report: object, console: Console) -> None:
    """Display an insight report in Rich tables."""
    overview = Table(show_header=False, box=None)
    overview.add_row("Topic", getattr(report, "topic", ""))
    overview.add_row("Paper Count", str(getattr(report, "paper_count", 0)))
    console.print(overview)

    sections = [
        ("Major Themes", getattr(report, "major_themes", [])),
        ("Common Methodologies", getattr(report, "common_methodologies", [])),
        ("Emerging Subtopics", getattr(report, "emerging_subtopics", [])),
        ("Notable Observations", getattr(report, "notable_observations", [])),
    ]

    for title, items in sections:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("#", width=4, justify="right")
        table.add_column("Insight", overflow="fold")

        for index, item in enumerate(items, start=1):
            table.add_row(str(index), item)

        console.print(table)


def _display_gap_report(report: object, console: Console) -> None:
    """Display a gap report in Rich tables."""
    overview = Table(show_header=False, box=None)
    overview.add_row("Topic", getattr(report, "topic", ""))
    overview.add_row("Paper Count", str(getattr(report, "paper_count", 0)))
    console.print(overview)

    contradiction_table = Table(title="Contradictions", show_header=True, header_style="bold red")
    contradiction_table.add_column("#", width=4, justify="right")
    contradiction_table.add_column("Paper A", overflow="fold")
    contradiction_table.add_column("Paper B", overflow="fold")
    contradiction_table.add_column("Contradiction", overflow="fold")

    contradictions = getattr(report, "contradictions", [])
    if contradictions:
        for index, item in enumerate(contradictions, start=1):
            contradiction_table.add_row(str(index), item.paper_a, item.paper_b, item.contradiction)
    else:
        contradiction_table.add_row("-", "-", "-", "No strong contradictions identified.")

    console.print(contradiction_table)

    sections = [
        ("Recurring Limitations", getattr(report, "recurring_limitations", [])),
        ("Underexplored Directions", getattr(report, "underexplored_directions", [])),
        ("Explicit Research Gaps", getattr(report, "explicit_research_gaps", [])),
    ]

    for title, items in sections:
        table = Table(title=title, show_header=True, header_style="bold yellow")
        table.add_column("#", width=4, justify="right")
        table.add_column("Finding", overflow="fold")

        for index, item in enumerate(items, start=1):
            table.add_row(str(index), item)

        console.print(table)


def _run_summarize(args: argparse.Namespace, console: Console) -> None:
    """Run the summarize subcommand."""
    try:
        from .config import Config
        from .summarizer import summarize_all
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from config import Config
        from summarizer import summarize_all

    banner = Panel.fit(
        f"[bold]Provider:[/bold] {args.provider}\n"
        f"[bold]Model:[/bold] {args.model}\n"
        f"[bold]Input:[/bold] {args.input}",
        title=f"{APP_NAME} Summary",
        border_style="green",
    )
    console.print(banner)

    try:
        papers = load_from_json(args.input)
    except FileNotFoundError as error:
        console.print(f"[yellow]{INPUT_LOAD_ERROR.format(error=error)}[/yellow]")
        return

    config = Config(provider=args.provider, model=args.model)

    try:
        summarized_papers = summarize_all(papers, config, verbose=args.verbose)
    except ValueError as error:
        console.print(f"[yellow]{SUMMARIZE_CONFIG_ERROR.format(error=error)}[/yellow]")
        return

    save_to_json(summarized_papers, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")
    _display_summary_table(summarized_papers, console)


def _infer_report_topic(papers: list[Paper], explicit_topic: str) -> str:
    """Infer a report topic when one is not provided."""
    provided_topic = explicit_topic.strip()
    if provided_topic:
        return provided_topic

    if papers and papers[0].categories:
        return " / ".join(papers[0].categories)

    return UNKNOWN_TOPIC


def _run_report(args: argparse.Namespace, console: Console) -> None:
    """Run the report subcommand."""
    banner = Panel.fit(
        f"[bold]Input:[/bold] {args.input}\n"
        f"[bold]Output:[/bold] {args.output}",
        title=f"{APP_NAME} Report",
        border_style="cyan",
    )
    console.print(banner)

    try:
        papers = load_from_json(args.input)
    except FileNotFoundError as error:
        console.print(f"[yellow]{INPUT_LOAD_ERROR.format(error=error)}[/yellow]")
        return

    topic = _infer_report_topic(papers, args.topic)
    report_str = generate_report(papers, topic)
    save_report(report_str, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")

    if args.print_report:
        console.print(Markdown(report_str))


def _run_pdf(args: argparse.Namespace, console: Console) -> None:
    """Run the pdf subcommand."""
    try:
        from .pdf_parser import build_pdf_stats, enrich_papers_with_pdf_text
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from pdf_parser import build_pdf_stats, enrich_papers_with_pdf_text

    if args.max_pages <= 0:
        console.print(f"[yellow]{PDF_MAX_PAGES_ERROR}[/yellow]")
        return

    banner = Panel.fit(
        f"[bold]Input:[/bold] {args.input}\n"
        f"[bold]Output:[/bold] {args.output}\n"
        f"[bold]PDF Dir:[/bold] {args.pdf_dir}\n"
        f"[bold]Max Pages:[/bold] {args.max_pages}",
        title=f"{APP_NAME} PDF",
        border_style="magenta",
    )
    console.print(banner)

    try:
        papers = load_from_json(args.input)
    except FileNotFoundError as error:
        console.print(f"[yellow]{INPUT_LOAD_ERROR.format(error=error)}[/yellow]")
        return

    enriched_papers = enrich_papers_with_pdf_text(
        papers,
        pdf_dir=args.pdf_dir,
        max_pages=args.max_pages,
        verbose=args.verbose,
    )
    save_to_json(enriched_papers, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")
    _display_pdf_stats(build_pdf_stats(enriched_papers), console)


def _run_insights(args: argparse.Namespace, console: Console) -> None:
    """Run the insights subcommand."""
    try:
        from .config import Config
        from .insights import extract_insights
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from config import Config
        from insights import extract_insights

    try:
        papers = load_from_json(args.input)
    except FileNotFoundError as error:
        console.print(f"[yellow]{INPUT_LOAD_ERROR.format(error=error)}[/yellow]")
        return

    topic = _infer_report_topic(papers, args.topic)
    banner = Panel.fit(
        f"[bold]Input:[/bold] {args.input}\n"
        f"[bold]Output:[/bold] {args.output}\n"
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Provider:[/bold] {args.provider}\n"
        f"[bold]Model:[/bold] {args.model}",
        title=f"{APP_NAME} Insights",
        border_style="yellow",
    )
    console.print(banner)

    config = Config(provider=args.provider, model=args.model)

    try:
        insight_report = extract_insights(papers, topic, config)
    except ValueError as error:
        console.print(f"[yellow]{SUMMARIZE_CONFIG_ERROR.format(error=error)}[/yellow]")
        return

    save_json_data(insight_report.to_dict(), args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")

    if args.print_insights:
        try:
            _display_insight_report(insight_report, console)
        except Exception as error:
            console.print(f"[yellow]{INSIGHTS_PRINT_ERROR.format(error=error)}[/yellow]")


def _run_gaps(args: argparse.Namespace, console: Console) -> None:
    """Run the gaps subcommand."""
    try:
        from .config import Config
        from .gaps import extract_gaps_and_contradictions, save_gap_report
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from config import Config
        from gaps import extract_gaps_and_contradictions, save_gap_report

    try:
        papers = load_from_json(args.input)
    except FileNotFoundError as error:
        console.print(f"[yellow]{INPUT_LOAD_ERROR.format(error=error)}[/yellow]")
        return

    topic = _infer_report_topic(papers, args.topic)
    banner = Panel.fit(
        f"[bold]Input:[/bold] {args.input}\n"
        f"[bold]Output:[/bold] {args.output}\n"
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Provider:[/bold] {args.provider}\n"
        f"[bold]Model:[/bold] {args.model}",
        title=f"{APP_NAME} Gaps",
        border_style="red",
    )
    console.print(banner)

    config = Config(provider=args.provider, model=args.model)

    try:
        gap_report = extract_gaps_and_contradictions(papers, topic, config)
    except ValueError as error:
        console.print(f"[yellow]{SUMMARIZE_CONFIG_ERROR.format(error=error)}[/yellow]")
        return

    save_gap_report(gap_report, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")

    if args.print_gaps:
        try:
            _display_gap_report(gap_report, console)
        except Exception as error:
            console.print(f"[yellow]{GAPS_PRINT_ERROR.format(error=error)}[/yellow]")


def main() -> None:
    """Run the AI Research Copilot CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    console = Console()

    if args.command == "fetch":
        _run_fetch(args, console)
        return

    if args.command == "summarize":
        _run_summarize(args, console)
        return

    if args.command == "report":
        _run_report(args, console)
        return

    if args.command == "pdf":
        _run_pdf(args, console)
        return

    if args.command == "insights":
        _run_insights(args, console)
        return

    if args.command == "gaps":
        _run_gaps(args, console)
        return


if __name__ == "__main__":
    main()
