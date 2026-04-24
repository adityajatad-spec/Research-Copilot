"""Command-line interface for the AI Research Copilot."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

try:
    from .agent_loop import run_agent
    from .eval_harness import safe_run_benchmark
    from .fetcher import fetch_papers
    from .models import ExperimentPlan, HypothesisItem, Paper
    from .persistent_memory import load_run_history
    from .reporter import generate_report
    from .utils import display_papers, load_from_json, save_json_data, save_report, save_to_json
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_loop import run_agent
    from eval_harness import safe_run_benchmark
    from fetcher import fetch_papers
    from models import ExperimentPlan, HypothesisItem, Paper
    from persistent_memory import load_run_history
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
DEFAULT_HYPOTHESES_PAPERS_PATH = "output/papers_with_pdf.json"
DEFAULT_HYPOTHESES_INSIGHTS_PATH = "output/insights.json"
DEFAULT_HYPOTHESES_GAPS_PATH = "output/gaps.json"
DEFAULT_HYPOTHESES_OUTPUT_PATH = "output/hypotheses.json"
DEFAULT_EXPERIMENT_HYPOTHESES_PATH = "output/hypotheses.json"
DEFAULT_EXPERIMENT_OUTPUT_PATH = "output/experiment.py"
DEFAULT_GAPS_INPUT_PATH = "output/summaries.json"
DEFAULT_GAPS_OUTPUT_PATH = "output/gaps.json"
DEFAULT_PDF_INPUT_PATH = "output/results.json"
DEFAULT_PDF_OUTPUT_PATH = "output/papers_with_pdf.json"
DEFAULT_PDF_DIR = "output/pdfs"
DEFAULT_PDF_MAX_PAGES = 10
DEFAULT_RUN_EXPERIMENT_SCRIPT = "output/experiment.py"
DEFAULT_RUN_EXPERIMENT_DATASET = "demo-dataset"
DEFAULT_RUN_EXPERIMENT_OUTPUT_DIR = "output/experiment_run"
DEFAULT_RUN_EXPERIMENT_EPOCHS = 1
DEFAULT_RUN_EXPERIMENT_LEARNING_RATE = 1e-4
DEFAULT_RUN_EXPERIMENT_SEED = 42
DEFAULT_RUN_EXPERIMENT_TIMEOUT = 120
DEFAULT_RUN_EXPERIMENT_SUMMARY_PATH = "output/experiment_run_summary.json"
DEFAULT_AGENT_OUTPUT_PATH = "output/agent_run.json"
DEFAULT_AGENT_MAX_ITERATIONS = 6
DEFAULT_BENCHMARK_OUTPUT_DIR = "output/evals"
DEFAULT_REPAIR_REPORT_OUTPUT = "output/repair_report.json"
DEFAULT_REPAIR_REPORT_LIMIT = 10
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3.2"
EMPTY_TOPIC_MESSAGE = "Please provide a non-empty research topic."
INVALID_MAX_MESSAGE = "--max must be greater than 0."
INVALID_AGENT_ITERATIONS_MESSAGE = "--max-iterations must be greater than 0."
INVALID_EXPERIMENT_TIMEOUT_MESSAGE = "--timeout must be greater than 0."
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
HYPOTHESES_PRINT_ERROR = "Could not render hypotheses preview: {error}"
HYPOTHESES_INPUT_ERROR = "Could not load hypothesis file: {filepath}"
HYPOTHESES_EMPTY_ERROR = "No hypotheses found in the hypothesis report."
EXPERIMENT_PRINT_ERROR = "Could not render experiment scaffold preview: {error}"
RUN_EXPERIMENT_PRINT_ERROR = "Could not render run summary preview: {error}"
AGENT_RUN_ERROR = "Agent run failed: {error}"
AGENT_PRINT_ERROR = "Could not render agent run summary: {error}"
BENCHMARK_RUN_ERROR = "Benchmark run failed: {error}"
BENCHMARK_PRINT_ERROR = "Could not render benchmark summary: {error}"
REPAIR_REPORT_PRINT_ERROR = "Could not render repair report summary: {error}"


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

    hypotheses_parser = subparsers.add_parser("hypotheses", help="Generate evidence-grounded research hypotheses")
    hypotheses_parser.add_argument(
        "--papers",
        default=DEFAULT_HYPOTHESES_PAPERS_PATH,
        help="JSON file containing papers with summaries and optional full text",
    )
    hypotheses_parser.add_argument(
        "--insights",
        default=DEFAULT_HYPOTHESES_INSIGHTS_PATH,
        help="Optional insights JSON file used as context",
    )
    hypotheses_parser.add_argument(
        "--gaps",
        default=DEFAULT_HYPOTHESES_GAPS_PATH,
        help="Optional gaps JSON file used as context",
    )
    hypotheses_parser.add_argument(
        "--output",
        default=DEFAULT_HYPOTHESES_OUTPUT_PATH,
        help="Path to save the generated hypotheses JSON",
    )
    hypotheses_parser.add_argument(
        "--topic",
        default="",
        help="Research topic label for hypothesis generation",
    )
    hypotheses_parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help=PROVIDER_HELP_TEXT,
    )
    hypotheses_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=MODEL_HELP_TEXT,
    )
    hypotheses_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_hypotheses",
        help="Print generated hypotheses in the terminal",
    )

    experiment_parser = subparsers.add_parser("experiment", help="Generate a Python experiment scaffold from top hypothesis")
    experiment_parser.add_argument(
        "--hypotheses",
        default=DEFAULT_EXPERIMENT_HYPOTHESES_PATH,
        help="JSON file containing generated hypotheses",
    )
    experiment_parser.add_argument(
        "--output",
        default=DEFAULT_EXPERIMENT_OUTPUT_PATH,
        help="Path to save the generated Python experiment script",
    )
    experiment_parser.add_argument(
        "--topic",
        default="",
        help="Optional topic override for the script header",
    )
    experiment_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_experiment",
        help="Print the generated script in the terminal",
    )

    run_experiment_parser = subparsers.add_parser("run-experiment", help="Run a generated experiment script locally")
    run_experiment_parser.add_argument(
        "--script",
        default=DEFAULT_RUN_EXPERIMENT_SCRIPT,
        help="Path to the experiment Python script",
    )
    run_experiment_parser.add_argument(
        "--dataset",
        default=DEFAULT_RUN_EXPERIMENT_DATASET,
        help="Dataset name or path to pass to the experiment script",
    )
    run_experiment_parser.add_argument(
        "--output-dir",
        default=DEFAULT_RUN_EXPERIMENT_OUTPUT_DIR,
        help="Directory where the experiment script should write run outputs",
    )
    run_experiment_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_RUN_EXPERIMENT_EPOCHS,
        help="Epoch count passed to the experiment script",
    )
    run_experiment_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_RUN_EXPERIMENT_LEARNING_RATE,
        help="Learning rate passed to the experiment script",
    )
    run_experiment_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RUN_EXPERIMENT_SEED,
        help="Random seed passed to the experiment script",
    )
    run_experiment_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_RUN_EXPERIMENT_TIMEOUT,
        help="Timeout in seconds for experiment execution",
    )
    run_experiment_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_run_experiment",
        help="Print a compact execution summary in the terminal",
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

    agent_parser = subparsers.add_parser("agent", help="Run the autonomous research agent loop")
    agent_parser.add_argument(
        "--topic",
        required=True,
        help="Research topic for autonomous pipeline execution",
    )
    agent_parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help=PROVIDER_HELP_TEXT,
    )
    agent_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=MODEL_HELP_TEXT,
    )
    agent_parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_AGENT_MAX_ITERATIONS,
        help="Maximum autonomous planning iterations (hard-capped internally at 8)",
    )
    agent_parser.add_argument(
        "--output",
        default=DEFAULT_AGENT_OUTPUT_PATH,
        help="Path to save the final agent run JSON",
    )
    agent_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_agent",
        help="Print a readable summary of the agent run",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a lightweight benchmark over multiple topics")
    benchmark_parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help=PROVIDER_HELP_TEXT,
    )
    benchmark_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=MODEL_HELP_TEXT,
    )
    benchmark_parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_AGENT_MAX_ITERATIONS,
        help="Maximum iterations to allow per benchmark task",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of benchmark tasks to run",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default=DEFAULT_BENCHMARK_OUTPUT_DIR,
        help="Directory to save benchmark run and score artifacts",
    )
    benchmark_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_benchmark",
        help="Print aggregate metrics and per-task score rows",
    )

    repair_report_parser = subparsers.add_parser(
        "repair-report",
        help="Summarize common failures and repair strategies from recent agent runs",
    )
    repair_report_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_REPAIR_REPORT_LIMIT,
        help="Number of recent runs to include in the repair report",
    )
    repair_report_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_repair_report",
        help="Print a readable repair summary table in the terminal",
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


def _display_hypothesis_report(report: object, console: Console) -> None:
    """Display a hypothesis report in Rich tables."""
    overview = Table(show_header=False, box=None)
    overview.add_row("Topic", getattr(report, "topic", ""))
    overview.add_row("Paper Count", str(getattr(report, "paper_count", 0)))
    console.print(overview)

    source_gaps = getattr(report, "generated_from_gaps", [])
    gaps_table = Table(title="Generated From Gaps", show_header=True, header_style="bold yellow")
    gaps_table.add_column("#", width=4, justify="right")
    gaps_table.add_column("Gap", overflow="fold")
    for index, item in enumerate(source_gaps, start=1):
        gaps_table.add_row(str(index), item)
    console.print(gaps_table)

    hypotheses = getattr(report, "hypotheses", [])
    for index, item in enumerate(hypotheses, start=1):
        table = Table(title=f"Hypothesis {index}: {item.title}", show_header=False)
        table.add_row("Hypothesis", item.hypothesis)
        table.add_row("Novelty", item.novelty_rationale)
        table.add_row("Feasibility", item.feasibility_rationale)
        table.add_row("Objective", item.experiment_plan.objective)
        table.add_row("Datasets", ", ".join(item.experiment_plan.datasets) or "-")
        table.add_row("Baselines", ", ".join(item.experiment_plan.baselines) or "-")
        table.add_row("Metrics", ", ".join(item.experiment_plan.metrics) or "-")
        console.print(table)


def _hypothesis_item_from_dict(data: dict) -> HypothesisItem:
    """Deserialize one hypothesis item from JSON-compatible data."""
    experiment_plan_data = data.get("experiment_plan", {}) if isinstance(data, dict) else {}
    plan = ExperimentPlan(
        objective=str(experiment_plan_data.get("objective", "")),
        datasets=[str(item) for item in experiment_plan_data.get("datasets", []) if str(item).strip()],
        baselines=[str(item) for item in experiment_plan_data.get("baselines", []) if str(item).strip()],
        metrics=[str(item) for item in experiment_plan_data.get("metrics", []) if str(item).strip()],
        implementation_notes=[
            str(item) for item in experiment_plan_data.get("implementation_notes", []) if str(item).strip()
        ],
    )
    return HypothesisItem(
        title=str(data.get("title", "")).strip(),
        hypothesis=str(data.get("hypothesis", "")).strip(),
        novelty_rationale=str(data.get("novelty_rationale", "")).strip(),
        feasibility_rationale=str(data.get("feasibility_rationale", "")).strip(),
        experiment_plan=plan,
    )


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


def _run_hypotheses(args: argparse.Namespace, console: Console) -> None:
    """Run the hypotheses subcommand."""
    try:
        from .config import Config
        from .hypotheses import extract_hypotheses, load_optional_json, save_hypothesis_report
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from config import Config
        from hypotheses import extract_hypotheses, load_optional_json, save_hypothesis_report

    try:
        papers = load_from_json(args.papers)
    except FileNotFoundError as error:
        console.print(f"[yellow]{INPUT_LOAD_ERROR.format(error=error)}[/yellow]")
        return

    insights_data = load_optional_json(args.insights)
    gaps_data = load_optional_json(args.gaps)
    topic = _infer_report_topic(papers, args.topic)

    banner = Panel.fit(
        f"[bold]Papers:[/bold] {args.papers}\n"
        f"[bold]Insights:[/bold] {args.insights}\n"
        f"[bold]Gaps:[/bold] {args.gaps}\n"
        f"[bold]Output:[/bold] {args.output}\n"
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Provider:[/bold] {args.provider}\n"
        f"[bold]Model:[/bold] {args.model}",
        title=f"{APP_NAME} Hypotheses",
        border_style="bright_blue",
    )
    console.print(banner)

    config = Config(provider=args.provider, model=args.model)
    try:
        report = extract_hypotheses(
            papers=papers,
            topic=topic,
            config=config,
            insights_data=insights_data,
            gap_data=gaps_data,
        )
    except ValueError as error:
        console.print(f"[yellow]{SUMMARIZE_CONFIG_ERROR.format(error=error)}[/yellow]")
        return

    save_hypothesis_report(report, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")

    if args.print_hypotheses:
        try:
            _display_hypothesis_report(report, console)
        except Exception as error:
            console.print(f"[yellow]{HYPOTHESES_PRINT_ERROR.format(error=error)}[/yellow]")


def _run_experiment(args: argparse.Namespace, console: Console) -> None:
    """Run the experiment subcommand."""
    try:
        from .experiment_writer import generate_experiment_script, save_experiment_script
        from .hypotheses import load_optional_json
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from experiment_writer import generate_experiment_script, save_experiment_script
        from hypotheses import load_optional_json

    data = load_optional_json(args.hypotheses)
    if data is None:
        console.print(f"[yellow]{HYPOTHESES_INPUT_ERROR.format(filepath=args.hypotheses)}[/yellow]")
        return

    hypothesis_entries = data.get("hypotheses", []) if isinstance(data, dict) else []
    if not isinstance(hypothesis_entries, list) or not hypothesis_entries:
        console.print(f"[yellow]{HYPOTHESES_EMPTY_ERROR}[/yellow]")
        return

    top_hypothesis = _hypothesis_item_from_dict(hypothesis_entries[0])
    if not top_hypothesis.title or not top_hypothesis.hypothesis:
        console.print(f"[yellow]{HYPOTHESES_EMPTY_ERROR}[/yellow]")
        return

    topic = args.topic.strip() or str(data.get("topic", UNKNOWN_TOPIC)).strip() or UNKNOWN_TOPIC
    banner = Panel.fit(
        f"[bold]Hypotheses:[/bold] {args.hypotheses}\n"
        f"[bold]Output:[/bold] {args.output}\n"
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Top Hypothesis:[/bold] {top_hypothesis.title}",
        title=f"{APP_NAME} Experiment",
        border_style="green",
    )
    console.print(banner)

    script = generate_experiment_script(top_hypothesis, topic)
    save_experiment_script(script, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")

    if args.print_experiment:
        try:
            console.print(Markdown(f"```python\n{script}\n```"))
        except Exception as error:
            console.print(f"[yellow]{EXPERIMENT_PRINT_ERROR.format(error=error)}[/yellow]")


def _display_run_experiment_summary(run_result: dict, signals: dict, console: Console) -> None:
    """Display a compact summary for one experiment execution."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("Success", str(run_result.get("success", False)))
    table.add_row("Return Code", str(run_result.get("returncode")))
    table.add_row("Results Path", str(run_result.get("results_path") or "-"))
    table.add_row("Metric Keys", ", ".join(str(item) for item in signals.get("metric_keys", [])) or "-")
    stderr_tail = str(run_result.get("stderr", ""))
    stderr_tail = stderr_tail[-300:] if len(stderr_tail) > 300 else stderr_tail
    table.add_row("stderr tail", stderr_tail or "-")
    console.print(table)


def _run_run_experiment(args: argparse.Namespace, console: Console) -> None:
    """Run the run-experiment subcommand."""
    try:
        from .result_parser import extract_result_signals, load_experiment_results, summarize_experiment_result
        from .run_experiment import safe_run_experiment
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from result_parser import extract_result_signals, load_experiment_results, summarize_experiment_result
        from run_experiment import safe_run_experiment

    if args.timeout <= 0:
        console.print(f"[yellow]{INVALID_EXPERIMENT_TIMEOUT_MESSAGE}[/yellow]")
        return

    banner = Panel.fit(
        f"[bold]Script:[/bold] {args.script}\n"
        f"[bold]Dataset:[/bold] {args.dataset}\n"
        f"[bold]Output Dir:[/bold] {args.output_dir}\n"
        f"[bold]Epochs:[/bold] {args.epochs}\n"
        f"[bold]Learning Rate:[/bold] {args.learning_rate}\n"
        f"[bold]Seed:[/bold] {args.seed}\n"
        f"[bold]Timeout:[/bold] {args.timeout}s",
        title=f"{APP_NAME} Run Experiment",
        border_style="bright_magenta",
    )
    console.print(banner)

    run_result = safe_run_experiment(
        script_path=args.script,
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        timeout=args.timeout,
    )
    parsed_results = None
    results_path = run_result.get("results_path")
    if isinstance(results_path, str) and results_path:
        parsed_results = load_experiment_results(results_path)

    signals = extract_result_signals(parsed_results)
    summary_text = summarize_experiment_result(run_result, parsed_results)
    summary_payload = {
        "summary": summary_text,
        "success": bool(run_result.get("success")),
        "returncode": run_result.get("returncode"),
        "script_path": run_result.get("script_path"),
        "output_dir": run_result.get("output_dir"),
        "results_path": run_result.get("results_path"),
        "error": run_result.get("error"),
        "signals": signals,
    }
    save_json_data(summary_payload, DEFAULT_RUN_EXPERIMENT_SUMMARY_PATH)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=DEFAULT_RUN_EXPERIMENT_SUMMARY_PATH)}[/green]")

    if args.print_run_experiment:
        try:
            _display_run_experiment_summary(run_result, signals, console)
        except Exception as error:
            console.print(f"[yellow]{RUN_EXPERIMENT_PRINT_ERROR.format(error=error)}[/yellow]")


def _display_agent_run_summary(run_result: dict, console: Console) -> None:
    """Display a readable summary of one autonomous agent run."""
    overview = Table(show_header=False, box=None)
    overview.add_row("Topic", str(run_result.get("topic", "")))
    overview.add_row("Done", str(run_result.get("done", False)))
    overview.add_row("Iterations", str(run_result.get("iterations", 0)))
    overview.add_row("Partial Result", str(run_result.get("partial_result", False)))
    console.print(overview)

    history = run_result.get("history", [])
    history_table = Table(title="Action History", show_header=True, header_style="bold cyan")
    history_table.add_column("Step", justify="right", width=6)
    history_table.add_column("Action", width=12)
    history_table.add_column("Status", width=10)
    history_table.add_column("Reason", overflow="fold")

    if isinstance(history, list) and history:
        for item in history:
            if not isinstance(item, dict):
                continue
            history_table.add_row(
                str(item.get("step", "")),
                str(item.get("action", "")),
                str(item.get("status", "")),
                _truncate(str(item.get("reason", "")), limit=100),
            )
    else:
        history_table.add_row("-", "-", "-", "No actions recorded.")

    console.print(history_table)

    memory_summary = str(run_result.get("memory_summary", "No memory summary available."))
    console.print(Panel.fit(memory_summary, title="Memory Summary", border_style="green"))

    failure_summary = str(run_result.get("failure_summary", "No failure summary available."))
    repair_summary = str(run_result.get("repair_summary", "No repair summary available."))
    console.print(Panel.fit(failure_summary, title="Failure Summary", border_style="red"))
    console.print(Panel.fit(repair_summary, title="Repair Summary", border_style="yellow"))

    output_paths = run_result.get("final_output_paths", {})
    paths_table = Table(title="Output Paths", show_header=True, header_style="bold magenta")
    paths_table.add_column("Artifact")
    paths_table.add_column("Path", overflow="fold")

    if isinstance(output_paths, dict) and output_paths:
        for key, value in output_paths.items():
            paths_table.add_row(str(key), str(value))
    else:
        paths_table.add_row("-", "No output artifacts found.")

    console.print(paths_table)


def _display_benchmark_summary(result: dict, console: Console) -> None:
    """Display benchmark aggregate metrics and per-task scores."""
    aggregate = result.get("aggregate", {})
    aggregate_table = Table(title="Benchmark Aggregate", show_header=False, box=None)
    aggregate_table.add_row("Task Count", str(aggregate.get("task_count", 0)))
    aggregate_table.add_row("Completed Count", str(aggregate.get("completed_count", 0)))
    aggregate_table.add_row("Avg Total Score", str(aggregate.get("average_total_score", 0.0)))
    aggregate_table.add_row("Avg Artifact Score", str(aggregate.get("average_artifact_score", 0.0)))
    aggregate_table.add_row("Avg Iteration Score", str(aggregate.get("average_iteration_score", 0.0)))
    aggregate_table.add_row("Avg Experiment Score", str(aggregate.get("average_experiment_score", 0.0)))
    aggregate_table.add_row("Partial Result Count", str(aggregate.get("partial_result_count", 0)))
    aggregate_table.add_row("Best Task", str(aggregate.get("best_task", "")))
    aggregate_table.add_row("Worst Task", str(aggregate.get("worst_task", "")))
    failure_prone_tasks = aggregate.get("failure_prone_tasks", [])
    if isinstance(failure_prone_tasks, list):
        aggregate_table.add_row("Failure-Prone Tasks", ", ".join(str(item) for item in failure_prone_tasks) or "-")
    console.print(aggregate_table)

    task_table = Table(title="Benchmark Tasks", show_header=True, header_style="bold cyan")
    task_table.add_column("Topic", overflow="fold")
    task_table.add_column("Completed", width=10)
    task_table.add_column("Partial", width=8)
    task_table.add_column("Total", width=8, justify="right")
    task_table.add_column("Artifacts", width=10, justify="right")
    task_table.add_column("Iterations", width=10, justify="right")
    task_table.add_column("Repair Summary", overflow="fold")

    for row in result.get("scores", []):
        if not isinstance(row, dict):
            continue
        task_table.add_row(
            _truncate(str(row.get("topic", "")), limit=45),
            str(row.get("completed", False)),
            str(row.get("partial_result", False)),
            str(row.get("total_score", 0.0)),
            str(row.get("artifact_score", 0.0)),
            str(row.get("iterations", 0)),
            _truncate(str(row.get("repair_summary", "")), limit=70),
        )

    console.print(task_table)


def _run_agent(args: argparse.Namespace, console: Console) -> None:
    """Run the autonomous agent subcommand."""
    topic = args.topic.strip()
    if not topic:
        console.print(f"[yellow]{EMPTY_TOPIC_MESSAGE}[/yellow]")
        return

    if args.max_iterations <= 0:
        console.print(f"[yellow]{INVALID_AGENT_ITERATIONS_MESSAGE}[/yellow]")
        return

    banner = Panel.fit(
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Provider:[/bold] {args.provider}\n"
        f"[bold]Model:[/bold] {args.model}\n"
        f"[bold]Max Iterations:[/bold] {args.max_iterations}",
        title=f"{APP_NAME} Agent",
        border_style="bright_green",
    )
    console.print(banner)

    try:
        run_result = run_agent(
            topic=topic,
            provider=args.provider,
            model=args.model,
            max_iterations=args.max_iterations,
            log_steps=True,
            logger=lambda message: console.print(message, style="cyan", markup=False),
        )
    except Exception as error:
        console.print(f"[yellow]{AGENT_RUN_ERROR.format(error=error)}[/yellow]")
        return

    save_json_data(run_result, args.output)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=args.output)}[/green]")

    if args.print_agent:
        try:
            _display_agent_run_summary(run_result, console)
        except Exception as error:
            console.print(f"[yellow]{AGENT_PRINT_ERROR.format(error=error)}[/yellow]")


def _run_benchmark(args: argparse.Namespace, console: Console) -> None:
    """Run the benchmark subcommand."""
    if args.max_iterations <= 0:
        console.print(f"[yellow]{INVALID_AGENT_ITERATIONS_MESSAGE}[/yellow]")
        return

    limit_value = args.limit if args.limit and args.limit > 0 else None
    banner = Panel.fit(
        f"[bold]Provider:[/bold] {args.provider}\n"
        f"[bold]Model:[/bold] {args.model}\n"
        f"[bold]Max Iterations:[/bold] {args.max_iterations}\n"
        f"[bold]Limit:[/bold] {limit_value if limit_value is not None else 'all'}\n"
        f"[bold]Output Dir:[/bold] {args.output_dir}",
        title=f"{APP_NAME} Benchmark",
        border_style="bright_cyan",
    )
    console.print(banner)

    result = safe_run_benchmark(
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        limit=limit_value,
        output_dir=args.output_dir,
    )

    if not bool(result.get("success")):
        console.print(f"[yellow]{BENCHMARK_RUN_ERROR.format(error=result.get('error', 'unknown error'))}[/yellow]")
        return

    output_paths = result.get("output_paths", {})
    results_path = output_paths.get("results_json")
    scores_path = output_paths.get("scores_csv")
    if isinstance(results_path, str):
        console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=results_path)}[/green]")
    if isinstance(scores_path, str):
        console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=scores_path)}[/green]")

    if args.print_benchmark:
        try:
            _display_benchmark_summary(result, console)
        except Exception as error:
            console.print(f"[yellow]{BENCHMARK_PRINT_ERROR.format(error=error)}[/yellow]")


def _top_counter_rows(counter: Counter[str], label_key: str, limit: int = 5) -> list[dict]:
    """Convert a counter into top-N dictionaries."""
    rows: list[dict] = []
    for key, count in counter.most_common(limit):
        rows.append({label_key: key, "count": count})
    return rows


def _build_repair_report(limit: int) -> dict:
    """Build a repair summary report from recent persistent run history."""
    bounded_limit = limit if limit > 0 else DEFAULT_REPAIR_REPORT_LIMIT
    runs = load_run_history(limit=bounded_limit)

    failure_categories: Counter[str] = Counter()
    repair_strategies: Counter[str] = Counter()
    failing_actions: Counter[str] = Counter()
    partial_result_topics: list[str] = []
    recent_runs: list[dict] = []

    for run in runs:
        if not isinstance(run, dict):
            continue

        topic = str(run.get("topic", "")).strip()
        if bool(run.get("partial_result")) and topic:
            partial_result_topics.append(topic)

        history = run.get("history", [])
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                if str(item.get("status", "")) == "failed":
                    action = str(item.get("action", "")).strip()
                    if action:
                        failing_actions[action] += 1

        recent_failures = run.get("recent_failures", [])
        if isinstance(recent_failures, list):
            for failure in recent_failures:
                if not isinstance(failure, dict):
                    continue
                category = str(failure.get("category", "")).strip()
                if category:
                    failure_categories[category] += 1

        repair_history = run.get("repair_history", [])
        if isinstance(repair_history, list):
            for repair in repair_history:
                if not isinstance(repair, dict):
                    continue
                strategy = repair.get("strategy", {})
                if isinstance(strategy, dict):
                    strategy_name = str(strategy.get("strategy", "")).strip()
                    if strategy_name:
                        repair_strategies[strategy_name] += 1

        recent_runs.append(
            {
                "topic": topic,
                "done": bool(run.get("done", False)),
                "iterations": int(run.get("iterations", 0) or 0),
                "partial_result": bool(run.get("partial_result", False)),
                "failure_summary": str(run.get("failure_summary", "")),
                "repair_summary": str(run.get("repair_summary", "")),
            }
        )

    return {
        "generated_at": date.today().isoformat(),
        "run_count": len([row for row in runs if isinstance(row, dict)]),
        "common_failure_categories": _top_counter_rows(failure_categories, "category"),
        "common_repair_strategies": _top_counter_rows(repair_strategies, "strategy"),
        "partial_result_topics": partial_result_topics,
        "most_frequent_failing_actions": _top_counter_rows(failing_actions, "action"),
        "recent_runs": recent_runs[-bounded_limit:],
    }


def _display_repair_report(report: dict, console: Console) -> None:
    """Render a repair report summary in rich tables."""
    overview = Table(show_header=False, box=None)
    overview.add_row("Generated At", str(report.get("generated_at", "")))
    overview.add_row("Run Count", str(report.get("run_count", 0)))
    overview.add_row("Partial Result Topics", str(len(report.get("partial_result_topics", []))))
    console.print(overview)

    category_table = Table(title="Common Failure Categories", show_header=True, header_style="bold red")
    category_table.add_column("Category")
    category_table.add_column("Count", justify="right")
    categories = report.get("common_failure_categories", [])
    if isinstance(categories, list) and categories:
        for row in categories:
            if not isinstance(row, dict):
                continue
            category_table.add_row(str(row.get("category", "")), str(row.get("count", 0)))
    else:
        category_table.add_row("-", "0")
    console.print(category_table)

    strategy_table = Table(title="Common Repair Strategies", show_header=True, header_style="bold yellow")
    strategy_table.add_column("Strategy")
    strategy_table.add_column("Count", justify="right")
    strategies = report.get("common_repair_strategies", [])
    if isinstance(strategies, list) and strategies:
        for row in strategies:
            if not isinstance(row, dict):
                continue
            strategy_table.add_row(str(row.get("strategy", "")), str(row.get("count", 0)))
    else:
        strategy_table.add_row("-", "0")
    console.print(strategy_table)

    action_table = Table(title="Most Frequent Failing Actions", show_header=True, header_style="bold magenta")
    action_table.add_column("Action")
    action_table.add_column("Count", justify="right")
    actions = report.get("most_frequent_failing_actions", [])
    if isinstance(actions, list) and actions:
        for row in actions:
            if not isinstance(row, dict):
                continue
            action_table.add_row(str(row.get("action", "")), str(row.get("count", 0)))
    else:
        action_table.add_row("-", "0")
    console.print(action_table)

    partial_topics = report.get("partial_result_topics", [])
    if isinstance(partial_topics, list) and partial_topics:
        lines = "\n".join(f"- {topic}" for topic in partial_topics)
    else:
        lines = "No partial-result topics in recent runs."
    console.print(Panel.fit(lines, title="Partial Results", border_style="cyan"))


def _run_repair_report(args: argparse.Namespace, console: Console) -> None:
    """Run the repair-report subcommand."""
    limit_value = args.limit if args.limit and args.limit > 0 else DEFAULT_REPAIR_REPORT_LIMIT
    banner = Panel.fit(
        f"[bold]Limit:[/bold] {limit_value}\n"
        f"[bold]Output:[/bold] {DEFAULT_REPAIR_REPORT_OUTPUT}",
        title=f"{APP_NAME} Repair Report",
        border_style="bright_yellow",
    )
    console.print(banner)

    report = _build_repair_report(limit_value)
    save_json_data(report, DEFAULT_REPAIR_REPORT_OUTPUT)
    console.print(f"[green]{SAVE_SUCCESS_MESSAGE.format(filepath=DEFAULT_REPAIR_REPORT_OUTPUT)}[/green]")

    if args.print_repair_report:
        try:
            _display_repair_report(report, console)
        except Exception as error:
            console.print(f"[yellow]{REPAIR_REPORT_PRINT_ERROR.format(error=error)}[/yellow]")


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

    if args.command == "hypotheses":
        _run_hypotheses(args, console)
        return

    if args.command == "experiment":
        _run_experiment(args, console)
        return

    if args.command == "run-experiment":
        _run_run_experiment(args, console)
        return

    if args.command == "agent":
        _run_agent(args, console)
        return

    if args.command == "benchmark":
        _run_benchmark(args, console)
        return

    if args.command == "repair-report":
        _run_repair_report(args, console)
        return


if __name__ == "__main__":
    main()
