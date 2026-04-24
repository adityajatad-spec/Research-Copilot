"""Static benchmark tasks for lightweight autonomous-agent evaluation."""

from __future__ import annotations


BENCHMARK_TASKS: list[dict] = [
    {
        "id": "task_001",
        "topic": "vision transformers",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_002",
        "topic": "graph neural networks",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_003",
        "topic": "retrieval augmented generation",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_004",
        "topic": "diffusion models",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_005",
        "topic": "small language models",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_006",
        "topic": "multimodal learning",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_007",
        "topic": "medical image segmentation",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_008",
        "topic": "efficient transformers",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_009",
        "topic": "federated learning for healthcare",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
    {
        "id": "task_010",
        "topic": "self-supervised representation learning",
        "goal": "produce literature analysis and at least one hypothesis",
        "required_artifacts": ["results.json", "summaries.json", "insights.json", "gaps.json", "hypotheses.json"],
    },
]


def get_benchmark_tasks(limit: int | None = None) -> list[dict]:
    """Return full or truncated benchmark tasks."""
    if limit is None or limit <= 0:
        return list(BENCHMARK_TASKS)
    return list(BENCHMARK_TASKS[:limit])
