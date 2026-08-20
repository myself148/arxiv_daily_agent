from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStage(str, Enum):
    RESEARCH = "research"
    REVIEW = "review"
    REPORT = "report"


class PaperRecord(TypedDict):
    title: str
    authors: List[str]
    published_date: str
    summary: str
    entry_id: str
    pdf_url: str


class PipelineState(TypedDict, total=False):
    run_id: str
    mode: str
    query: str
    max_results: int
    rag_enabled: bool
    provider_name: Optional[str]
    skipped_count: int
    papers: List[Dict[str, Any]]
    full_texts: List[str]
    chunk_summaries: List[List[str]]
    retrieved_contexts: List[str]
    review_sources: List[str]
    reviews: List[str]
    paper_tags: List[List[str]]
    summaries: List[str]
    final_report: str


@dataclass(frozen=True)
class RunRequest:
    mode: str
    query: str
    max_results: int
    rag_enabled: bool = True
    provider_name: Optional[str] = None
    run_id: str = field(default_factory=lambda: uuid4().hex)

    def validate(self) -> None:
        if self.mode not in {"single", "graph"}:
            raise ValueError("mode must be either 'single' or 'graph'")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.max_results <= 0:
            raise ValueError("max_results must be greater than 0")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RunRequest":
        return cls(
            mode=str(value["mode"]),
            query=str(value["query"]),
            max_results=int(value["max_results"]),
            rag_enabled=bool(value.get("rag_enabled", True)),
            provider_name=value.get("provider_name"),
            run_id=str(value["run_id"]),
        )


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: StageStatus
    updates: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class RunSnapshot:
    request: RunRequest
    status: RunStatus
    state: PipelineState
    completed_stages: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    result: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def run_id(self) -> str:
        return self.request.run_id
