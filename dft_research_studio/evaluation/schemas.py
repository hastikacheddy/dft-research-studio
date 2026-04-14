from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class GenerationResult(BaseModel):
    text:         str
    input_tokens: int   = Field(ge=0)
    output_tokens:int   = Field(ge=0)
    total_tokens: int   = Field(ge=0)
    cost_usd:     float = Field(ge=0.0)
    api_calls:    int   = Field(ge=0)

    @model_validator(mode="after")
    def check_token_sum(self) -> "GenerationResult":
        if self.total_tokens != self.input_tokens + self.output_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self

class RetrievalMetrics(BaseModel):
    recall_at_k:    float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    mrr:            float = Field(ge=0.0, le=1.0)

class JudgeMetrics(BaseModel):
    correctness:             float = Field(ge=0.0, le=5.0)
    relevance:               float = Field(ge=0.0, le=5.0)
    groundedness:            float = Field(ge=0.0, le=1.0)
    retrieval_relevance_llm: float = Field(ge=0.0, le=5.0, default=0.0)
    rouge1_fmeasure:         float = Field(ge=0.0, le=1.0)
    rougeL_fmeasure:         float = Field(ge=0.0, le=1.0)
    cohens_d_vs_baseline:    Optional[float] = None
    gold_retrieval_recall_at_k:    float = Field(ge=0.0, le=1.0, default=0.0)
    gold_retrieval_precision_at_k: float = Field(ge=0.0, le=1.0, default=0.0)
    gold_retrieval_mrr:            float = Field(ge=0.0, le=1.0, default=0.0)
    judge_model: str = "unknown"

    @field_validator("correctness", mode="before")
    @classmethod
    def clamp_correctness(cls, v: Any) -> float:
        return max(1.0, min(5.0, float(v)))

    @field_validator("groundedness", mode="before")
    @classmethod
    def clamp_groundedness(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))

class ExperimentResult(BaseModel):
    question_id:                 str
    question:                    str
    ground_truth:                str
    gold_docs:                   List[str]       = Field(default_factory=list)
    model:                       str
    judge_model:                 str             = "unknown"
    distractor_ratio:            float           = Field(ge=0.0)
    experiment_type:             str
    mode:                        str
    generated_answer:            str
    context_used:                str             = ""
    retrieved_text_chunks:       List[str]       = Field(default_factory=list)
    retrieved_source_filenames:  List[str]       = Field(default_factory=list)
    metrics:                     Dict[str, Any]  = Field(default_factory=dict)
    latency_ms:                  Optional[float] = None

    @field_validator("distractor_ratio", mode="before")
    @classmethod
    def coerce_ratio(cls, v: Any) -> float:
        return float(v)

class GFGResult(BaseModel):
    model:            str
    distractor_ratio: float
    mean_correctness: float
    ceiling:          float
    gfg:              float = Field(ge=0.0, le=1.0)
    gfg_std_error:    float = Field(ge=0.0)
    interpretation:   str
    n_samples:        int   = Field(ge=1)
