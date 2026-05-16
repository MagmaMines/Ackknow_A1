from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    purpose: str


def load_default_catalog() -> list[DatasetEntry]:
    # 100+ placeholders representing integrated pretrain/runtime sets.
    base = [
        DatasetEntry("CommonCrawl-Refined", "web knowledge grounding"),
        DatasetEntry("CodeCorpus-Multi", "coding and patch reasoning"),
        DatasetEntry("Legal-Summaries", "compliance reference"),
        DatasetEntry("Medical-Abstracts", "risk-aware biosummaries"),
        DatasetEntry("News-Timeseries", "temporal event reasoning"),
    ]
    for i in range(1, 101):
        base.append(DatasetEntry(f"DomainPack-{i:03d}", "domain-specific adaptive memory"))
    return base
