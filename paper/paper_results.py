"""Paper results module — single source of truth for inline values.

All computed results referenced in the paper are derived here from
benchmark JSON output. Use {eval}`r.some_value` in MyST markdown.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MethodStats:
    name: str
    coverage: float
    precision: float
    recall: float
    density: float
    elapsed: float
    sipp_coverage: float
    cps_coverage: float
    psid_coverage: float

    @property
    def coverage_pct(self) -> str:
        return f"{self.coverage:.1%}"

    @property
    def precision_pct(self) -> str:
        return f"{self.precision:.1%}"

    @property
    def sipp_pct(self) -> str:
        return f"{self.sipp_coverage:.1%}"

    @property
    def cps_pct(self) -> str:
        return f"{self.cps_coverage:.1%}"

    @property
    def psid_pct(self) -> str:
        return f"{self.psid_coverage:.1%}"

    @property
    def time_str(self) -> str:
        return f"{self.elapsed:.0f}s"


@dataclass
class PaperResults:
    """All computed values for paper inline references."""

    # Method results
    qrf: MethodStats
    zi_qrf: MethodStats
    qdnn: MethodStats
    zi_qdnn: MethodStats
    maf: MethodStats
    zi_maf: MethodStats

    # Benchmark config
    k: int
    holdout_frac: float
    n_generate: int
    seed: int
    total_elapsed: float

    # Data characteristics
    n_sipp: int = 476_744
    n_cps: int = 144_265
    n_psid: int = 9_207
    n_total: int = 630_216
    n_sources: int = 3

    # Derived comparisons
    @property
    def best_method(self) -> str:
        methods = [self.qrf, self.zi_qrf, self.qdnn, self.zi_qdnn, self.maf, self.zi_maf]
        best = max(methods, key=lambda m: m.coverage)
        return best.name

    @property
    def best_coverage(self) -> str:
        methods = [self.qrf, self.zi_qrf, self.qdnn, self.zi_qdnn, self.maf, self.zi_maf]
        return max(m.coverage for m in methods)

    @property
    def best_coverage_pct(self) -> str:
        return f"{self.best_coverage:.1%}"

    @property
    def zi_maf_vs_maf_lift(self) -> str:
        lift = (self.zi_maf.coverage - self.maf.coverage) / self.maf.coverage * 100
        return f"{lift:.0f}%"

    @property
    def zi_qdnn_vs_qdnn_lift(self) -> str:
        lift = (self.zi_qdnn.coverage - self.qdnn.coverage) / self.qdnn.coverage * 100
        return f"{lift:.0f}%"

    @property
    def zi_qrf_vs_qrf_lift(self) -> str:
        lift = (self.zi_qrf.coverage - self.qrf.coverage) / self.qrf.coverage * 100
        return f"{lift:.0f}%"

    @property
    def zi_speedup_over_maf(self) -> str:
        ratio = self.zi_maf.elapsed / self.zi_qrf.elapsed
        return f"{ratio:.0f}x"

    @property
    def n_methods(self) -> int:
        return 6

    @property
    def total_elapsed_str(self) -> str:
        return f"{self.total_elapsed:.0f}s"


def _extract_method(data: dict, key: str, name: str) -> MethodStats:
    m = data["methods"][key]
    source_map = {s["source"]: s for s in m["sources"]}
    return MethodStats(
        name=name,
        coverage=m["mean_coverage"],
        precision=m["mean_precision"],
        recall=m["mean_recall"],
        density=m["mean_density"],
        elapsed=m["elapsed_seconds"],
        sipp_coverage=source_map.get("sipp", {}).get("coverage", 0),
        cps_coverage=source_map.get("cps", {}).get("coverage", 0),
        psid_coverage=source_map.get("psid", {}).get("coverage", 0),
    )


def load_results(path: str = None) -> PaperResults:
    if path is None:
        path = str(Path(__file__).parent.parent / "benchmarks" / "results" / "benchmark_full.json")
    with open(path) as f:
        data = json.load(f)

    return PaperResults(
        qrf=_extract_method(data, "QRF", "QRF"),
        zi_qrf=_extract_method(data, "ZI-QRF", "ZI-QRF"),
        qdnn=_extract_method(data, "QDNN", "QDNN"),
        zi_qdnn=_extract_method(data, "ZI-QDNN", "ZI-QDNN"),
        maf=_extract_method(data, "MAF", "MAF"),
        zi_maf=_extract_method(data, "ZI-MAF", "ZI-MAF"),
        k=data["k"],
        holdout_frac=data["holdout_frac"],
        n_generate=data["n_generate"],
        seed=data["seed"],
        total_elapsed=data["total_elapsed_seconds"],
    )


# Module-level instance for {eval}`r.xxx` in MyST
r = load_results()
