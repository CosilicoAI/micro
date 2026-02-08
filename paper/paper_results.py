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
class ReweightingMethodStats:
    name: str
    mean_relative_error: float
    max_relative_error: float
    weight_cv: float
    sparsity: float
    elapsed: float

    @property
    def mean_error_pct(self) -> str:
        return f"{self.mean_relative_error:.1%}"

    @property
    def max_error_pct(self) -> str:
        return f"{self.max_relative_error:.1%}"

    @property
    def cv_str(self) -> str:
        return f"{self.weight_cv:.3f}"

    @property
    def sparsity_pct(self) -> str:
        return f"{self.sparsity:.1%}"

    @property
    def time_str(self) -> str:
        return f"{self.elapsed:.2f}s"


@dataclass
class PaperResults:
    """All computed values for paper inline references."""

    # Synthesis method results
    qrf: MethodStats
    zi_qrf: MethodStats
    qdnn: MethodStats
    zi_qdnn: MethodStats
    maf: MethodStats
    zi_maf: MethodStats

    # Reweighting method results
    rw_ipf: ReweightingMethodStats
    rw_entropy: ReweightingMethodStats
    rw_sparse_cal: ReweightingMethodStats
    rw_l1: ReweightingMethodStats
    rw_l0: ReweightingMethodStats

    # Synthesis benchmark config
    k: int
    holdout_frac: float
    n_generate: int
    seed: int
    total_elapsed: float

    # Reweighting benchmark config
    rw_n_records: int = 5000
    rw_n_marginal_targets: int = 7
    rw_n_continuous_targets: int = 1

    # Data characteristics
    n_sipp: int = 476_744
    n_cps: int = 144_265
    n_psid: int = 9_207
    n_total: int = 630_216
    n_sources: int = 3

    # Synthesis derived comparisons
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

    # Reweighting derived comparisons
    @property
    def best_rw_method(self) -> str:
        """Best reweighting method by lowest mean relative error (calibration methods only)."""
        calibration_methods = [self.rw_ipf, self.rw_entropy, self.rw_sparse_cal]
        best = min(calibration_methods, key=lambda m: m.mean_relative_error)
        return best.name

    @property
    def best_rw_error(self) -> str:
        calibration_methods = [self.rw_ipf, self.rw_entropy, self.rw_sparse_cal]
        return f"{min(m.mean_relative_error for m in calibration_methods):.1%}"

    @property
    def entropy_vs_ipf_error_reduction(self) -> str:
        reduction = (self.rw_ipf.mean_relative_error - self.rw_entropy.mean_relative_error) / self.rw_ipf.mean_relative_error * 100
        return f"{reduction:.0f}%"

    @property
    def sparse_cal_cv_vs_ipf(self) -> str:
        reduction = (self.rw_ipf.weight_cv - self.rw_sparse_cal.weight_cv) / self.rw_ipf.weight_cv * 100
        return f"{reduction:.0f}%"

    @property
    def rw_n_targets_total(self) -> int:
        return self.rw_n_marginal_targets + self.rw_n_continuous_targets


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


def _extract_rw_method(data: dict, key: str) -> ReweightingMethodStats:
    m = data["methods"][key]
    return ReweightingMethodStats(
        name=m["method_name"],
        mean_relative_error=m["mean_relative_error"],
        max_relative_error=m["max_relative_error"],
        weight_cv=m["weight_cv"],
        sparsity=m["sparsity"],
        elapsed=m["elapsed_seconds"],
    )


def load_results(
    synthesis_path: str = None,
    reweighting_path: str = None,
) -> PaperResults:
    results_dir = Path(__file__).parent.parent / "benchmarks" / "results"

    if synthesis_path is None:
        synthesis_path = str(results_dir / "benchmark_full.json")
    if reweighting_path is None:
        reweighting_path = str(results_dir / "reweighting_full.json")

    with open(synthesis_path) as f:
        synth_data = json.load(f)

    with open(reweighting_path) as f:
        rw_data = json.load(f)

    return PaperResults(
        # Synthesis
        qrf=_extract_method(synth_data, "QRF", "QRF"),
        zi_qrf=_extract_method(synth_data, "ZI-QRF", "ZI-QRF"),
        qdnn=_extract_method(synth_data, "QDNN", "QDNN"),
        zi_qdnn=_extract_method(synth_data, "ZI-QDNN", "ZI-QDNN"),
        maf=_extract_method(synth_data, "MAF", "MAF"),
        zi_maf=_extract_method(synth_data, "ZI-MAF", "ZI-MAF"),
        k=synth_data["k"],
        holdout_frac=synth_data["holdout_frac"],
        n_generate=synth_data["n_generate"],
        seed=synth_data["seed"],
        total_elapsed=synth_data["total_elapsed_seconds"],
        # Reweighting
        rw_ipf=_extract_rw_method(rw_data, "IPF"),
        rw_entropy=_extract_rw_method(rw_data, "Entropy"),
        rw_sparse_cal=_extract_rw_method(rw_data, "SparseCalibrator"),
        rw_l1=_extract_rw_method(rw_data, "L1-Sparse"),
        rw_l0=_extract_rw_method(rw_data, "L0-Sparse"),
        rw_n_records=rw_data["n_records"],
        rw_n_marginal_targets=rw_data["n_marginal_targets"],
        rw_n_continuous_targets=rw_data["n_continuous_targets"],
    )


# Module-level instance for {eval}`r.xxx` in MyST
r = load_results()
