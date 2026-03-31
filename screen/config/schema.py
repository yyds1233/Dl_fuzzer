from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProfileArm:
    profile_id: str
    profile: Dict[str, object]


@dataclass
class HarnessCandidate:
    harness_id: str
    harness_path: Path
    group_id: str = "OTHERS"
    profiles: Optional[List[ProfileArm]] = None


@dataclass
class StepResult:
    t: int
    harness_id: str
    profile_id: str
    delta_ft: int
    delta_cov: int
    exec_s: float
    proxy_reward: float
    fast_reward: float
    delta_files_epoch: int

    audited_harnesses: int
    slow_harness: Optional[int]
    # kept for backward-compatible state shape; profile slow attribution is disabled
    slow_profile_credit: Optional[float] = None


@dataclass
class BanditParams:
    c_fast: float = 2.0
    c_slow: float = 2.0
    epsilon_harness: float = 0.02
    epsilon_profile: float = 0.05
    alpha_min: float = 0.2
    elim_margin: float = 0.0
    elim_patience: int = 3
    elim_min_pulls: int = 8
    cooldown_steps: int = 50
    seed: int = 0


@dataclass
class AuditParams:
    # NEW semantics: local per-harness trigger, not global step trigger.
    audit_every: int = 10
    # Optional second trigger: audit as soon as this harness accumulates enough new files.
    audit_min_delta_files: int = 0
    full_corpus_audit: bool = False
    audit_max_inputs: int = 0
    slow_metric: str = "BRH"  # BRH/LH/FNH

    # Deprecated: kept only for config compatibility; no longer used.
    audit_profile_topk: int = 5
    min_credit_inputs: int = 20
    zero_slow_penalty: float = 0.0

    cov_venv_activate: Path = Path("/root/pytorch_cov/bin/activate")
    cov_audit_script: Path = Path("cov_global_union_audit.py")
    global_dir: Path = Path("global_union")
    primary_object: str = ""
    extra_object: List[str] = field(default_factory=list)
    ignore_filename_regex: Optional[str] = None
    cov_replay_extra: str = ""


@dataclass
class RuntimeParams:
    root: Path = Path("fuzz_output")
    python: str = "python3"
    epoch: int = 60
    steps: int = 200
    fuzz_flags: str = "-ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1"
    mix: float = 0.7
    manifest_dir: Path = Path("manifests")


@dataclass
class PoolParams:
    k: int = 10
    refresh_every: int = 200
    keep_frac: float = 0.5
    replace_frac: float = 0.3
    inject_each_refresh: int = 1
    min_pulls_to_kill: int = 30


@dataclass
class PriorParams:
    elite_size: int = 100
    ewma_alpha: float = 0.4
    enabled: bool = True

    # Scheme B: harness slow is admission gate, profile fast is score.
    min_pulls_for_admit: int = 5
    top_n_fast: int = 2
    reward_clip: float = 1_000_000.0


@dataclass
class DriverConfig:
    runtime: RuntimeParams
    bandit: BanditParams
    audit: AuditParams

    harnesses_json: Optional[Path] = None
    harness: Optional[Path] = None
    harness_id: Optional[str] = None
    top_json: Optional[Path] = None
    groups_map: Optional[Path] = None

    pool: PoolParams = field(default_factory=PoolParams)
    prior: PriorParams = field(default_factory=PriorParams)

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        d["runtime"]["root"] = str(self.runtime.root)
        d["runtime"]["manifest_dir"] = str(self.runtime.manifest_dir)
        d["audit"]["cov_venv_activate"] = str(self.audit.cov_venv_activate)
        d["audit"]["cov_audit_script"] = str(self.audit.cov_audit_script)
        d["audit"]["global_dir"] = str(self.audit.global_dir)
        d["harnesses_json"] = str(self.harnesses_json) if self.harnesses_json else None
        d["harness"] = str(self.harness) if self.harness else None
        d["top_json"] = str(self.top_json) if self.top_json else None
        d["groups_map"] = str(self.groups_map) if self.groups_map else None
        return d