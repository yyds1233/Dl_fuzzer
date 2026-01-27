# screen/config/schema.py
from __future__ import annotations

from dataclasses import dataclass, asdict
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
    profiles: List[ProfileArm]


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
    slow_profile_credit: Optional[float]


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
    audit_every: int = 10
    full_corpus_audit: bool = False
    audit_max_inputs: int = 2000
    audit_profile_topk: int = 5
    slow_metric: str = "BRH"  # BRH/LH/FNH
    min_credit_inputs: int = 20
    zero_slow_penalty: float = 0.0

    cov_venv_activate: Path = Path("/root/pytorch_cov/bin/activate")
    cov_audit_script: Path = Path("cov_global_union_audit.py")
    global_dir: Path = Path("global_union")
    primary_object: str = ""
    extra_object: List[str] = None
    ignore_filename_regex: Optional[str] = None
    cov_replay_extra: str = ""

    def __post_init__(self):
        if self.extra_object is None:
            self.extra_object = []


@dataclass
class RuntimeParams:
    root: Path = Path("fuzz_output")
    python: str = "python3"
    epoch: int = 60
    steps: int = 200  # >0 run N; 0 run forever
    fuzz_flags: str = "-ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1"
    mix: float = 0.7
    manifest_dir: Path = Path("manifests")


@dataclass
class DriverConfig:
    runtime: RuntimeParams
    bandit: BanditParams
    audit: AuditParams

    # Inputs (either harnesses_json or legacy harness/top_json)
    harnesses_json: Optional[Path] = None
    harness: Optional[Path] = None
    harness_id: Optional[str] = None
    top_json: Optional[Path] = None

    def to_jsonable(self) -> Dict[str, Any]:
        # Path -> str for saving
        d = asdict(self)
        # convert nested Paths
        d["runtime"]["root"] = str(self.runtime.root)
        d["runtime"]["manifest_dir"] = str(self.runtime.manifest_dir)
        d["audit"]["cov_venv_activate"] = str(self.audit.cov_venv_activate)
        d["audit"]["cov_audit_script"] = str(self.audit.cov_audit_script)
        d["audit"]["global_dir"] = str(self.audit.global_dir)
        d["harnesses_json"] = str(self.harnesses_json) if self.harnesses_json else None
        d["harness"] = str(self.harness) if self.harness else None
        d["top_json"] = str(self.top_json) if self.top_json else None
        return d
