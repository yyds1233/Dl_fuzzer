from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from screen.config.schema import ProfileArm


@dataclass
class EliteEntry:
    profile_id: str
    profile: Dict[str, object]
    score: float = 0.0
    n: int = 0
    last_t: int = 0


class GroupPriorManager:
    """
    Per-group elite store.

    After the slow/profile decoupling, this store remains generic:
      - read path: cold-start / refresh injection
      - write path: caller decides the admission gate and the score signal
        (current Scheme B: harness slow gate + profile fast score)
    """

    def __init__(self, *, state_dir: Path, elite_size: int = 100, ewma_alpha: float = 0.4):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.elite_size = int(elite_size)
        self.ewma_alpha = float(ewma_alpha)
        self._cache: Dict[str, Dict[str, EliteEntry]] = {}

    def _path(self, group_id: str) -> Path:
        return self.state_dir / f"{group_id}.json"

    def _load_group(self, group_id: str) -> Dict[str, EliteEntry]:
        if group_id in self._cache:
            return self._cache[group_id]
        p = self._path(group_id)
        m: Dict[str, EliteEntry] = {}
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for e in data.get("elite", []):
                ent = EliteEntry(
                    profile_id=str(e["profile_id"]),
                    profile=dict(e["profile"]),
                    score=float(e.get("score", 0.0)),
                    n=int(e.get("n", 0)),
                    last_t=int(e.get("last_t", 0)),
                )
                m[ent.profile_id] = ent
        self._cache[group_id] = m
        return m

    def _save_group(self, group_id: str) -> None:
        m = self._load_group(group_id)
        elite = sorted(m.values(), key=lambda x: (x.score, x.last_t), reverse=True)[: self.elite_size]
        out = {"group_id": group_id, "elite_size": self.elite_size, "elite": [asdict(e) for e in elite]}
        self._path(group_id).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    def init_profiles(self, *, group_id: str, k: int, seed: int) -> List[ProfileArm]:
        if group_id == "OTHERS":
            return []
        import random

        rng = random.Random(seed)
        m = self._load_group(group_id)
        elite = sorted(m.values(), key=lambda x: (x.score, x.last_t), reverse=True)
        if not elite:
            return []

        window = elite[: min(len(elite), max(k * 3, k))]
        rng.shuffle(window)
        picked = window[: min(k, len(window))]
        return [ProfileArm(profile_id=e.profile_id, profile=e.profile) for e in picked]

    def observe(self, *, group_id: str, profile_id: str, profile: Dict[str, object], reward: float, t: int) -> None:
        if group_id == "OTHERS":
            return
        m = self._load_group(group_id)
        ent = m.get(profile_id)
        if ent is None:
            ent = EliteEntry(profile_id=profile_id, profile=profile, score=0.0, n=0, last_t=0)
            m[profile_id] = ent

        ent.n += 1
        ent.last_t = int(t)
        a = self.ewma_alpha
        ent.score = (1.0 - a) * float(ent.score) + a * float(reward)

        self._save_group(group_id)