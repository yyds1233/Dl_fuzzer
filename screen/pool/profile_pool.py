from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from screen.config.schema import ProfileArm
from screen.profile_space import make_profile_arm, mutate_profile, random_profile


@dataclass
class ArmState:
    profile_id: str
    profile: Dict[str, object]
    born_t: int = 0
    pulls: int = 0
    fast_ewma: float = 0.0
    # kept for backward-compatible state loading; no longer used for selection/refresh
    slow_ewma: float = 0.0
    slow_n: int = 0


class ProfilePoolManager:
    def __init__(
        self,
        *,
        state_dir: Path,
        k: int,
        refresh_every: int,
        keep_frac: float,
        replace_frac: float,
        inject_each_refresh: int,
        min_pulls_to_kill: int,
        seed: int = 0,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self.refresh_every = int(refresh_every)
        self.keep_frac = float(keep_frac)
        self.replace_frac = float(replace_frac)
        self.inject_each_refresh = int(inject_each_refresh)
        self.min_pulls_to_kill = int(min_pulls_to_kill)
        self.seed = int(seed)

        self._pools: Dict[str, Dict[str, ArmState]] = {}

    def _path(self, hid: str) -> Path:
        return self.state_dir / f"{hid}.json"

    def _load(self, hid: str) -> Dict[str, ArmState]:
        if hid in self._pools:
            return self._pools[hid]
        p = self._path(hid)
        m: Dict[str, ArmState] = {}
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for e in data.get("profiles", []):
                st = ArmState(
                    profile_id=str(e["profile_id"]),
                    profile=dict(e["profile"]),
                    born_t=int(e.get("born_t", 0)),
                    pulls=int(e.get("pulls", 0)),
                    fast_ewma=float(e.get("fast_ewma", 0.0)),
                    slow_ewma=float(e.get("slow_ewma", 0.0)),
                    slow_n=int(e.get("slow_n", 0)),
                )
                m[st.profile_id] = st
        self._pools[hid] = m
        return m

    def _save(self, hid: str) -> None:
        m = self._load(hid)
        profiles = [asdict(x) for x in m.values()]
        out = {"harness_id": hid, "k": self.k, "profiles": profiles}
        self._path(hid).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    def get_active_profiles(self, hid: str) -> List[ProfileArm]:
        m = self._load(hid)
        return [ProfileArm(profile_id=s.profile_id, profile=s.profile) for s in m.values()]

    def get_profile(self, hid: str, pid: str) -> Optional[Dict[str, object]]:
        m = self._load(hid)
        st = m.get(pid)
        return dict(st.profile) if st else None

    def get_arm_state(self, hid: str, pid: str) -> Optional[ArmState]:
        return self._load(hid).get(pid)

    def ranked_states(self, hid: str) -> List[ArmState]:
        items = list(self._load(hid).values())
        items.sort(key=lambda x: (self._fitness(x), x.fast_ewma, -x.born_t), reverse=True)
        return items

    def maybe_init_pool(
        self,
        *,
        hid: str,
        group_id: str,
        t: int,
        group_prior,
    ) -> None:
        m = self._load(hid)
        if len(m) >= self.k:
            return

        import random

        rng = random.Random(self.seed + (hash(hid) & 0xFFFF))
        need = self.k - len(m)

        arms: List[ProfileArm] = []
        if group_id != "OTHERS":
            take_prior = int(round(self.k * 0.4))
            arms.extend(group_prior.init_profiles(group_id=group_id, k=take_prior, seed=rng.randint(0, 2**31 - 1)))

        while len(arms) < need:
            p = random_profile(rng)
            pid, prof = make_profile_arm(p)
            arms.append(ProfileArm(profile_id=pid, profile=prof))

        for a in arms:
            if a.profile_id in m:
                continue
            m[a.profile_id] = ArmState(profile_id=a.profile_id, profile=a.profile, born_t=int(t))
            if len(m) >= self.k:
                break

        self._save(hid)

    def on_fast(self, hid: str, pid: str, reward: float) -> None:
        m = self._load(hid)
        st = m.get(pid)
        if not st:
            return
        st.pulls += 1
        a = 0.3
        st.fast_ewma = (1.0 - a) * st.fast_ewma + a * float(reward)
        self._save(hid)

    def on_slow_credit(self, hid: str, pid: str, credit: float) -> None:
        """Deprecated compatibility hook; profile slow attribution is disabled."""
        m = self._load(hid)
        st = m.get(pid)
        if not st:
            return
        st.slow_n += 1
        a = 0.4
        st.slow_ewma = (1.0 - a) * st.slow_ewma + a * float(credit)
        self._save(hid)

    def _fitness(self, st: ArmState) -> float:
        # Profile-level decisions are now fast-dominated.
        return st.fast_ewma

    def maybe_refresh(self, *, hid: str, group_id: str, t: int, group_prior) -> bool:
        if self.refresh_every <= 0 or (t % self.refresh_every != 0):
            return False

        m = self._load(hid)
        if len(m) < max(2, self.k // 2):
            return False

        items = list(m.values())
        items.sort(key=self._fitness, reverse=True)

        keep_n = max(1, int(round(self.k * self.keep_frac)))
        replace_n = max(0, int(round(self.k * self.replace_frac)))
        keep = items[:keep_n]
        tail = items[keep_n:]

        kill_candidates = [x for x in tail if x.pulls >= self.min_pulls_to_kill]
        if not kill_candidates:
            return False

        kill = kill_candidates[-replace_n:] if replace_n > 0 else []
        kill_ids = {x.profile_id for x in kill}

        for pid in kill_ids:
            m.pop(pid, None)

        import random

        rng = random.Random(self.seed + (hash(hid) & 0xFFFF) + t)

        if group_id != "OTHERS" and self.inject_each_refresh > 0:
            injected_arms = group_prior.init_profiles(group_id=group_id, k=self.inject_each_refresh, seed=rng.randint(0, 2**31 - 1))
            for a in injected_arms:
                if a.profile_id in m:
                    continue
                m[a.profile_id] = ArmState(profile_id=a.profile_id, profile=a.profile, born_t=int(t))
                if len(m) >= self.k:
                    break

        while len(m) < self.k:
            parent = rng.choice(keep)
            child_profile = mutate_profile(parent.profile, rng)
            cid, cprof = make_profile_arm(child_profile)
            if cid in m:
                continue
            m[cid] = ArmState(profile_id=cid, profile=cprof, born_t=int(t))

        self._save(hid)
        return True