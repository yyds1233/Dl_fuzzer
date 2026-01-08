import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DualStats:
    n_fast: int = 0
    mean_fast: float = 0.0
    n_slow: int = 0
    mean_slow: float = 0.0

    bad_streak: int = 0           # for soft elimination
    inactive_until: int = 0       # cooldown end step (based on selector step t)
    disabled: bool = False        # hard drop (optional)


class DualChannelUCBSoftElim:
    """
    Dual-channel UCB with cooldown-based soft elimination.

    - fast reward: frequent (cheap proxy)
    - slow reward: sparse (audit global increment)
    - selection uses combined mu + combined bonus
    - "soft elimination" becomes cooldown:
         if UCB_i < best_LCB - margin for patience times => inactive_until = t + cooldown_steps
      (not permanently dead; epsilon can still sample any non-disabled arms)
    """

    def __init__(
        self,
        *,
        c_fast: float = 2.0,
        c_slow: float = 2.0,
        epsilon: float = 0.02,
        elim_margin: float = 0.0,
        elim_patience: int = 3,
        elim_min_pulls: int = 8,
        alpha_min: float = 0.2,
        cooldown_steps: int = 50,   # NEW
        seed: int = 0,
    ):
        self.c_fast = float(c_fast)
        self.c_slow = float(c_slow)
        self.epsilon = float(epsilon)
        self.elim_margin = float(elim_margin)
        self.elim_patience = int(elim_patience)
        self.elim_min_pulls = int(elim_min_pulls)
        self.alpha_min = float(alpha_min)
        self.cooldown_steps = int(cooldown_steps)

        self.t = 0  # selection steps
        self.stats: Dict[str, DualStats] = {}
        self.rng = random.Random(seed)

    def ensure(self, arm_id: str) -> DualStats:
        if arm_id not in self.stats:
            self.stats[arm_id] = DualStats()
        return self.stats[arm_id]

    def disable(self, arm_id: str) -> None:
        s = self.ensure(arm_id)
        s.disabled = True

    def is_active(self, arm_id: str) -> bool:
        s = self.ensure(arm_id)
        if s.disabled:
            return False
        return self.t >= s.inactive_until

    def _alpha(self, s: DualStats) -> float:
        # adaptive: more slow samples => trust slow more; alpha is fast weight
        if s.n_slow <= 0:
            return 1.0
        a = 1.0 / (1.0 + float(s.n_slow))
        return max(self.alpha_min, a)

    def _mu(self, s: DualStats) -> float:
        a = self._alpha(s)
        return a * s.mean_fast + (1.0 - a) * s.mean_slow

    def _bonus(self, s: DualStats) -> float:
        lf = math.log(self.t + 1.0)
        b_fast = self.c_fast * math.sqrt(lf / max(1, s.n_fast))
        b_slow = self.c_slow * math.sqrt(lf / max(1, s.n_slow))
        return b_fast + b_slow

    def ucb_lcb(self, arm_id: str) -> Tuple[float, float]:
        s = self.ensure(arm_id)
        mu = self._mu(s)
        b = self._bonus(s)
        return mu + b, mu - b

    def update_fast(self, arm_id: str, reward: float) -> None:
        s = self.ensure(arm_id)
        if s.disabled:
            return
        s.n_fast += 1
        s.mean_fast += (float(reward) - s.mean_fast) / float(s.n_fast)

    def update_slow(self, arm_id: str, reward: float) -> None:
        s = self.ensure(arm_id)
        if s.disabled:
            return
        s.n_slow += 1
        s.mean_slow += (float(reward) - s.mean_slow) / float(s.n_slow)

    def maybe_soft_eliminate(self, arm_ids: List[str]) -> None:
        if not arm_ids:
            return

        # best LCB among ACTIVE & sufficiently-sampled arms
        best_lcb = -1e99
        any_ref = False
        for aid in arm_ids:
            s = self.ensure(aid)
            if s.disabled:
                continue
            if not self.is_active(aid):
                continue
            total_pulls = s.n_fast + s.n_slow
            if total_pulls < self.elim_min_pulls:
                continue
            _, lcb = self.ucb_lcb(aid)
            any_ref = True
            best_lcb = max(best_lcb, lcb)

        if not any_ref:
            return

        for aid in arm_ids:
            s = self.ensure(aid)
            if s.disabled:
                continue
            if not self.is_active(aid):
                # in cooldown => don't accumulate bad_streak
                continue

            total_pulls = s.n_fast + s.n_slow
            if total_pulls < self.elim_min_pulls:
                s.bad_streak = 0
                continue

            ucb, _ = self.ucb_lcb(aid)
            if ucb < best_lcb - self.elim_margin:
                s.bad_streak += 1
            else:
                s.bad_streak = 0

            if s.bad_streak >= self.elim_patience:
                # cooldown instead of permanent inactive
                s.bad_streak = 0
                s.inactive_until = max(s.inactive_until, self.t + self.cooldown_steps)

    def select(self, arm_ids: List[str]) -> str:
        if not arm_ids:
            raise ValueError("no arms")

        # filter disabled
        valid = [aid for aid in arm_ids if not self.ensure(aid).disabled]
        if not valid:
            raise ValueError("no valid arms (all disabled)")

        # epsilon exploration: can pick cooldown arms too (but not disabled)
        if self.rng.random() < self.epsilon:
            self.t += 1
            return self.rng.choice(valid)

        active = [aid for aid in valid if self.is_active(aid)]
        candidates = active if active else valid

        # ensure each candidate tried at least once (fast channel)
        for aid in candidates:
            s = self.ensure(aid)
            if s.n_fast == 0:
                self.t += 1
                return aid

        self.t += 1
        best_id: Optional[str] = None
        best_score = -1e99
        for aid in candidates:
            s = self.ensure(aid)
            score = self._mu(s) + self._bonus(s)
            if score > best_score:
                best_score = score
                best_id = aid
        assert best_id is not None
        return best_id

    def to_jsonable(self) -> Dict:
        return {
            "t": self.t,
            "c_fast": self.c_fast,
            "c_slow": self.c_slow,
            "epsilon": self.epsilon,
            "elim_margin": self.elim_margin,
            "elim_patience": self.elim_patience,
            "elim_min_pulls": self.elim_min_pulls,
            "alpha_min": self.alpha_min,
            "cooldown_steps": self.cooldown_steps,
            "stats": {
                k: {
                    "n_fast": v.n_fast,
                    "mean_fast": v.mean_fast,
                    "n_slow": v.n_slow,
                    "mean_slow": v.mean_slow,
                    "bad_streak": v.bad_streak,
                    "inactive_until": v.inactive_until,
                    "disabled": v.disabled,
                }
                for k, v in self.stats.items()
            },
        }
