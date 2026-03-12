# screen/profile_space.py
from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Tuple

# Move profile grid here (formerly in screen_api_profile.py)
PROFILE_GRID: Dict[str, List[object]] = {
    "MUT_STEPS_MAX": [2, 6, 10],
    "P_TYPE_MUT": [0.2, 0.5, 0.8],
    "P_SHAPE_MUT": [0.0, 0.05, 0.15],
    "SEED_TRIES": [3, 8, 12],
    "MUT_ATTEMPTS": [3, 6, 10],
    "P_NONCONTIG": [0.0, 0.02, 0.05],
    "P_RECONTIG": [0.0, 0.05, 0.10],
    "ALLOW_EMPTY": [0, 1],
    "P_EMPTY_DIM": [0.0, 0.005, 0.01],
    "P_EMPTY_NC": [0.0, 0.0005, 0.001],
    "ENABLE_MUT": [0, 1],
}


def canonical_profile_id(profile: Dict[str, object], length: int = 10) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def random_profile(rng: random.Random) -> Dict[str, object]:
    p = {k: rng.choice(v) for k, v in PROFILE_GRID.items()}
    return p


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def mutate_profile(parent: Dict[str, object], rng: random.Random) -> Dict[str, object]:
    """
    Minimal, robust mutate:
      - small prob per key to perturb
      - discrete: neighbor jump preferred if list looks ordered
      - float prob-like: +/- 0.02 clamp
      - bool/int switch: rare flip
    """
    child = dict(parent)
    for k, choices in PROFILE_GRID.items():
        if k not in child:
            continue

        # mutation probability per parameter (conservative)
        if rng.random() > 0.15:
            continue

        cur = child[k]
        # binary-ish
        if choices == [0, 1] or set(choices) == {0, 1}:
            if rng.random() < 0.5:
                child[k] = 1 if int(cur) == 0 else 0
            continue

        # float-like in [0,1]
        if isinstance(cur, float) and all(isinstance(x, (int, float)) for x in choices):
            if 0.0 <= float(cur) <= 1.0:
                delta = rng.choice([-1.0, 1.0]) * 0.02
                child[k] = clamp01(float(cur) + delta)
                # snap to nearest choice if grid is discrete floats
                # (keeps search inside your designed grid)
                nearest = min(choices, key=lambda x: abs(float(x) - float(child[k])))
                child[k] = float(nearest)
                continue

        # discrete list: try neighbor jump
        try:
            idx = choices.index(cur)
        except ValueError:
            idx = None

        if idx is not None and len(choices) >= 2:
            if rng.random() < 0.7:
                step = rng.choice([-1, 1])
                nidx = max(0, min(len(choices) - 1, idx + step))
                child[k] = choices[nidx]
            else:
                child[k] = rng.choice(choices)
        else:
            child[k] = rng.choice(choices)

    return child


def make_profile_arm(profile: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    pid = canonical_profile_id(profile)
    return pid, profile