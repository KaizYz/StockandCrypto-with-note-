from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from typing import Iterable, List

import numpy as np


def utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def parse_horizons(values: Iterable[int | str]) -> List[int]:
    out: List[int] = []
    for v in values:
        if isinstance(v, int):
            out.append(v)
        else:
            out.append(int(str(v)))
    return out

