import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
