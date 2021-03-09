from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.0001
    batch_size: int = 2
    val_batch_size: int = 1
    resolution: Tuple[int, int] = (160, 240)
    num_slots: int = 7
    num_iterations: int = 3
    data_root: str = "data/CLEVRER_v2/"
    gpus: int = 8
    max_epochs: int = 500
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 8
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    project_name: str = 'slot-attention-clevrer'
    logger_name: str = 'pred4_160_240_firstrun'
    restore: str = '' #wandb/run-20210307_235252-10cnzsc0/files/slot-attention-clevrer/10cnzsc0/checkpoints/epoch=18-step=17137.ckpt'

