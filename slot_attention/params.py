from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.001
    batch_size: int = 1
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
    num_workers: int = 16
    n_samples: int = 5
    warmup_steps_pct: float = 0
    decay_steps_pct: float = 0.05
    project_name: str = 'slot-attention-clevrer'
    logger_name: str = '匈牙利loss,10pred2'
    #restore: str = 'data/tmp/epoch=20-step=2658.ckpt'
    #restore: str = 'wandb/run-20210311_145725-3jreynlz/files/slot-attention-clevrer/3jreynlz/checkpoints/epoch=31-step=7562.ckpt'
    restore: str = '/mnt/diskb/qu_tang/slot_dynamic/wandb/run-20210312_152522-2nr2vu5b/files/slot-attention-clevrer/2nr2vu5b/checkpoints/epoch=26-step=6047.ckpt'

