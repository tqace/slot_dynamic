from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.0004
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
    warmup_steps_pct: float = 0
    decay_steps_pct: float = 0.01
    project_name: str = 'slot-attention-clevrer'
    logger_name: str = 'pred4<-'
    restore: str = '/data1/qu_tang/slot_dynamic/wandb/run-20210310_222153-13s39vbr/files/slot-attention-clevrer/13s39vbr/checkpoints/epoch=208-step=25467.ckpt'

