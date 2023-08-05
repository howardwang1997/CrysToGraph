from collections import Counter
from bisect import bisect_right
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 milestones: int,
                 warmup_steps: int=10,
                 gamma: float=0.1,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:
        self.milestones = milestones
        self.gamma = gamma

        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        score = min(self.gamma ** bisect_right(milestones, self.last_epoch), \
                    1 / warmup_steps * self._step_count)
        return [base_lr * score
                for base_lr in self.base_lrs]
