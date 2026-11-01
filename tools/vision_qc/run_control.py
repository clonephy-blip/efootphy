"""Epoch-boundary stopping and atomic progress for bounded research runs."""
import json
import math
import time
from pathlib import Path


class RunControl:
    def __init__(self, output, minutes=None, clock=time.monotonic):
        if minutes is not None and (not math.isfinite(minutes) or minutes <= 0):
            raise ValueError('Budget must be positive and finite')
        self.output = Path(output)
        self.budget = None if minutes is None else minutes * 60
        self.clock = clock
        self.started = self.previous = None
        self.reason = None
        self.seconds = 0.0

    def start(self, trainer=None):
        self.started = self.previous = self.clock()

    def epoch_end(self, trainer):
        if self.started is None:
            raise RuntimeError('RunControl was not started')
        now = self.clock()
        epoch_seconds = now - self.previous
        self.previous = now
        self.seconds = now - self.started
        if (self.output / 'STOP_AFTER_EPOCH').exists():
            self.reason = 'stop_file'
        elif self.budget is not None and self.seconds + epoch_seconds * 1.15 >= self.budget:
            self.reason = 'budget_next_epoch_estimate'
        if self.reason:
            trainer.stop = True
        import torch
        progress = {
            'epochs_completed': int(trainer.epoch) + 1,
            'elapsed_seconds': round(self.seconds, 3),
            'last_epoch_seconds': round(epoch_seconds, 3),
            'budget_seconds': self.budget,
            'stop_reason': self.reason,
            'torch_threads': torch.get_num_threads(),
            'optimizer_updates': getattr(getattr(trainer,'ema',None),'updates',None),
            'gradient_accumulation': getattr(trainer,'accumulate',None),
            'nominal_batch_size': getattr(getattr(trainer,'args',None),'nbs',None),
            'last_checkpoint_present': Path(trainer.last).is_file(),
            'best_checkpoint_present': Path(trainer.best).is_file(),
            'metrics': {k: float(v) for k, v in trainer.metrics.items()},
            'budget_scope': 'training epoch boundaries; setup, final validation/export excluded',
        }
        temporary = self.output / 'progress.json.tmp'
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(self.output / 'progress.json')
        print('VISION_PROGRESS ' + json.dumps(progress), flush=True)
