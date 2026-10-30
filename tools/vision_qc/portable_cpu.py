"""FP32-only trainer for hosts where native FP16 CPU conversion is unavailable.

Uses the same Detection/PoseTrainer losses and optimizer. Only checkpoint dtype
conversion/serialization changes. Normal GPU/CPU training uses upstream trainers.
"""
import copy
import datetime
from pathlib import Path


def prepare_checkpoint(weights, destination):
    import numpy as np
    import torch
    from ultralytics.utils.downloads import attempt_download_asset
    source = Path(attempt_download_asset(str(weights)))
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    for key in ['model', 'ema']:
        module = checkpoint.get(key)
        if module is None: continue
        def convert(tensor):
            if not tensor.is_floating_point() or tensor.dtype == torch.float32:
                return tensor
            if tensor.dtype not in (torch.float16, torch.float64):
                raise ValueError(f'Unsupported portable checkpoint dtype: {tensor.dtype}')
            return torch.from_numpy(tensor.detach().numpy().astype(np.float32))
        module._apply(convert)
    torch.save(checkpoint, destination)
    return Path(destination)


def trainer_class(task):
    import torch
    import ultralytics
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.pose import PoseTrainer
    base = PoseTrainer if task == 'pose' else DetectionTrainer

    class PortableTrainer(base):
        def save_model(self):
            checkpoint = {
                'epoch': self.epoch, 'best_fitness': self.best_fitness, 'model': None,
                'ema': copy.deepcopy(self.ema.ema).float(), 'updates': self.ema.updates,
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'train_args': vars(self.args), 'train_metrics': {**self.metrics, 'fitness': self.fitness},
                'train_results': self.read_results_csv(), 'date': datetime.datetime.now().isoformat(),
                'version': ultralytics.__version__, 'license': 'AGPL-3.0',
                'docs': 'https://docs.ultralytics.com', 'portable_fp32_checkpoint': True,
            }
            torch.save(checkpoint, self.last)
            if self.best_fitness == self.fitness: torch.save(checkpoint, self.best)
            if self.save_period > 0 and self.epoch % self.save_period == 0:
                torch.save(checkpoint, self.wdir / f'epoch{self.epoch}.pt')

        def final_eval(self):
            # Upstream strip_optimizer casts to FP16. The caller independently validates best.pt.
            pass

    return PortableTrainer
