from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import mean
from time import time

from torch.utils.tensorboard import SummaryWriter

from code_transformer.utils.log import Logger


class MetricsLogger(ABC):

    @abstractmethod
    def log_scalar(self, name, value, step, timestep=None, **kwargs):
        pass

    @abstractmethod
    def log_text(self, name: str, text: str, step: int = None, timestep: float = None, **kwargs):
        pass

    def log_scalars(self, name, values_dict: dict, step, timestep=None, **kwargs):
        pass

    @abstractmethod
    def flush(self):
        pass


class ExperimentLogger(Logger):

    def __init__(self, name, metrics_logger: MetricsLogger):
        super(ExperimentLogger, self).__init__(name)
        self.sub_batch_metrics = defaultdict(list)
        self.time_last_flush = None
        self.last_step = None
        self.metrics_logger = metrics_logger

    def log_metrics(self, metrics: dict, step=None):
        for metric_name, value in metrics.items():
            self.metrics_logger.log_scalar(metric_name, value, step=step)

    def log_text(self, key: str, text: str, step=None):
        self.metrics_logger.log_text(key, text, step=step)

    def log_sub_batch_metrics(self, metrics: dict):
        for metric_name, value in metrics.items():
            if value is not None:
                self.sub_batch_metrics[metric_name].append(value)

    def flush_batch_metrics(self, step=None):
        avg_metrics = {metric_name: mean(values) for metric_name, values in self.sub_batch_metrics.items()}
        self.log_metrics(avg_metrics, step=step)
        self.sub_batch_metrics.clear()
        if self.time_last_flush is not None:
            time_since_last_flush = time() - self.time_last_flush
            if self.last_step is not None and step is not None and step > self.last_step:
                # Normalize to get time per sample
                time_since_last_flush = time_since_last_flush / (step - self.last_step)
            self.log_metrics({"time_per_sample": time_since_last_flush}, step)
        self.last_step = step
        self.time_last_flush = time()
        self.metrics_logger.flush()

    def log_time(self, time, name, step):
        self.log_metrics({name: time}, step)


class TensorboardLogger(MetricsLogger):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step, timestep=None, **kwargs):
        self.writer.add_scalar(name, value, global_step=step, walltime=timestep)

    def log_text(self, name: str, text: str, step: int = None, timestep: float = None, **kwargs):
        self.writer.add_text(name, text, global_step=step, walltime=timestep)

    def log_scalars(self, name, values_dict: dict, step, timestep=None, **kwargs):
        self.writer.add_scalars(name, values_dict, global_step=step, walltime=timestep)

    def flush(self):
        self.writer.flush()

    def log_hyperparameters(self, hyperparameters: dict):
        self.writer.add_hparams(hyperparameters, dict())
