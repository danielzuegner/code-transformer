import io
import os
from abc import abstractmethod
from pathlib import Path

import torch

from code_transformer.configuration.configuration_utils import ModelConfiguration
from code_transformer.utils.io import create_directories, load_pickled, load_json, save_json, save_pickled
from code_transformer.utils.log import list_file_numbering, generate_run_name
from code_transformer.utils.loss import LabelSmoothingLoss


class ModelManager:
    """
    Generic manager that provides utility methods for easy loading/saving of model configs and snapshots.
    """

    def __init__(self, model_store_location, model_type, run_prefix):
        self.model_store_location = model_store_location
        self.model_type = model_type
        self.run_prefix = run_prefix

    def save_snapshot(self, run_id, model_params, iteration):
        snapshot_path = f"{self._snapshot_location(run_id)}/snapshot-{iteration}.p"
        create_directories(snapshot_path)
        with open(snapshot_path, 'wb') as f:
            torch.save(model_params, f)

    def delete_snapshot(self, run_id, snapshot_iteration):
        snapshot_path = self._snapshot_location(run_id)
        os.remove(f"{snapshot_path}/snapshot-{snapshot_iteration}.p")

    def load_config(self, run_id):
        return load_json(f"{self._snapshot_location(run_id)}/config")

    def save_config(self, run_id, config: ModelConfiguration):
        save_json(config, f"{self._snapshot_location(run_id)}/config")

    def save_artifact(self, run_id, artifact, name, snapshot_iteration=None):
        if snapshot_iteration is not None:
            name = f"{name}-snapshot-{snapshot_iteration}"
        save_pickled(artifact, f"{self._snapshot_location(run_id)}/{name}")

    def load_artifact(self, run_id, name, snapshot_iteration=None):
        if snapshot_iteration is not None:
            name = f"{name}-snapshot-{snapshot_iteration}"
        load_pickled(f"{self._snapshot_location(run_id)}/{name}")

    def load_parameters(self, run_id, snapshot_iteration, gpu=True):
        if gpu:
            map_location = None
        else:
            map_location = torch.device('cpu')

        if snapshot_iteration == 'latest':
            list_of_paths = Path(self._snapshot_location(run_id)).glob('snapshot-*.p')
            snapshot_path = max(list_of_paths, key=os.path.getctime)
        else:
            snapshot_path = f"{self._snapshot_location(run_id)}/snapshot-{snapshot_iteration}.p"

        try:
            # Legacy: model snapshots that were migrated from MongoDB are stored as pickle files, while newer
            # runs are directly persisted by torch.save()
            model_params = load_pickled(snapshot_path)
            return torch.load(io.BytesIO(model_params), map_location=map_location)
        except TypeError:
            return torch.load(snapshot_path, map_location=map_location)

    @abstractmethod
    def load_model(self, run_id, snapshot_iteration, **kwargs):
        pass

    def get_available_runs(self):
        runs_dir = Path(f"{self.model_store_location}/{self.model_type}")
        return [run_dir.stem for run_dir in runs_dir.iterdir()]

    def get_available_snapshots(self, run_id):
        return list_file_numbering(self._snapshot_location(run_id), 'snapshot', '.p')

    def generate_run_name(self):
        return generate_run_name(f"{self.model_store_location}/{self.model_type}", self.run_prefix)

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _snapshot_location(self, run_id):
        return f"{self.model_store_location}/{self.model_type}/{self._run_id(run_id)}"

    def _prepare_model_config(self, config):
        config = config['model'].copy()
        loss_fct = LabelSmoothingLoss(config['label_smoothing'])
        config['loss_fct'] = loss_fct
        del config['label_smoothing']
        del config['with_cuda']
        return config

    def _run_id(self, run_id):
        if isinstance(run_id, int):
            return f"{self.run_prefix}-{run_id}"
        else:
            return run_id
