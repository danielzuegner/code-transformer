from code_transformer.experiments.experiment import ExperimentSetup


class XLNetExperimentSetup(ExperimentSetup):

    def __init__(self):
        super(XLNetExperimentSetup, self).__init__()

    def _init_data_transforms(self, max_distance_mask=None, relative_distances=[], distance_binning={'type': 'regular', 'n_fixed_bins': 0}):
        super(XLNetExperimentSetup, self)._init_data_transforms(max_distance_mask, relative_distances, distance_binning)