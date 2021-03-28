from code_transformer.experiments.experiment import ex
from code_transformer.experiments.mixins.xl_net_lm import XLNetLanguageModelingMixin


class XLNetLanguageModelingExperimentSetup(XLNetLanguageModelingMixin):
    pass


@ex.automain
def main():
    experiment = XLNetLanguageModelingExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return XLNetLanguageModelingExperimentSetup()
