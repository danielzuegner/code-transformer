from code_transformer.experiments.experiment import ex
from code_transformer.experiments.mixins.code_summarization import CTCodeSummarizationMixin
from code_transformer.experiments.mixins.xl_net_transformer import XLNetTransformerMixin


class XLNetCodeSummarizationTransformerExperimentSetup(CTCodeSummarizationMixin, XLNetTransformerMixin):
    pass


@ex.automain
def main():
    experiment = XLNetCodeSummarizationTransformerExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return XLNetCodeSummarizationTransformerExperimentSetup()