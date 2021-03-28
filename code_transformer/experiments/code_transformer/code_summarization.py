from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.experiments.mixins.code_summarization import CTCodeSummarizationMixin
from code_transformer.experiments.mixins.code_trans_transformer import CodeTransformerDecoderMixin


class CodeTransDecoderExperimentSetup(CodeTransformerDecoderMixin,
                                      CTCodeSummarizationMixin,
                                      ExperimentSetup):
    pass


@ex.automain
def main():
    experiment = CodeTransDecoderExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return CodeTransDecoderExperimentSetup()