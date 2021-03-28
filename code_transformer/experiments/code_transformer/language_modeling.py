from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.experiments.mixins.code_trans_transformer import CodeTransformerDecoderMixin
from code_transformer.modeling.modelmanager import CodeTransformerLMModelManager


class CodeTransLanguageModelingTransformerExperimentSetup(CodeTransformerDecoderMixin,
                                                          ExperimentSetup):

    def __init__(self):
        super(CodeTransLanguageModelingTransformerExperimentSetup, self).__init__()
        self.model_manager = CodeTransformerLMModelManager()
        # Overwrite model manager to ensure saving language modeling experiments in a different folder even if
        # they share the exact same architecture as models trained for code summarization


@ex.automain
def main():
    experiment = CodeTransLanguageModelingTransformerExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return CodeTransLanguageModelingTransformerExperimentSetup()
