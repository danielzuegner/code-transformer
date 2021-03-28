from abc import ABC

from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDatasetEdgeTypes
from code_transformer.preprocessing.graph.distances import DistanceBinning
from code_transformer.preprocessing.graph.transform import TokenDistancesTransform
from env import DATA_PATH_STAGE_2


class CTCodeSummarizationGreatMixin(ExperimentSetup, ABC):

    @ex.capture(prefix="data_setup")
    def _init_data(self, language, use_validation=False, mini_dataset=False,
                   num_sub_tokens=5, num_subtokens_output=5, use_only_ast=False, sort_by_length=False,
                   shuffle=True, use_pointer_network=False):

        self.num_sub_tokens = num_sub_tokens
        self.use_validation = use_validation
        self.use_pointer_network = use_pointer_network
        self.data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2, language, shuffle=shuffle,
                                                  infinite_loading=True,
                                                  mini_dataset=mini_dataset, sort_by_length=sort_by_length)
        self.word_vocab, self.token_type_vocab, self.node_type_vocab = self.data_manager.load_vocabularies()

        if ',' in language:
            self.num_languages = len(language.split(','))

        self.use_separate_vocab = False

        token_distances = None
        if TokenDistancesTransform.name in self.relative_distances:
            token_distances = TokenDistancesTransform(DistanceBinning(self.data_manager.load_config()['binning']['num_bins'],
                                                                      self.distance_binning['n_fixed_bins'],
                                                                      self.distance_binning['trans_func']
                                                                      ))

        self.dataset_train = CTCodeSummarizationDatasetEdgeTypes(self.data_manager,
                                                                 token_distances=token_distances,
                                                                 max_distance_mask=self.max_distance_mask,
                                                                 num_sub_tokens=num_sub_tokens,
                                                                 num_sub_tokens_output=num_subtokens_output,
                                                                 use_pointer_network=use_pointer_network)

        if self.use_validation:
            data_manager_validation = CTBufferedDataManager(DATA_PATH_STAGE_2, language, partition="valid",
                                                            shuffle=True, infinite_loading=True,
                                                            mini_dataset=mini_dataset)
            self.dataset_validation = CTCodeSummarizationDatasetEdgeTypes(data_manager_validation,
                                                                          token_distances=token_distances,
                                                                          max_distance_mask=self.max_distance_mask,
                                                                          num_sub_tokens=num_sub_tokens,
                                                                          num_sub_tokens_output=num_subtokens_output,
                                                                          use_pointer_network=use_pointer_network)

        self.dataset_validation_creator = \
            lambda infinite_loading: self._create_validation_dataset(DATA_PATH_STAGE_2, language, token_distances,
                                                                     num_sub_tokens, num_subtokens_output,
                                                                     infinite_loading=infinite_loading,
                                                                     use_pointer_network=use_pointer_network,
                                                                     filter_language=None, dataset_imbalance=None)

    def _create_validation_dataset(self, data_location, language, token_distances, num_sub_tokens,
                                   num_subtokens_output, infinite_loading, use_pointer_network, filter_language,
                                   dataset_imbalance):
        data_manager_validation = CTBufferedDataManager(data_location, language, partition="valid",
                                                        shuffle=True, infinite_loading=infinite_loading)
        dataset_validation = CTCodeSummarizationDatasetEdgeTypes(data_manager_validation,
                                                                 token_distances=token_distances,
                                                                 max_distance_mask=self.max_distance_mask,
                                                                 num_sub_tokens=num_sub_tokens,
                                                                 num_sub_tokens_output=num_subtokens_output,
                                                                 use_pointer_network=use_pointer_network)

        return dataset_validation
