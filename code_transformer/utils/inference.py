from code_transformer.modeling.constants import NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.modeling.modelmanager import CodeTransformerModelManager, GreatModelManager, XLNetModelManager
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.ablation import CTCodeSummarizationOnlyASTDataset
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDatasetNoPunctuation, \
    CTCodeSummarizationDataset, CTCodeSummarizationDatasetEdgeTypes
from code_transformer.preprocessing.graph.binning import ExponentialBinning, EqualBinning
from code_transformer.preprocessing.graph.distances import PersonalizedPageRank, ShortestPaths, AncestorShortestPaths, \
    SiblingShortestPaths, DistanceBinning
from code_transformer.preprocessing.graph.transform import DistancesTransformer, TokenDistancesTransform
from code_transformer.preprocessing.nlp.vocab import CodeSummarizationVocabularyTransformer, VocabularyTransformer
from code_transformer.preprocessing.pipeline.stage1 import CTStage1Preprocessor
from code_transformer.preprocessing.pipeline.stage2 import CTStage2Sample
from env import DATA_PATH_STAGE_2


def get_model_manager(model_type):
    if model_type == 'code_transformer':
        return CodeTransformerModelManager()
    elif model_type == 'great':
        return GreatModelManager()
    elif model_type == 'xl_net':
        return XLNetModelManager()
    else:
        raise ValueError(f"Unknown model type {model_type}")


def make_batch_from_sample(stage2_sample: CTStage2Sample, model_config, model_type):
    assert isinstance(stage2_sample.token_mapping, dict), f"Please re-generate the sample"
    data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, model_config['data_setup']['language'],
                                             partition='train', shuffle=True)

    # Setup dataset to generate batch as input for model
    LIMIT_TOKENS = 1000
    token_distances = None
    if TokenDistancesTransform.name in model_config['data_transforms']['relative_distances']:
        num_bins = data_manager.load_config()['num_bins']
        distance_binning_config = model_config['data_transforms']['distance_binning']
        if distance_binning_config['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning_config['growth_factor'])
        else:
            trans_func = EqualBinning()
        token_distances = TokenDistancesTransform(
            DistanceBinning(num_bins, distance_binning_config['n_fixed_bins'], trans_func))

    use_pointer_network = model_config['data_setup']['use_pointer_network']
    if model_type in {'great'}:
        dataset_type = 'great'
    elif 'use_only_ast' in model_config['data_setup'] and model_config['data_setup']['use_only_ast']:
        dataset_type = 'only_ast'
    elif 'use_no_punctuation' in model_config['data_setup'] and model_config['data_setup']['use_no_punctuation']:
        dataset_type = 'no_punctuation'
    else:
        dataset_type = 'regular'

    if dataset_type == 'great':
        dataset = CTCodeSummarizationDatasetEdgeTypes(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                      use_pointer_network=use_pointer_network,
                                                      token_distances=token_distances, max_num_tokens=LIMIT_TOKENS)
    elif dataset_type == 'regular':
        dataset = CTCodeSummarizationDataset(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                             use_pointer_network=use_pointer_network, max_num_tokens=LIMIT_TOKENS,
                                             token_distances=token_distances)
    elif dataset_type == 'no_punctuation':
        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager,
                                                          num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                          use_pointer_network=use_pointer_network,
                                                          max_num_tokens=LIMIT_TOKENS,
                                                          token_distances=token_distances)
    elif dataset_type == 'only_ast':
        dataset = CTCodeSummarizationOnlyASTDataset(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                    use_pointer_network=use_pointer_network,
                                                    max_num_tokens=LIMIT_TOKENS, token_distances=token_distances)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Hijack dataset to only contain user specified code snippet
    dataset.dataset = (stage2_sample for _ in range(1))
    processed_sample = next(dataset)
    batch = dataset.collate_fn([processed_sample])

    return batch


def remove_duplicates(tokens):
    unique_tokens = []
    for t in tokens:
        if t not in unique_tokens:
            unique_tokens.append(t)
    return unique_tokens


def reverse_lookup(token, batch, method_name_vocab):
    token = token if isinstance(token, int) else token.item()
    if 'extended_vocabulary' in batch and batch.extended_vocabulary[0] and token >= len(method_name_vocab):
        for word, token_id in batch.extended_vocabulary[0].items():
            if token_id == token:
                return word
    else:
        return method_name_vocab.reverse_lookup(token)


def decode_predicted_tokens(tokens, batch, data_manager):
    vocabs = data_manager.load_vocabularies()
    if len(vocabs) == 4:
        method_name_vocab = vocabs[-1]
    else:
        method_name_vocab = vocabs[0]

    prediction = remove_duplicates(tokens)
    predicted_method_name = [reverse_lookup(sub_token_id, batch, method_name_vocab) for sub_token_id in prediction if
                             sub_token_id.item() != 3 and sub_token_id.item() != 0]
    return predicted_method_name


def predict_method_name(model, model_config, code_snippet: str, method_name_place_holder='f'):
    language = model_config['data_setup']['language']

    # Build data manager and load vocabularies + configs
    data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language, partition='train', shuffle=True)
    vocabs = data_manager.load_vocabularies()
    if len(vocabs) == 4:
        method_name_vocab = vocabs[-1]
    else:
        method_name_vocab = vocabs[0]
    word_vocab = vocabs[0]
    data_config = data_manager.load_config()

    # Stage 1 Preprocessing (Compute AST)
    lexer_language = 'java' if language in {'java-small', 'java-medium', 'java-large', 'java-small-pretrain',
                                            'java-pretrain'} else language
    preprocessor = CTStage1Preprocessor(lexer_language, allow_empty_methods=True)
    stage1 = preprocessor.process([(method_name_place_holder, "", code_snippet)], 0)

    # Stage 2 Preprocessing (Compute Distance Matrices)
    distances_config = data_config['distances']
    PPR_ALPHA = distances_config['ppr_alpha']
    PPR_USE_LOG = distances_config['ppr_use_log']
    PPR_THRESHOLD = distances_config['ppr_threshold']

    SP_THRESHOLD = distances_config['sp_threshold']

    ANCESTOR_SP_FORWARD = distances_config['ancestor_sp_forward']
    ANCESTOR_SP_BACKWARD = distances_config['ancestor_sp_backward']
    ANCESTOR_SP_NEGATIVE_REVERSE_DISTS = distances_config['ancestor_sp_negative_reverse_dists']
    ANCESTOR_SP_THRESHOLD = distances_config['ancestor_sp_threshold']

    SIBLING_SP_FORWARD = distances_config['sibling_sp_forward']
    SIBLING_SP_BACKWARD = distances_config['sibling_sp_backward']
    SIBLING_SP_NEGATIVE_REVERSE_DISTS = distances_config['sibling_sp_negative_reverse_dists']
    SIBLING_SP_THRESHOLD = distances_config['sibling_sp_threshold']

    binning_config = data_config['binning']
    EXPONENTIAL_BINNING_GROWTH_FACTOR = binning_config['exponential_binning_growth_factor']
    N_FIXED_BINS = binning_config['n_fixed_bins']
    NUM_BINS = binning_config['num_bins']  # How many bins should be calculated for the values in distance matrices

    preprocessing_config = data_config['preprocessing']
    REMOVE_PUNCTUATION = preprocessing_config['remove_punctuation']

    distance_metrics = [
        PersonalizedPageRank(threshold=PPR_THRESHOLD, log=PPR_USE_LOG, alpha=PPR_ALPHA),
        ShortestPaths(threshold=SP_THRESHOLD),
        AncestorShortestPaths(forward=ANCESTOR_SP_FORWARD, backward=ANCESTOR_SP_BACKWARD,
                              negative_reverse_dists=ANCESTOR_SP_NEGATIVE_REVERSE_DISTS,
                              threshold=ANCESTOR_SP_THRESHOLD),
        SiblingShortestPaths(forward=SIBLING_SP_FORWARD, backward=SIBLING_SP_BACKWARD,
                             negative_reverse_dists=SIBLING_SP_NEGATIVE_REVERSE_DISTS,
                             threshold=SIBLING_SP_THRESHOLD)]

    db = DistanceBinning(NUM_BINS, N_FIXED_BINS, ExponentialBinning(EXPONENTIAL_BINNING_GROWTH_FACTOR))

    distances_transformer = DistancesTransformer(distance_metrics, db)
    if len(vocabs) == 4:
        vocabulary_transformer = CodeSummarizationVocabularyTransformer(*vocabs)
    else:
        vocabulary_transformer = VocabularyTransformer(*vocabs)

    stage2 = stage1[0]
    if REMOVE_PUNCTUATION:
        stage2.remove_punctuation()
    stage2 = vocabulary_transformer(stage2)
    stage2 = distances_transformer(stage2)

    # Setup dataset to generate batch as input for model
    LIMIT_TOKENS = 1000
    token_distances = None
    if TokenDistancesTransform.name in model_config['data_transforms']['relative_distances']:
        num_bins = data_manager.load_config()['num_bins']
        distance_binning_config = model_config['data_transforms']['distance_binning']
        if distance_binning_config['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning_config['growth_factor'])
        else:
            trans_func = EqualBinning()
        token_distances = TokenDistancesTransform(
            DistanceBinning(num_bins, distance_binning_config['n_fixed_bins'], trans_func))
    if model_config['data_setup']['use_no_punctuation'] == True:
        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager,
                                                          num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                          use_pointer_network=model_config['data_setup'][
                                                              'use_pointer_network'],
                                                          max_num_tokens=LIMIT_TOKENS,
                                                          token_distances=token_distances)
    else:
        dataset = CTCodeSummarizationDataset(data_manager,
                                             num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                             use_pointer_network=model_config['data_setup']['use_pointer_network'],
                                             max_num_tokens=LIMIT_TOKENS,
                                             token_distances=token_distances)

    # Hijack dataset to only contain user specified code snippet
    dataset.dataset = (stage2 for _ in range(1))
    processed_sample = next(dataset)
    batch = dataset.collate_fn([processed_sample])

    # Obtain model prediction
    output = model.forward_batch(batch)

    return output
