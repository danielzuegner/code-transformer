"""
Evaluates a stored snapshot of a multi-lingual model on the valid or test partition.
Performance is measured in micro-F1 score.
Usage: python -m scripts.evaluate-multilanguage {model} {run_id} {snapshot_iteration} {partition} [--filter-language {language}]
The dataset to run the evaluation is inferred from the trained model. The --filter-language option is only necessary
for models that were first pre-trained on a multilanguage dataset and then fine-tuned on a single language.
"""

from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_transformer.modeling.constants import PAD_TOKEN, UNKNOWN_TOKEN, NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.modeling.modelmanager import CodeTransformerModelManager, \
    XLNetModelManager, GreatModelManager
from code_transformer.preprocessing.datamanager.base import batch_to_device
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.ablation import CTCodeSummarizationOnlyASTDataset
from code_transformer.preprocessing.dataset.code_summarization import \
    CTCodeSummarizationDatasetNoPunctuation, CTCodeSummarizationDatasetEdgeTypes
from code_transformer.utils.metrics import compute_rouge, micro_precision, micro_recall, micro_f1_score, \
    get_best_non_unk_predictions
from env import DATA_PATH_STAGE_2

parser = ArgumentParser()
parser.add_argument("model",
                    choices=['code_transformer', 'xl_net', 'great'])
parser.add_argument("run_id", type=str)
parser.add_argument("snapshot_iteration", type=str)
parser.add_argument("partition", type=str, choices=['train', 'valid', 'test'], default='valid')
parser.add_argument('--no-gpu', action='store_true', default=False)
parser.add_argument('--filter-language', default=None, const=None, nargs='?')
args = parser.parse_args()

BATCH_SIZE = 4
LIMIT_TOKENS = 1000  # MAX_NUM_TOKENS


def format_scores(scores: dict):
    return f"\tF: {scores['f'] * 100:0.2f}\n" \
           f"\tPrec: {scores['p'] * 100:0.2f}\n" \
           f"\tRec: {scores['r'] * 100:0.2f}\n"


def print_results():
    for lang_id, language_name in enumerate(language_mapping):
        if predictions[lang_id] and (not args.filter_language or language_mapping[lang_id] == args.filter_language):
            lab = torch.stack(labels[lang_id])

            scores = compute_rouge(predictions[lang_id], lab, pad_id=word_vocab[PAD_TOKEN],
                                   predictions_provided=True)

            pred = torch.stack(best_non_unk_predictions[lang_id])

            micro_prec = micro_precision(pred, lab, predictions_provided=True,
                                         pad_id=word_vocab[PAD_TOKEN], unk_id=word_vocab[UNKNOWN_TOKEN])
            micro_rec = micro_recall(pred, lab, predictions_provided=True,
                                     pad_id=word_vocab[PAD_TOKEN], unk_id=word_vocab[UNKNOWN_TOKEN])
            micro_f1 = micro_f1_score(pred, lab, predictions_provided=True,
                                      pad_id=word_vocab[PAD_TOKEN], unk_id=word_vocab[UNKNOWN_TOKEN])

            print()
            print('-----------------')
            print(f"{language_name}:")
            print('-----------------')
            print(
                f"F1: \n{format_scores(scores['rouge-1'])}"
                f"Rouge-2: \n{format_scores(scores['rouge-2'])}"
                f"Rouge-L: \n{format_scores(scores['rouge-l'])}"
                f"micro-F1: \n"
                f"\tF1: {micro_f1 * 100:0.2f}\n"
                f"\tPrec: {micro_prec * 100:0.2f}\n"
                f"\tRec: {micro_rec * 100:0.2f}\n"
            )


if __name__ == '__main__':

    if args.model == 'code_transformer':
        model_manager = CodeTransformerModelManager()
    elif args.model == 'great':
        model_manager = GreatModelManager()
    elif args.model == 'xl_net':
        model_manager = XLNetModelManager()
    else:
        raise ValueError(f"Unknown model type `{args.model}`")

    config = model_manager.load_config(args.run_id)

    data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2,
                                         config['data_setup']['language'],
                                         partition=args.partition,
                                         shuffle=False,
                                         filter_language=args.filter_language)

    model = model_manager.load_model(args.run_id, args.snapshot_iteration, gpu=not args.no_gpu)

    vocabularies = data_manager.load_vocabularies()
    if len(vocabularies) == 3:
        word_vocab, _, _ = vocabularies
    else:
        word_vocab, _, _, _ = vocabularies
    language_mapping = config['data_setup']['language'].split(',')

    model = model.eval()
    if not args.no_gpu:
        model = model.cuda()

    if not args.no_gpu:
        model = model.cuda()

    use_pointer_network = config['data_setup']['use_pointer_network']
    if args.model in {'great'}:
        dataset_type = 'great'
    elif 'use_only_ast' in config['data_setup'] and config['data_setup']['use_only_ast']:
        dataset_type = 'only_ast'
    elif 'use_no_punctuation' in config['data_setup'] and config['data_setup']['use_no_punctuation']:
        dataset_type = 'no_punctuation'
    else:
        dataset_type = 'regular'

    # Load data
    if dataset_type == 'great':
        dataset = CTCodeSummarizationDatasetEdgeTypes(data_manager,
                                                      num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                      use_pointer_network=use_pointer_network,
                                                      max_num_tokens=LIMIT_TOKENS)
    elif dataset_type == 'no_punctuation':
        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager,
                                                          num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                          use_pointer_network=use_pointer_network,
                                                          max_num_tokens=LIMIT_TOKENS)
    elif dataset_type == 'only_ast':
        dataset = CTCodeSummarizationOnlyASTDataset(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                    use_pointer_network=use_pointer_network,
                                                    max_num_tokens=LIMIT_TOKENS)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=BATCH_SIZE)

    print('Model:', args.run_id)
    print('Snapshot:', args.snapshot_iteration)
    print('Partition:', args.partition)
    print('Language:', config['data_setup']['language'])
    print('Filter Language:', args.filter_language)
    print('GPU:', not args.no_gpu)
    print('use_pointer_network: ', use_pointer_network)
    print('dataset_type: ', dataset_type)

    predictions = defaultdict(list)
    labels = defaultdict(list)
    best_non_unk_predictions = defaultdict(list)

    # Generate predictions
    for i, batch in tqdm(enumerate(dataloader), total=int(data_manager.approximate_total_samples() / BATCH_SIZE)):
        if not args.no_gpu:
            batch = batch_to_device(batch)
        with torch.no_grad():
            output = model.forward_batch(batch)

        batch_logits = output.logits.detach().cpu().squeeze()
        batch_predictions = output.logits.detach().argmax(-1).squeeze().cpu()
        batch_non_unk_predictions = get_best_non_unk_predictions(output.logits.detach().cpu(),
                                                                 unk_id=word_vocab[UNKNOWN_TOKEN])
        batch_labels = batch.labels.detach().cpu().squeeze()
        if len(batch_predictions.shape) == 1:
            # only one sample in the batch
            batch_predictions = batch_predictions.unsqueeze(0)
            batch_labels = batch_labels.unsqueeze(0)
        for prediction, non_unk_prediction, label, lang_id in zip(batch_predictions,
                                                                  batch_non_unk_predictions,
                                                                  batch_labels,
                                                                  batch.languages.cpu()):
            if len(prediction.shape) == 0:
                print(batch_predictions, batch.labels.detach().cpu().squeeze(), batch.languages.cpu())
            lang_id = lang_id.item()
            predictions[lang_id].append(prediction)
            labels[lang_id].append(label)
            best_non_unk_predictions[lang_id].append(non_unk_prediction)

    data_manager.shutdown()

    # Evaluate predictions
    print()
    print('==============')
    print('Final results:')
    print('==============')
    print_results()

    for lang_id in predictions.keys():
        if predictions[lang_id]:
            predictions[lang_id] = torch.stack(predictions[lang_id])
            labels[lang_id] = torch.stack(labels[lang_id])
            best_non_unk_predictions[lang_id] = torch.stack(best_non_unk_predictions[lang_id])

    print()
    print(
        f"Storing predictions into {model_manager._snapshot_location(args.run_id)}/predictions-snapshot-{args.snapshot_iteration}.p")
    model_manager.save_artifact(args.run_id,
                                (predictions, best_non_unk_predictions, labels),
                                'predictions',
                                snapshot_iteration=args.snapshot_iteration)
    data_manager.shutdown()
