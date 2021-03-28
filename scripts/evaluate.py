"""
Evaluates a stored snapshot on the valid or test partition. Performance is measured in micro-F1 score.
Usage: python -m scripts.evaluate {model} {run_id} {snapshot_iteration} {partition}
The actual language is inferred from the trained model.
"""

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_transformer.modeling.constants import PAD_TOKEN, UNKNOWN_TOKEN, NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.modeling.modelmanager import GreatModelManager, XLNetModelManager
from code_transformer.modeling.modelmanager.code_transformer import CodeTransformerModelManager
from code_transformer.preprocessing.datamanager.base import batch_to_device, batch_filter_distances
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.ablation import CTCodeSummarizationOnlyASTDataset
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDataset, \
    CTCodeSummarizationDatasetEdgeTypes, CTCodeSummarizationDatasetNoPunctuation
from code_transformer.preprocessing.graph.binning import ExponentialBinning, EqualBinning
from code_transformer.preprocessing.graph.distances import DistanceBinning
from code_transformer.preprocessing.graph.transform import TokenDistancesTransform
from code_transformer.utils.metrics import f1_score, compute_rouge, micro_precision, micro_recall, micro_f1_score, \
    get_best_non_unk_predictions
from env import DATA_PATH_STAGE_2

parser = ArgumentParser()
parser.add_argument("model",
                    choices=['code_transformer', 'xl_net', 'great'])
parser.add_argument("run_id", type=str)
parser.add_argument("snapshot_iteration", type=str)
parser.add_argument("partition", type=str, choices=['train', 'valid', 'test'], default='valid')
parser.add_argument("--no-gpu", action='store_true', default=False)
args = parser.parse_args()

BATCH_SIZE = 8
LIMIT_TOKENS = 1000  # MAX_NUM_TOKENS


def format_scores(scores: dict):
    return f"\tF: {scores['f'] * 100:0.2f}\n" \
           f"\tPrec: {scores['p'] * 100:0.2f}\n" \
           f"\tRec: {scores['r'] * 100:0.2f}\n"


if __name__ == '__main__':

    if args.model == 'code_transformer':
        model_manager = CodeTransformerModelManager()
    elif args.model == 'great':
        model_manager = GreatModelManager()
    elif args.model == 'xl_net':
        model_manager = XLNetModelManager()
    else:
        raise ValueError(f"Unknown model type `{args.model}`")

    model = model_manager.load_model(args.run_id, args.snapshot_iteration, gpu=not args.no_gpu)
    model = model.eval()
    if not args.no_gpu:
        model = model.cuda()

    config = model_manager.load_config(args.run_id)
    data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2,
                                         config['data_setup']['language'],
                                         partition=args.partition,
                                         shuffle=False)
    vocabularies = data_manager.load_vocabularies()
    if len(vocabularies) == 3:
        word_vocab, _, _ = vocabularies
    else:
        word_vocab, _, _, _ = vocabularies

    token_distances = None
    if TokenDistancesTransform.name in config['data_transforms']['relative_distances']:
        num_bins = data_manager.load_config()['binning']['num_bins']
        distance_binning_config = config['data_transforms']['distance_binning']
        if distance_binning_config['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning_config['growth_factor'])
        else:
            trans_func = EqualBinning()
        token_distances = TokenDistancesTransform(
            DistanceBinning(num_bins, distance_binning_config['n_fixed_bins'], trans_func))

    use_pointer_network = config['data_setup']['use_pointer_network']
    if args.model in {'great'}:
        dataset_type = 'great'
    elif 'use_only_ast' in config['data_setup'] and config['data_setup']['use_only_ast']:
        dataset_type = 'only_ast'
    elif 'use_no_punctuation' in config['data_setup'] and config['data_setup']['use_no_punctuation']:
        dataset_type = 'no_punctuation'
    else:
        dataset_type = 'regular'

    print(
        f"Evaluating model snapshot-{args.snapshot_iteration} from run {args.run_id} on {config['data_setup']['language']} partition {args.partition}")
    print(f"gpu: {not args.no_gpu}")
    print(f"dataset_type: {dataset_type}")
    print(f"model: {args.model}")
    print(f"use_pointer_network: {use_pointer_network}")

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

    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=BATCH_SIZE)

    relative_distances = config['data_transforms']['relative_distances']

    pad_id = word_vocab[PAD_TOKEN]
    unk_id = word_vocab[UNKNOWN_TOKEN]

    f1_scores = []
    precisions = []
    recalls = []
    predictions = []
    best_non_unk_predictions = []
    labels = []
    losses = []
    progress = tqdm(enumerate(dataloader), total=int(data_manager.approximate_total_samples() / BATCH_SIZE))
    for i, batch in progress:
        batch = batch_filter_distances(batch, relative_distances)
        if not args.no_gpu:
            batch = batch_to_device(batch)

        label = batch.labels.detach().cpu()

        with torch.no_grad():
            output = model.forward_batch(batch).cpu()
        losses.append(output.loss.item())
        f1, prec, rec = f1_score(output.logits, label, pad_id=pad_id, unk_id=unk_id,
                                 output_precision_recall=True)
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)

        batch_logits = output.logits.detach().cpu()
        best_non_unk_predictions.extend(get_best_non_unk_predictions(output.logits, unk_id=unk_id))
        predictions.extend(batch_logits.argmax(-1).squeeze(1))
        labels.extend(label.squeeze(1))

        progress.set_description()
        del batch

    data_manager.shutdown()

    predictions = torch.stack(predictions)
    labels = torch.stack(labels)
    pred = torch.cat(best_non_unk_predictions)

    micro_prec = micro_precision(pred, labels, predictions_provided=True,
                                 pad_id=pad_id, unk_id=unk_id)
    micro_rec = micro_recall(pred, labels, predictions_provided=True,
                             pad_id=pad_id, unk_id=unk_id)
    micro_f1 = micro_f1_score(pred, labels, predictions_provided=True,
                              pad_id=pad_id, unk_id=unk_id)

    scores = compute_rouge(predictions, labels, pad_id=pad_id, predictions_provided=True)

    print()
    print('==============')
    print('Final results:')
    print('==============')
    print(
        f"F1: \n{format_scores(scores['rouge-1'])}"
        f"Rouge-2: \n{format_scores(scores['rouge-2'])}"
        f"Rouge-L: \n{format_scores(scores['rouge-l'])}"
    )

    print(
        f"micro-F1: {micro_f1 * 100:0.2f} (micro-precision: {micro_prec * 100: 0.2f}, micro-recall: {micro_rec * 100:0.2f})")

    print()
    print(
        f"Storing predictions into {model_manager._snapshot_location(args.run_id)}/predictions-snapshot-{args.snapshot_iteration}.p")
    model_manager.save_artifact(args.run_id,
                                (predictions, best_non_unk_predictions, labels),
                                'predictions',
                                snapshot_iteration=args.snapshot_iteration)
