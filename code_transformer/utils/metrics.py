from statistics import mean
from typing import Union

import torch
from rouge import Rouge


def get_best_non_unk_predictions(logits: torch.Tensor, unk_id):
    """
    For every sample, returns the prediction with the highest score. If this is the <unk> token, then the prediction
    with the second highest score will be returned instead
    :param logits: batch_size x num_predict x num_subtokens x vocab_size
    :param unk_id: ID of the <unk> token
    """
    top_2_predictions = logits.topk(2, -1).indices
    best_non_unk_predictions = top_2_predictions[:, :, :, 0]
    idx_first_unk = best_non_unk_predictions == unk_id
    best_non_unk_predictions[idx_first_unk] = top_2_predictions[:, :, :, 1][idx_first_unk]
    return best_non_unk_predictions


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor, unk_id=None, pad_id=None, predictions_provided=False):
    """
    Averaged per sample:
        1, if the prediction with highest confidence was the correct one for all sub tokens
        0, else
    if `unk_id` is given, the best non <unk> prediction will be used and <unk> tokens in the labels will be
    completely ignored, i.e., it does not matter what the prediction at this position was.
    """
    assert unk_id is None or pad_id is not None, "When unk_id is given, pad_id must be given as well"
    if unk_id is None:
        predictions = logits if predictions_provided else logits.argmax(-1)
        correct = predictions == labels
        all_correct = correct.prod(-1)
    else:
        predictions = logits if predictions_provided else get_best_non_unk_predictions(logits, unk_id)
        idx_unk_labels = (labels == unk_id)
        correct = (predictions == labels)
        correct[idx_unk_labels] = True
        all_correct = correct.prod(-1)
        idx_label_all_unk = ((labels == unk_id) | (labels == pad_id)).all(-1) & ~(labels == pad_id).all(-1)
        all_correct = all_correct[~idx_label_all_unk]  # Completely ignore if labels only
        # consists of <unk>

    top1 = all_correct.float().mean().item()
    return top1


def topk_accuracy(k, logits: torch.Tensor, labels: torch.Tensor, unk_id=None, pad_id=None):
    """
    Averaged per sample:
        1, if one of the k predictions with highest confidence was the correct one for all sub tokens
        0, else
    if `unk_id` is given, the best non <unk> prediction will be used and <unk> tokens in the labels will be
    completely ignored, i.e., it does not matter what the 5 predictions with highest score at this position were.
    :param logits: batch_size x num_predict x num_sub_tokens x vocab_size
    :param labels: batch_size x num_predict x num_sub_tokens
    :param k: The k highest logits will be considered as predictions and for every sub token when any of the k
    highest valued predictions was the correct one, it will count as an accurate prediction
    :param unk_id: ID of the <unk> token
    """
    assert unk_id is None or pad_id is not None, "When unk_id is given, pad_id must be given as well"
    topk_pred = logits.topk(k, -1)
    # Accept if any of the top k predictions is the label
    topk_correct = (topk_pred.indices == labels.unsqueeze(-1).expand((-1, -1, -1, k)))
    topk_correct = topk_correct.any(-1)
    if unk_id is None:
        topk_correct = topk_correct.all(-1)  # Only accept if for all sub tokens a top k prediction was correct
    if unk_id is not None:
        idx_unk_labels = (labels == unk_id)
        topk_correct[idx_unk_labels] = True
        topk_correct = topk_correct.all(-1)  # Only accept if for all sub tokens a top k prediction was correct
        idx_label_all_unk = ((labels == unk_id) | (labels == pad_id)).all(-1) & ~(labels == pad_id).all(-1)
        topk_correct = topk_correct[~idx_label_all_unk]  # Completely ignore if labels only
        # consists of <unk>

    return topk_correct.float().mean().item()


def non_trivial_words_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_id, unk_id=None,
                               predictions_provided=False):
    """
    Averaged per sample:
        1, if the prediction with highest confidence was the correct one for all sub tokens and the label was non
        trivial (i.e., >= 2 non-padding sub tokens)
        0, else
    if `unk_id` is given, the best non <unk> prediction will be used for each sub token and <unk> tokens in the labels
    will be completely ignored, i.e., it does not matter what the prediction at this position was.
    """
    if predictions_provided:
        predictions = logits
    else:
        predictions = logits.argmax(-1) if unk_id is None else get_best_non_unk_predictions(logits, unk_id)

    pad_mask = (labels == pad_id).logical_not()
    non_pad_per_label = pad_mask.sum(-1)

    idx_non_trivial_labels = non_pad_per_label >= 2
    accurate_predictions = (predictions == labels)
    if unk_id is not None:
        idx_label_all_unk = ((labels == unk_id) | (labels == pad_id)).all(-1) & ~(labels == pad_id).all(-1)
        idx_non_trivial_labels &= ~idx_label_all_unk
        accurate_predictions[labels == unk_id] = True

    if (~idx_non_trivial_labels).all():
        # All labels are trivial
        return None

    accurate_predictions = accurate_predictions[idx_non_trivial_labels].all(-1)
    return accurate_predictions.float().mean().item()


def precision(logits: torch.Tensor, labels: torch.Tensor, pad_id=None, unk_id=None, predictions_provided=False,
              average=True):
    """
    Calculates for every token how many of the predicted sub tokens were correct (are in the corresponding label)
    and averages that over all samples.
    :param logits: batch_size x num_predict x num_sub_tokens x vocab_size
    :param labels: batch_size x num_predict x num_sub_tokens
    :param pad_id: ID of the [PAD] token. Necessary, as we don't want to count when a prediction is only [PAD]
    :param unk_id: ID of the <unk> token. if given, the best non <unk> prediction per subtoken will be used and <unk>
    tokens in the labels will be ignored, i.e., it does not matter what the prediction at this position was.
    """

    if predictions_provided:
        predictions = logits
        num_sub_tokens = logits.shape[1]
    else:
        num_sub_tokens = logits.shape[2]
        predictions = logits.argmax(-1) if unk_id is None else get_best_non_unk_predictions(logits, unk_id)

    # Duplicate every subtoken db -> [db, db, db, db, db]
    predictions_expanded = predictions.unsqueeze(-1).expand((-1, -1, -1, num_sub_tokens))
    # Duplicate labels: [create, db, conn, [PAD], [PAD]] ->
    #   [[create, db, conn, [PAD], [PAD]]
    #    [create, db, conn, [PAD], [PAD]]
    #    [create, db, conn, [PAD], [PAD]]
    #    [create, db, conn, [PAD], [PAD]]
    #    [create, db, conn, [PAD], [PAD]]]
    labels_expanded = labels.unsqueeze(-2).expand((-1, -1, num_sub_tokens, -1))
    precise_predictions = (predictions_expanded == labels_expanded).any(-1)

    if pad_id is None:
        if unk_id is None:
            precision_per_token = precise_predictions.sum(-1).float() / num_sub_tokens
        else:
            precision_per_token = precise_predictions[~(labels == unk_id).all()].sum(-1).float() / num_sub_tokens
    else:
        pad_mask = (predictions == pad_id).logical_not()

        precision_per_token = torch.einsum("bpt,bpt->bp", precise_predictions.int(), pad_mask.int())
        non_pad_per_token = pad_mask.sum(-1)

        idx_label_all_pad = (labels == pad_id).all(-1)
        idx_prediction_all_pad = (~pad_mask).all(-1)

        # If only [PAD] was predicted, but the label was not all [PAD] this prediction has 0 precision
        precision_per_token[idx_prediction_all_pad & ~ idx_label_all_pad] = 0

        # If label and prediction were all [PAD] then precision is 1
        precision_per_token[idx_prediction_all_pad & idx_label_all_pad] = 1
        non_pad_per_token[idx_prediction_all_pad] = 1

        if unk_id is not None:
            # Completely ignore precision when label only consist of unk
            idx_label_all_unk = ((labels == unk_id) | (labels == pad_id)).all(-1) & ~idx_label_all_pad
            precision_per_token = precision_per_token[~idx_label_all_unk]
            non_pad_per_token = non_pad_per_token[~idx_label_all_unk]

        # Precision is number of correct tokens / number of predicted tokens
        precision_per_token = precision_per_token.float() / non_pad_per_token

    if average:
        return precision_per_token.mean().item()
    else:
        return precision_per_token


def recall(logits: torch.Tensor, labels: torch.Tensor, pad_id=None, unk_id=None, predictions_provided=False,
           average=True):
    if predictions_provided:
        num_sub_tokens = logits.shape[1]
        predictions = logits
    else:
        num_sub_tokens = logits.shape[2]
        predictions = logits.argmax(-1) if unk_id is None else get_best_non_unk_predictions(logits, unk_id)

    # Duplicate every label db -> [db, db, db, db, db]
    labels_expanded = labels.unsqueeze(-1).expand((-1, -1, -1, num_sub_tokens))
    predictions_expanded = predictions.unsqueeze(-2).expand((-1, -1, num_sub_tokens, -1))
    recall_predictions = (predictions_expanded == labels_expanded).any(-1)

    if unk_id is not None:
        idx_non_unk_labels = ~(labels == unk_id)
        num_sub_tokens = idx_non_unk_labels.sum(-1)

    if pad_id is None:
        # Completely ignore a prediction if the label was all unk_id (num_sub_tokens == 0)
        recall_per_token = recall_predictions[num_sub_tokens > 0].sum(-1).float() / num_sub_tokens[num_sub_tokens > 0]
    else:
        pad_mask = (labels == pad_id).logical_not()
        if unk_id is not None:
            # Do not add up to sub token count if label sub token was <unk>
            recall_per_token = torch.einsum("bpt,bpt->bp", recall_predictions.int(),
                                            (pad_mask & idx_non_unk_labels).int())
            non_pad_per_label = (pad_mask & idx_non_unk_labels).sum(-1)

        else:
            recall_per_token = torch.einsum("bpt,bpt->bp", recall_predictions.int(), pad_mask.int())
            non_pad_per_label = pad_mask.sum(-1)

        idx_prediction_all_pad = (predictions == pad_id).all(-1)
        idx_label_all_pad = (labels == pad_id).all(-1)

        # If label is all [PAD] accept prediction all [PAD] with Recall 1
        recall_per_token[idx_label_all_pad & idx_prediction_all_pad] = 1
        non_pad_per_label[idx_label_all_pad] = 1

        if unk_id is not None:
            # Completely ignore precision when label only consist of unk
            idx_label_all_unk = ((labels == unk_id) | (labels == pad_id)).all(-1) & ~idx_label_all_pad
            recall_per_token = recall_per_token[~idx_label_all_unk]
            non_pad_per_label = non_pad_per_label[~idx_label_all_unk]

        recall_per_token = recall_per_token.float() / non_pad_per_label

    if average:
        return recall_per_token.mean().item()
    else:
        return recall_per_token


def f1_score(logits, labels, pad_id=None, unk_id=None, predictions_provided=False, output_precision_recall=False):
    if predictions_provided:
        predictions = logits
    else:
        predictions = logits.argmax(-1) if unk_id is None else get_best_non_unk_predictions(logits, unk_id)
    scores = [get_micro_precision_recall_f1(pred, label, unk_id=unk_id, pad_id=pad_id, predictions_provided=True)
              for pred, label in zip(predictions, labels)]

    precisions, recalls, f1s = zip(*scores)
    if output_precision_recall:
        return mean(f1s), mean(precisions), mean(recalls)
    else:
        return mean(f1s)


def filter_impossible_names(tokens, impossible_names):
    return [t for t in tokens if t not in impossible_names]


def calculate_metrics(predictions, labels, pad_id=None, unk_id=None):
    impossible_names = []
    if pad_id is not None:
        impossible_names.append(pad_id)
    if unk_id is not None:
        impossible_names.append(unk_id)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    if isinstance(predictions, torch.Tensor) and len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    if isinstance(labels, torch.Tensor) and len(labels.shape) == 1:
        labels = labels.unsqueeze(0)
    for predicted, label in zip(predictions, labels):
        predicted = [t.item() if isinstance(t, torch.Tensor) else t for t in predicted]
        label = [t.item() if isinstance(t, torch.Tensor) else t for t in label]
        filtered_predicted_names = filter_impossible_names(predicted, impossible_names)
        filtered_original_subtokens = filter_impossible_names(label, impossible_names)

        if all([st == pad_id for st in predicted]) and all([st == pad_id for st in label]):
            # Edge case in JavaScript: empty method names are allowed which are represented as only [PAD] tokens
            # If prediction was correct, it counts as 1 true positive.
            # Currently, we drop all code snippets with anonymous functions since the AST leaks the method name due to
            # the missing function name node in such cases (see CTCodeSummarizationDataset).
            # Hence, this if statement could be removed. However, we leave it as a future reminder in case the
            # preprocessing is updated to account for anonymous functions (e.g., by inserting a fake method name
            # into the code snippet sequence before computing the AST).
            true_positive += 1
            continue
        if filtered_original_subtokens == filtered_predicted_names:
            true_positive += len(filtered_original_subtokens)
            continue

        for subtok in filtered_predicted_names:
            if subtok in filtered_original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in filtered_original_subtokens:
            if not subtok in filtered_predicted_names:
                false_negative += 1

    return true_positive, false_positive, false_negative


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def get_micro_precision_recall_f1(logits: torch.Tensor, labels: torch.Tensor, pad_id=None, unk_id=None,
                                  predictions_provided=False):
    if predictions_provided:
        predictions = logits
    else:
        predictions = logits.argmax(-1) if unk_id is None else get_best_non_unk_predictions(logits, unk_id)
    tp, fp, fn = calculate_metrics(
        predictions.squeeze() if isinstance(predictions, torch.Tensor) else predictions,
        labels.squeeze() if isinstance(labels, torch.Tensor) else labels, pad_id=pad_id, unk_id=unk_id)
    return calculate_results(tp, fp, fn)


def micro_f1_score(logits: torch.Tensor, labels: torch.Tensor, pad_id=None, unk_id=None, predictions_provided=False):
    return get_micro_precision_recall_f1(logits, labels, pad_id, unk_id, predictions_provided)[-1]


def micro_precision(logits: torch.Tensor, labels: torch.Tensor, pad_id=None, unk_id=None, predictions_provided=False):
    return get_micro_precision_recall_f1(logits, labels, pad_id, unk_id, predictions_provided)[0]


def micro_recall(logits: torch.Tensor, labels: torch.Tensor, pad_id=None, unk_id=None, predictions_provided=False):
    return get_micro_precision_recall_f1(logits, labels, pad_id, unk_id, predictions_provided)[1]


def compute_rouge(logits: Union[torch.Tensor, list], labels: torch.Tensor, pad_id=None, predictions_provided=False):
    if isinstance(logits, torch.Tensor):
        if len(logits.shape) == 4:
            # Flatten first two dimensions, if batch size and num_predict is given
            logits = logits.view(-1, logits.shape[2], logits.shape[3])
    if len(labels.shape) == 3:
        # Flatten first two dimensions, if batch size and num_predict is given
        labels = labels.view(-1, labels.shape[2])

    if predictions_provided:
        predictions = logits
    else:
        if isinstance(logits, torch.Tensor):
            predictions = logits.argmax(-1)
        else:
            predictions = [sample_logits.argmax(-1).squeeze() for sample_logits in logits]

    predictions = [" ".join([str(p.item()) for p in prediction if pad_id is None or p.item() != pad_id]).lower()
                   for prediction in predictions]
    predictions = [prediction if prediction else "EMPTY" for prediction in predictions]
    targets = [" ".join([str(t.item()) for t in target if pad_id is None or t.item() != pad_id]).lower()
               for target in labels]
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, refs=targets, avg=True)
    return scores


def rouge_2(logits: Union[torch.Tensor, list], labels: torch.Tensor, pad_id=None, predictions_provided=False):
    scores = compute_rouge(logits, labels, pad_id=pad_id, predictions_provided=predictions_provided)
    return scores['rouge-2']['f']


def rouge_l(logits: Union[torch.Tensor, list], labels: torch.Tensor, pad_id=None, predictions_provided=False):
    scores = compute_rouge(logits, labels, pad_id=pad_id, predictions_provided=predictions_provided)
    return scores['rouge-l']['f']
