import pickle
from argparse import ArgumentParser

import numpy as np

'''
This script preprocesses the data from MethodPaths. It truncates methods with too many contexts,
and pads methods with less paths with spaces.
'''


import re
import subprocess
import sys


class Common:
    internal_delimiter = '|'
    SOS = '<S>'
    EOS = '</S>'
    PAD = '<PAD>'
    UNK = '<UNK>'

    @staticmethod
    def normalize_word(word):
        stripped = re.sub(r'[^a-zA-Z]', '', word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def load_histogram(path, max_size=None):
        histogram = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                parts = line.split(' ')
                if not len(parts) == 2:
                    continue
                histogram[parts[0]] = int(parts[1])
        sorted_histogram = [(k, histogram[k]) for k in sorted(histogram, key=histogram.get, reverse=True)]
        return dict(sorted_histogram[:max_size])

    @staticmethod
    def load_vocab_from_dict(word_to_count, add_values=[], max_size=None):
        word_to_index, index_to_word = {}, {}
        current_index = 0
        for value in add_values:
            word_to_index[value] = current_index
            index_to_word[current_index] = value
            current_index += 1
        sorted_counts = [(k, word_to_count[k]) for k in sorted(word_to_count, key=word_to_count.get, reverse=True)]
        limited_sorted = dict(sorted_counts[:max_size])
        for word, count in limited_sorted.items():
            word_to_index[word] = current_index
            index_to_word[current_index] = word
            current_index += 1
        return word_to_index, index_to_word, current_index

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [Common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [Common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def binary_to_string_3d(binary_string_tensor):
        return [Common.binary_to_string_matrix(l) for l in binary_string_tensor]

    @staticmethod
    def legal_method_names_checker(name):
        return not name in [Common.UNK, Common.PAD, Common.EOS]

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(Common.legal_method_names_checker, top_words))
        return result

    @staticmethod
    def unique(sequence):
        unique = []
        [unique.append(item) for item in sequence if item not in unique]
        return unique

    @staticmethod
    def parse_results(result, pc_info_dict, topk=5):
        prediction_results = {}
        results_counter = 0
        for single_method in result:
            original_name, top_suggestions, top_scores, attention_per_context = list(single_method)
            current_method_prediction_results = PredictionResults(original_name)
            if attention_per_context is not None:
                word_attention_pairs = [(word, attention) for word, attention in
                                        zip(top_suggestions, attention_per_context) if
                                        Common.legal_method_names_checker(word)]
                for predicted_word, attention_timestep in word_attention_pairs:
                    current_timestep_paths = []
                    for context, attention in [(key, attention_timestep[key]) for key in
                                               sorted(attention_timestep, key=attention_timestep.get, reverse=True)][
                                              :topk]:
                        if context in pc_info_dict:
                            pc_info = pc_info_dict[context]
                            current_timestep_paths.append((attention.item(), pc_info))

                    current_method_prediction_results.append_prediction(predicted_word, current_timestep_paths)
            else:
                for predicted_seq in top_suggestions:
                    filtered_seq = [word for word in predicted_seq if Common.legal_method_names_checker(word)]
                    current_method_prediction_results.append_prediction(filtered_seq, None)

            prediction_results[results_counter] = current_method_prediction_results
            results_counter += 1
        return prediction_results

    @staticmethod
    def compute_bleu(ref_file_name, predicted_file_name):
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(["perl", "scripts/multi-bleu.perl", ref_file_name], stdin=predicted_file,
                                    stdout=sys.stdout, stderr=sys.stderr)


class PredictionResults:
    def __init__(self, original_name):
        self.original_name = original_name
        self.predictions = list()

    def append_prediction(self, name, current_timestep_paths):
        self.predictions.append(SingleTimeStepPrediction(name, current_timestep_paths))

class SingleTimeStepPrediction:
    def __init__(self, prediction, attention_paths):
        self.prediction = prediction
        if attention_paths is not None:
            paths_with_scores = []
            for attention_score, pc_info in attention_paths:
                path_context_dict = {'score': attention_score,
                                     'path': pc_info.longPath,
                                     'token1': pc_info.token1,
                                     'token2': pc_info.token2}
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


class PathContextInformation:
    def __init__(self, context):
        self.token1 = context['name1']
        self.longPath = context['path']
        self.shortPath = context['shortPath']
        self.token2 = context['name2']

    def __str__(self):
        return '%s,%s,%s' % (self.token1, self.shortPath, self.token2)


def save_dictionaries(dataset_name, subtoken_to_count, node_to_count, target_to_count, max_contexts, num_examples):
    save_dict_file_path = '{}.dict.c2s'.format(dataset_name)
    with open(save_dict_file_path, 'wb') as file:
        pickle.dump(subtoken_to_count, file)
        pickle.dump(node_to_count, file)
        pickle.dump(target_to_count, file)
        pickle.dump(max_contexts, file)
        pickle.dump(num_examples, file)
        print('Dictionaries saved to: {}'.format(save_dict_file_path))


def process_file(file_path, data_file_role, dataset_name, max_contexts, max_data_contexts):
    sum_total = 0
    sum_sampled = 0
    total = 0
    max_unfiltered = 0
    max_contexts_to_sample = max_data_contexts if data_file_role == 'train' else max_contexts
    output_path = '{}.{}.c2s'.format(dataset_name, data_file_role)
    with open(output_path, 'w') as outfile:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.rstrip('\n').split(' ')
                target_name = parts[0]
                contexts = parts[1:]

                if len(contexts) > max_unfiltered:
                    max_unfiltered = len(contexts)

                sum_total += len(contexts)
                if len(contexts) > max_contexts_to_sample:
                    contexts = np.random.choice(contexts, max_contexts_to_sample, replace=False)

                sum_sampled += len(contexts)

                csv_padding = " " * (max_data_contexts - len(contexts))
                total += 1
                outfile.write(target_name + ' ' + " ".join(contexts) + csv_padding + '\n')

    print('File: ' + data_file_path)
    print('Average total contexts: ' + str(float(sum_total) / total))
    print('Average final (after sampling) contexts: ' + str(float(sum_sampled) / total))
    print('Total examples: ' + str(total))
    print('Max number of contexts per word: ' + str(max_unfiltered))
    return total


def context_full_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           and context_parts[1] in path_to_count and context_parts[2] in word_to_count


def context_partial_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           or context_parts[1] in path_to_count or context_parts[2] in word_to_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-trd", "--train_data", dest="train_data_path",
                        help="path to training data file", required=True)
    parser.add_argument("-ted", "--test_data", dest="test_data_path",
                        help="path to test data file", required=True)
    parser.add_argument("-vd", "--val_data", dest="val_data_path",
                        help="path to validation data file", required=True)
    parser.add_argument("-mc", "--max_contexts", dest="max_contexts", default=200,
                        help="number of max contexts to keep in test+validation", required=False)
    parser.add_argument("-mdc", "--max_data_contexts", dest="max_data_contexts", default=1000,
                        help="number of max contexts to keep in the dataset", required=False)
    parser.add_argument("-svs", "--subtoken_vocab_size", dest="subtoken_vocab_size", default=186277,
                        help="Max number of source subtokens to keep in the vocabulary", required=False)
    parser.add_argument("-tvs", "--target_vocab_size", dest="target_vocab_size", default=26347,
                        help="Max number of target words to keep in the vocabulary", required=False)
    parser.add_argument("-sh", "--subtoken_histogram", dest="subtoken_histogram",
                        help="subtoken histogram file", metavar="FILE", required=True)
    parser.add_argument("-nh", "--node_histogram", dest="node_histogram",
                        help="node_histogram file", metavar="FILE", required=True)
    parser.add_argument("-th", "--target_histogram", dest="target_histogram",
                        help="target histogram file", metavar="FILE", required=True)
    parser.add_argument("-o", "--output_name", dest="output_name",
                        help="output name - the base name for the created dataset", required=True, default='data')
    args = parser.parse_args()

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    subtoken_histogram_path = args.subtoken_histogram
    node_histogram_path = args.node_histogram

    subtoken_to_count = Common.load_histogram(subtoken_histogram_path,
                                                     max_size=int(args.subtoken_vocab_size))
    node_to_count = Common.load_histogram(node_histogram_path,
                                                 max_size=None)
    target_to_count = Common.load_histogram(args.target_histogram,
                                                   max_size=int(args.target_vocab_size))
    print('subtoken vocab size: ', len(subtoken_to_count))
    print('node vocab size: ', len(node_to_count))
    print('target vocab size: ', len(target_to_count))

    num_training_examples = 0
    for data_file_path, data_role in zip([test_data_path, val_data_path, train_data_path], ['test', 'val', 'train']):
        num_examples = process_file(file_path=data_file_path, data_file_role=data_role, dataset_name=args.output_name,
                                    max_contexts=int(args.max_contexts), max_data_contexts=int(args.max_data_contexts))
        if data_role == 'train':
            num_training_examples = num_examples

    save_dictionaries(dataset_name=args.output_name, subtoken_to_count=subtoken_to_count,
                      node_to_count=node_to_count, target_to_count=target_to_count,
                      max_contexts=int(args.max_data_contexts), num_examples=num_training_examples)
