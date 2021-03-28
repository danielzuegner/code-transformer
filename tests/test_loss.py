from unittest import TestCase

import torch
from torch.nn import CrossEntropyLoss

from code_transformer.utils.loss import LabelSmoothingLoss

VOCAB_SIZE = 37
NUM_SUB_TOKENS = 5
PAD_ID = 3


class TestLoss(TestCase):

    @staticmethod
    def create_prediction(*desired_words):
        token = []
        for desired_word in desired_words:
            vocab_distribution = [0 for _ in range(VOCAB_SIZE)]
            if isinstance(desired_word, list):
                for i, w in enumerate(desired_word):
                    vocab_distribution[w] = len(desired_word) - i
            else:
                vocab_distribution[desired_word] = 1
            token.append(vocab_distribution)
        for _ in range(NUM_SUB_TOKENS - len(desired_words)):
            vocab_distribution = [0 for _ in range(VOCAB_SIZE)]
            vocab_distribution[PAD_ID] = 1
            token.append(vocab_distribution)
        return token

    @staticmethod
    def create_label(*desired_words):
        label = [w for w in desired_words]
        for i in range(NUM_SUB_TOKENS - len(desired_words)):
            label.append(PAD_ID)
        return label

    def test_label_smoothing(self):
        label_smoothing = LabelSmoothingLoss()
        label_smoothing_01 = LabelSmoothingLoss(0.1)
        cross_entropy = CrossEntropyLoss()

        logits = torch.tensor([[TestLoss.create_prediction(10),
                                TestLoss.create_prediction(11)],
                               [TestLoss.create_prediction(20, 21),
                                TestLoss.create_prediction()]], dtype=torch.float32)
        labels = torch.tensor([[TestLoss.create_label(10),
                                TestLoss.create_label(10)],
                               [TestLoss.create_label(20),
                                TestLoss.create_label(20)]])

        self.assertAlmostEqual(label_smoothing(logits.view(-1, VOCAB_SIZE), labels.view(-1)),
                               cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1)))

        logits = torch.tensor(TestLoss.create_prediction(10), dtype=torch.float32)
        labels = torch.tensor(TestLoss.create_label(10))
        self.assertTrue(label_smoothing_01(logits, labels) > cross_entropy(logits, labels))

