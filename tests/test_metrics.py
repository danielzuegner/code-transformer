from statistics import mean
from unittest import TestCase

import torch

from code_transformer.utils.metrics import precision, recall, topk_accuracy, top1_accuracy, \
    non_trivial_words_accuracy, get_best_non_unk_predictions

VOCAB_SIZE = 39
UNK_ID = 0
PAD_ID = 3
NUM_SUB_TOKENS = 5


class TestMetrics(TestCase):

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

    def test_precision(self):
        # Simple case
        logits = torch.tensor([[TestMetrics.create_prediction(10, 11, 12),
                                TestMetrics.create_prediction(10, 10, 10, 11)]])
        labels = torch.tensor([[TestMetrics.create_label(11, 10),
                                TestMetrics.create_label(10, 11, 12, 13, 14)]])

        self.assertAlmostEqual(precision(logits, labels, PAD_ID), 5 / 6)

        # Multiple predictions
        logits = torch.tensor([[TestMetrics.create_prediction(10, 11, 12),
                                TestMetrics.create_prediction(13, 14)]])
        labels = torch.tensor([[TestMetrics.create_label(11, 10, 12),
                                TestMetrics.create_label(15, 16, 17)]])

        self.assertAlmostEqual(precision(logits, labels, PAD_ID), 1 / 2)

        # Multiple samples
        logits = torch.tensor([[TestMetrics.create_prediction(10, 11, 12),
                                TestMetrics.create_prediction(13, 14)],
                               [TestMetrics.create_prediction(),
                                TestMetrics.create_prediction()]
                               ])
        labels = torch.tensor([[TestMetrics.create_label(11, 10, 12),
                                TestMetrics.create_label()],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(15)]])

        self.assertAlmostEqual(precision(logits, labels, PAD_ID), 1 / 2)

        # Without ignoring [PAD]
        self.assertAlmostEqual(precision(logits, labels), mean([1, 3 / 5, 1, 1]))

    def test_recall(self):
        logits = torch.tensor([[TestMetrics.create_prediction(10, 11),
                                TestMetrics.create_prediction(10, 11, 12, 13)],
                               [TestMetrics.create_prediction(),
                                TestMetrics.create_prediction()],
                               [TestMetrics.create_prediction(10, 11),
                                TestMetrics.create_prediction(10, 11)]
                               ])
        labels = torch.tensor([[TestMetrics.create_label(11, 10, 12),
                                TestMetrics.create_label(11, 10, 12)],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(15)],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(15)]
                               ])

        self.assertAlmostEqual(recall(logits, labels, PAD_ID), mean([2 / 3, 1, 1, 0, 0, 0]))

    def test_topk_accuracy(self):
        logits = torch.tensor([[TestMetrics.create_prediction([10, 20, 30], [11, 21, 31]),
                                TestMetrics.create_prediction([10, 20, 30], [11, 21, 31]),
                                TestMetrics.create_prediction([10, 20, 30], [11, 21, 31])]])
        labels = torch.tensor([[TestMetrics.create_label(10, 11),
                                TestMetrics.create_label(20, 31),
                                TestMetrics.create_label(15, 21)]])

        self.assertAlmostEqual(topk_accuracy(3, logits, labels), 2 / 3)

        logits = torch.tensor([[TestMetrics.create_prediction(10, 11),
                                TestMetrics.create_prediction(10, 11)],
                               [TestMetrics.create_prediction(),
                                TestMetrics.create_prediction()],
                               [TestMetrics.create_prediction(10, 11),
                                TestMetrics.create_prediction(10, 11)]
                               ])

        labels = torch.tensor([[TestMetrics.create_label(10, 11),
                                TestMetrics.create_label(10, 11, 12)],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(15)],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(10)]
                               ])

        self.assertAlmostEqual(topk_accuracy(1, logits, labels), mean([1, 0, 1, 0, 0, 0]))
        self.assertAlmostEqual(top1_accuracy(logits, labels), mean([1, 0, 1, 0, 0, 0]))

    def test_nontrivial_accuracy(self):
        logits = torch.tensor([[TestMetrics.create_prediction([10, 20, 30], [11, 21, 31]),
                                TestMetrics.create_prediction([10, 20, 30], [11, 21, 31]),
                                TestMetrics.create_prediction([10, 20, 30], [11, 21, 31])]])
        labels = torch.tensor([[TestMetrics.create_label(10),
                                TestMetrics.create_label(20),
                                TestMetrics.create_label(15)]])

        self.assertIsNone(non_trivial_words_accuracy(logits, labels, PAD_ID))

    def test_non_unk_predictions(self):
        logits = torch.tensor([[TestMetrics.create_prediction(10, [UNK_ID, 11]),
                                TestMetrics.create_prediction(20, 21),
                                TestMetrics.create_prediction([UNK_ID, 30], [UNK_ID, 31])]])
        self.assertTrue(
            (get_best_non_unk_predictions(logits, UNK_ID)[:, :3, :2]
             == torch.tensor([[[10, 11], [20, 21], [30, 31]]]))
                .all())

        logits = torch.tensor([[TestMetrics.create_prediction(10, [UNK_ID, 11], 12),
                                TestMetrics.create_prediction(13, [UNK_ID, 14])],
                               [TestMetrics.create_prediction(),
                                TestMetrics.create_prediction()]
                               ])
        labels = torch.tensor([[TestMetrics.create_label(11, 10, 12),
                                TestMetrics.create_label()],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(15)]])

        self.assertAlmostEqual(precision(logits, labels, pad_id=PAD_ID, unk_id=UNK_ID), (1 + 0 + 1 + 0) / 4)

    def test_non_unk_labels(self):
        logits = torch.tensor([[TestMetrics.create_prediction(10, 12, [UNK_ID, 11]),
                                TestMetrics.create_prediction(13, [UNK_ID, 14])],
                               [TestMetrics.create_prediction(),
                                TestMetrics.create_prediction()],
                               [TestMetrics.create_prediction(10, 10, 10),
                                TestMetrics.create_prediction(10, 10)]])
        labels = torch.tensor([[TestMetrics.create_label(11, 10, 9, UNK_ID),
                                TestMetrics.create_label(UNK_ID, 14)],
                               [TestMetrics.create_label(),
                                TestMetrics.create_label(UNK_ID)],
                               [TestMetrics.create_label(10),
                                TestMetrics.create_label(UNK_ID, UNK_ID, 10)]])

        self.assertAlmostEqual(precision(logits, labels, pad_id=PAD_ID, unk_id=UNK_ID), mean([2 / 3, 1 / 2, 1, 1, 1]))
        self.assertAlmostEqual(recall(logits, labels, pad_id=PAD_ID, unk_id=UNK_ID), mean([2 / 3, 1, 1, 1, 1]),
                               places=5)

        logits = torch.tensor([[TestMetrics.create_prediction(10, 11, 13),
                                TestMetrics.create_prediction(10, [UNK_ID, 11], [UNK_ID, 12])],
                               [TestMetrics.create_prediction([10, UNK_ID, 30], [11, 21, 31], [UNK_ID, 22, 32]),
                                TestMetrics.create_prediction(10)],
                               [TestMetrics.create_prediction([UNK_ID, 20]),
                                TestMetrics.create_prediction()]
                               ])
        labels = torch.tensor([[TestMetrics.create_label(10, 11, UNK_ID),
                                TestMetrics.create_label(10, 11, 12)],
                               [TestMetrics.create_label(30, 31, 32),
                                TestMetrics.create_label(UNK_ID)],
                               [TestMetrics.create_label(20),
                                TestMetrics.create_label()]
                               ])

        self.assertAlmostEqual(top1_accuracy(logits, labels, unk_id=UNK_ID, pad_id=PAD_ID), mean([1, 1, 0, 1, 1]))
        self.assertAlmostEqual(topk_accuracy(5, logits, labels, unk_id=UNK_ID, pad_id=PAD_ID), mean([1, 1, 1, 1, 1]))
        self.assertAlmostEqual(non_trivial_words_accuracy(logits, labels, PAD_ID, unk_id=UNK_ID), mean([1, 1, 0]))
