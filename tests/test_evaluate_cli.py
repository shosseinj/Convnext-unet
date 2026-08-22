import unittest

import torch.nn as nn

from evaluation_core import count_parameters


class EvaluateCliTests(unittest.TestCase):
    def test_parameter_counts_distinguish_trainable(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
        for parameter in model[1].parameters():
            parameter.requires_grad = False
        trainable, total = count_parameters(model)
        self.assertEqual(trainable, 16)
        self.assertEqual(total, 21)


if __name__ == "__main__":
    unittest.main()
