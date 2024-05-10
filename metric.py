from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('true_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        for class_idx in range(self.num_classes):
            class_preds = preds == class_idx
            class_target = target == class_idx

            true_positives = torch.sum(class_preds & class_target).float()
            false_positives = torch.sum(class_preds & ~class_target).float()
            false_negatives = torch.sum(~class_preds & class_target).float()

            self.true_positives[class_idx] += true_positives
            self.false_positives[class_idx] += false_positives
            self.false_negatives[class_idx] += false_negatives

    def compute(self):
        eps = 1e-7  # to avoid division by zero
        precision = self.true_positives / (self.true_positives + self.false_positives + eps)
        recall = self.true_positives / (self.true_positives + self.false_negatives + eps)
        f1_scores = 2 * (precision * recall) / (precision + recall + eps)
        return torch.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0).mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
