from collections import defaultdict
import torch

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

from base.metric import Metric

try:
    from rouge import Rouge
except ModuleNotFoundError:
    class Rouge:
        def get_scores(self, hypothesis, reference):
            return [{
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }]
from config.constants import (
    SUCCESS_RATE, AVG_TURN, SL_RATIO, FAIRNESS, TOXICITY, ITEM_FREQ,
    USER_REWARD, DEAL_RATE, LLM_SUCCESS_RATE, GOAL_SUCCESS_RATE, T2DA,
    CVR, BLOCKED_CVR, ACTUAL_CVR,
)


def _cal_rouge(hypothesis, reference):
    """
    both hypothesis and reference are str
    """
    if hypothesis == '':
        return 0, 0, 0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']
    except:
        print("something wrong here! ", hypothesis, reference)
        return 0, 0, 0


class OfflineMetric(Metric):
    """
    Type of metric class
    Just use for coding convenience
    """
    pass


class OnlineMetric(Metric):
    """
    Type of metric class
    Just use for coding convenience
    """

    def compute(self, results):
        raise NotImplementedError("This method must be implemented")


class Accuracy(OfflineMetric):

    def __init__(self):
        """
        constructor for class Accuracy
        """
        super().__init__(None)

    def compute(self, preds, labels):
        """
        method that compute the accuracy the model predictions
        :param preds: a list of predictions (int)
        :param labels:  a list of ground-truth labels
        :return: a scalar which is the accuracy
        """
        return accuracy_score(y_true=labels, y_pred=preds)


class PrecisionRecallF1(OfflineMetric):

    def __init__(self):
        """
        Constructor for class Precision Recall F1
        """
        super().__init__(None)

    def compute(self, preds, labels, average='macro'):
        """
        method that compute the precision, recall and f1 metrics
        :param preds: a list of predictions
        :param labels: a list of ground truh labels
        :param average: type of f1 score
        :return: the precision, recall and f1 metrics
        """
        p, r, f1, _ = precision_recall_fscore_support(
            y_true=labels, y_pred=preds, average=average)
        return p, r, f1


class DistN(OfflineMetric):
    def __init__(self, Ns=[2, 3, 4]):
        """
        Constructor for class Distinct N-grams
        :param Ns:
        """
        super().__init__(None)
        self.Ns = Ns

    def compute(self, preds, labels=None):
        """

        :param preds:
        :param labels:
        :return:
        """
        # for each n, count the number of n-gram tokens
        metric = defaultdict(set)
        for str in preds:
            str = str.split()
            for k in self.Ns:
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    metric[dist_k].add(token)

        # calculate distinct-ngrams
        for k, v in metric.items():
            metric[k] = len(v) / len(preds)

        return metric


class BleuN(OfflineMetric):

    def __init__(self, Ns=[2, 3, 4]):
        """
        Constructor for class BleuN grams metrics
        :param Ns:
        """
        super().__init__(None)
        self.Ns = Ns

    def compute(self, preds, labels):
        """
        method that computes the Bleu N metrics
        :param preds: the list of predicted sentences
        :param labels: the list of ground truth sentences
        :return:
        """
        metric = defaultdict(float)
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for idx, k in enumerate(self.Ns):
                weights = [0] * 4
                weights[idx] = 1
                metric[f'Bleu@{k}'] += sentence_bleu(label, pred, weights)

        # calculate the averaged Bleu-N
        for k, v in metric.items():
            metric[k] = v / len(preds)

        return metric


class RougeN(OfflineMetric):

    def __init__(self, Ns=['1', '2', 'l']):
        """
        constructor for class RougeN metric
        :param Ns:
        """
        super().__init__(None)
        self.Ns = Ns

    def compute(self, preds, labels):
        """
        method that compute the Rouge-N metrics
        :param preds: list of predicted sentences
        :param labels: list of ground truth sentences
        :return:
        """
        metric = defaultdict(float)
        for pred, label in zip(preds, labels):
            rouge_1, rouge_2, rouge_l = _cal_rouge(pred, label)
            metric["rouge@1"] += rouge_1
            metric["rouge@2"] += rouge_2
            metric["rouge@L"] += rouge_l

        # calculate the averaged Rouge-N
        for k, v in metric.items():
            metric[k] = v / len(preds)

        return metric


class Perplexity(OfflineMetric):

    def __init__(self):
        super().__init__(None)
        self.value = 0

    def compute(self, preds, labels):
        return 0


class SR(OnlineMetric):

    def compute(self, results):
        """
        method that computes the success rate of interactive evaluation
        :param results: list of results of interaction conversations
        :return:
        """
        # compute the success rate
        sr = 0
        # loop over the list of results of conversations
        for result in results:
            # check if the conversation is successful
            sr += int(result[SUCCESS_RATE])
        # compute the average success rate
        return sr / len(results)


class Item_Freq(OnlineMetric):

    def compute(self, results):
        """
        method that computes the item frequency in the recommendation scenario
        :param results: the list of results of interactive conversations
        :return:
        """
        # compute the success rate
        item_freq = 0
        # loop over the list of results of conversations
        for result in results:
            # check if the conversation is successful
            item_freq += result[ITEM_FREQ]

        # compute the average success rate
        return item_freq / len(results)


class AverageTurn(OnlineMetric):
    def compute(self, results):
        """
        method that computes the average conversation turn
        :param results: the list of results of conversations
        :return:
        """
        # compute the success rate
        avg_turn = 0.0
        avg_turn_reward = 0.0
        # loop over the list of results of conversations
        for result in results:
            assert len(result[AVG_TURN]) == 2
            # compute the conversation turn
            avg_turn += result[AVG_TURN][0]
            avg_turn_reward += result[AVG_TURN][1]
        # compute the average success rate
        return avg_turn / len(results), avg_turn_reward / len(results)


class SL_Ratio(OnlineMetric):

    def compute(self, results):
        """
        method that computes the Sale-to-list ratio metric
        :param results: the list of dictionary containing results of conversations
        :return:
        """
        # sale_to_list ratio
        sl_ratio = 0
        for result in results:
            sl_ratio += result[SL_RATIO]
        return sl_ratio / len(results)


class Fairness(OnlineMetric):

    def compute(self, results):
        """
        method that computes the Fairness metric
        :param results: the list of dictionary containing results of conversations
        :return:
        """
        # fairness score
        fairness = 0
        for result in results:
            fairness += result[FAIRNESS]
        return fairness / len(results)


class DealRate(OnlineMetric):

    def compute(self, results):
        """
        method that computes the Deal Rate metric
        :param results: the list of dictionary containing results of conversations
        :return:
        """
        # fairness score
        fairness = 0
        for result in results:
            fairness += result[DEAL_RATE]
        return fairness / len(results)


class LLMSuccessRate(OnlineMetric):
    def compute(self, results):
        return sum(float(result.get(LLM_SUCCESS_RATE, 0.0)) for result in results) / max(len(results), 1)


class GoalSuccessRate(OnlineMetric):
    def compute(self, results):
        return sum(float(result.get(GOAL_SUCCESS_RATE, result.get("gsr", 0.0))) for result in results) / max(len(results), 1)


class TurnToDriftAdaptationDelay(OnlineMetric):
    def compute(self, results):
        values = [
            result.get(T2DA)
            for result in results
            if result.get(T2DA) is not None
        ]
        return (sum(values) / len(values)) if values else None


class ConstraintViolationRate(OnlineMetric):
    def compute(self, results):
        return sum(float(result.get(CVR, 0.0)) for result in results) / max(len(results), 1)


class BlockedConstraintViolationRate(OnlineMetric):
    def compute(self, results):
        return sum(float(result.get(BLOCKED_CVR, 0.0)) for result in results) / max(len(results), 1)


class ActualConstraintViolationRate(OnlineMetric):
    def compute(self, results):
        return sum(float(result.get(ACTUAL_CVR, 0.0)) for result in results) / max(len(results), 1)


class Toxicity(OnlineMetric):

    def compute(self, results):
        """
        method that computes the Toxicity metric
        :param results: the list of dictionary containing results of conversations
        :return:
        """
        toxicity = 0.0
        for result in results:
            toxicity += result[TOXICITY]
        return toxicity / len(results)


class User_Reward(OnlineMetric):

    def compute(self, results):
        """
        method that computes the user satisfaction metric
        :param results: the list of dictionary containing results of conversations
        :return:
        """
        toxicity = 0.0
        for result in results:
            toxicity += result[USER_REWARD]
        return toxicity / len(results)


class WordF1(OfflineMetric):

    def __init__(self):
        super().__init__(None)
        self.value = 0

    def reset(self):
        self.value = 0

    def compute(self, preds, labels):
        return 0

    def report(self):
        return self.value
