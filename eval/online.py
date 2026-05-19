from collections import defaultdict
from loguru import logger
from base.evaluator import Evaluator


class OnlineEvaluator(Evaluator):

    def __init__(self, metrics):
        """
        constructor for class online evaluator
        :param metrics: set of interested metrics, should be something like SR, Avg-turn, Fairness
        """
        super().__init__()
        self.metrics = metrics
        self.preds = []
        self.values = {}

    def record(self, results):
        """
        method that records the values of each conversation during evaluation
        :param results: the results of each conversation
        :return: None
        """
        logger.debug(f"online eval results: {results}")
        self.preds.append(results)

    def reset(self):
        """
        method that resets values of metrics
        :return: None
        """
        self.preds = []
        for metric in self.metrics:
            self.values[str(metric.__class__.__name__)] = 0.0

    def report(self):
        """
        method that reports the values of metrics
        :return: None
        """
        # compute the values of metrics
        for metric in self.metrics:
            self.values[metric.__class__.__name__.lower()] = metric.compute(self.preds)
        return self.values

    @staticmethod
    def record_and_report_turn_results(turn_id, list_turn_results):
        """_summary_

        Args:
            turn_results (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert len(list_turn_results) >= 1
        avg_turn_results = defaultdict(list)

        # loop over the list of turn results
        for turn_results in list_turn_results:
            for metric_name, value in turn_results.items():
                avg_turn_results[metric_name].append(value)
            
        # compute the average results each turn
        for metric_name, values in avg_turn_results.items():
            avg_turn_results[metric_name] = sum(values) / len(values)

        return avg_turn_results
            


