import torch

from base.config import Config
from config.constants import rec_special_tokens_dict, SUCCESS_RATE, ITEM_FREQ, AVG_TURN, SL_RATIO, FAIRNESS, \
    TOXICITY


class DatasetConfig(Config):
    dataset_name = ""
    train_data_path = ""
    dev_data_path = ""
    test_data_path = ""
    save_train_convs = True
    log = True
    log_dir = ""
    saved_dir = ""


class DatasetConfigForRecommendation(DatasetConfig):
    save_goal_topic = True
    goal_path = ""
    topic_path = ""
    n_goals = 14
    n_topics = 13566
    num_train_items = 0
    num_dev_items = 0
    num_test_items = 0

    def __init__(self, params):
        """
        constructor for class dataset for recommendation
        :param params: set of parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class DatasetConfigForNegotiation(DatasetConfig):
    save_goal = True
    goal_path = ""
    n_goals = 14
    num_train_cases = 0
    num_dev_cases = 0
    num_test_cases = 0

    def __init__(self, params):
        """
        constructor for class dataset for negotiation scenario
        :param params: set of parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class DatasetConfigForEmotionalSupport(DatasetConfig):
    save_goal = True
    goal_path = ""
    n_goals = 14
    num_train_cases = 0
    num_dev_cases = 0
    num_test_cases = 0

    def __init__(self, params):
        """
        constructor for class dataset for emotional support
        :param params: set of parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class ModelConfig(Config):
    """
    Default Parameters
    """

    per_device_train_batch_size = 5
    per_device_eval_batch_soze = 5
    device = torch.device("cuda:0")
    num_workers = 0
    max_sequence_lenth = 512
    cached_dir = ""

    # dictionary containing special tokens
    # should be different for each setting
    special_tokens_dict = rec_special_tokens_dict

    weight_decay: 0.01
    learning_rate: 5e-5
    num_warmup_steps: 3000
    gradient_accumulation_steps: 1
    num_train_epochs: 5
    dropout = 0.1

    # default objective weights
    objective_weights = [1.0, 1.0, 1.0]


class GenerationConfig(Config):
    """
    common parameters for generation
    """
    max_gen_length = 30
    scenario_name = "recommendation"
    dataset = ''

    def __init__(self, params):
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class EvaluationConfig(Config):
    def __init__(self, params):
        """
        constructor for class evaluation config
        :param params: a set of parameter names and their values
        """
        super().__init__()
        for k, v in params.items():
            if k in self.__dir__():
                setattr(self, k, v)


class GameConfig(Config):
    name = ""
    log_dir = ""
    saved_dir = ""
    n = 10
    epsilon = 1.0
    max_horizon = 10
    seed = 42

    def __init__(self, params):
        """
        constructor for class evaluation config
        :param params: a set of parameter names and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)
        # YAML may override `objectives`; recompute n_objectives accordingly
        # so the model's n_objectives matches the actual reward vector length.
        if hasattr(self, 'objectives') and self.objectives is not None:
            self.n_objectives = len(self.objectives)


class RecommendationGameConfig(GameConfig):
    name = 'recommendation'
    epsilon = 1.0
    terminated_action = "Say goodbye"
    max_horizon = 5
    objectives = [SUCCESS_RATE, ITEM_FREQ]
    n_objectives = len(objectives)
    pass


# class RecommendationGameConfig(GameConfig):
#     name = 'recommendation'
#     epsilon = 1.0
#     terminated_action = "Say goodbye"
#     max_horizon = 5
#     objectives = [SUCCESS_RATE, ITEM_FREQ, AVG_TURN]
#     n_objectives = len(objectives)
#     pass


# class NegotiationGameConfig(GameConfig):
#     name = 'negotiation'
#     epsilon = 1.0
#     terminated_action = "Accept"
#     max_horizon = 5
#     objectives = [SL_RATIO, FAIRNESS]
#     n_objectives = len(objectives)
#     pass


class NegotiationGameConfig(GameConfig):
    name = 'negotiation'
    epsilon = 1.0
    terminated_action = "Say goodbye"
    max_horizon = 5
    objectives = [SL_RATIO, FAIRNESS, SUCCESS_RATE]
    n_objectives = len(objectives)
    # If True, compute_reward delegates buyer price extraction to the LLM
    # (CLI: --use_llm_price_extraction). Default off → regex heuristic.
    use_llm_price_extraction = False
    # If True, the buyer/seller LLMs append a [[PRICE: x]] tag to every reply
    # and compute_reward parses the buyer tag directly (CLI: --use_price_tag).
    use_price_tag = False
    pass


class EmotionalSupportGameConfig(GameConfig):
    name = 'emotional_support'
    epsilon = 1.0
    terminated_action = "Say goodbye"
    max_horizon = 5
    objectives = [SUCCESS_RATE, TOXICITY, AVG_TURN]
    n_objectives = len(objectives)
    # the reward dictionary
    # which convert a textual output to scalar reward
    # borrowed from the ppdpp model
    reward_dict = {
        'worse': -1.0,
        'same': -0.5,
        'better': 0.5,
        'solved': 1.0,
    }
    pass


class ClarificationGameConfig(GameConfig):
    pass
