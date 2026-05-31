try:
    import wandb
except ModuleNotFoundError:
    class _NoOpRun:
        def log(self, *args, **kwargs):
            return None

    class _NoOpWandB:
        config = type("_NoOpConfig", (), {})()

        def login(self, *args, **kwargs):
            return None

        def init(self, *args, **kwargs):
            return _NoOpRun()

    wandb = _NoOpWandB()
from base.logger import Logger
from eval.metric import *
from config.constants import RECOMMENDATION, NEGOTIATION


class WanDBLogger(Logger):

    def __init__(self, game_config, dataset_config, model_config, project_name, wandb_key, local_time,
                 random_seed,
                 model_name, exp_name, **kwargs):
        """
        constructor for class Wandb Logger
        :param game_config: an instance of the scenario config class
        :param dataset_config: an instacne of the dataset config class
        :param model_config: an instance of the model config class
        :param project_name: the project's name
        :param wandb_key: wandb login key
        :param local_time: the current local time
        :param random_seed: the current random seed
        :param model_name: the model name
        """
        super().__init__(**kwargs)
        self.scenario_config = game_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.local_time = local_time
        self.random_seed = random_seed
        self.model_name = model_name
        self.wandb_key = wandb_key

        # login with your wandb account
        wandb.login(
            key=wandb_key,
            relogin=False,
        )

        # for recommendation scenario only
        if game_config.name == RECOMMENDATION:
            exp_name = exp_name + "_" + self.dataset_config.domain

        # initialize the run
        self.run = wandb.init(
            project=project_name,
            group=f"{game_config.name}",
            job_type=f"{self.dataset_config.dataset_name}",
            reinit=True,
            name=f"{random_seed}|{exp_name}|{model_name}|{local_time}"
        )

        # tracking the scenario configurations
        for k, v in game_config.get_params():
            setattr(wandb.config, k, v)

        # tracking the dataset configurations
        for k, v in dataset_config.get_params():
            setattr(wandb.config, k, v)

        # # tracking the model configurations
        # for k, v in model_config.get_params():
        #     setattr(wandb.config, k, v)

    def record(self, results, steps=None):
        """
        method that record the results using wandb logger
        :param results: the current results, in form of a dictionary
        :param steps: the current step
        :return: None
        """
        for k, v in results.items():
            # precision_recall_f1,
            # distinct_n_grams
            # rouge_n
            if isinstance(v, list) or isinstance(v, tuple):
                # precision, recall, f1 scores
                if k == str(PrecisionRecallF1.__name__):
                    p, r, f1 = v
                    self.run.log({'Precision': p}, step=steps)
                    self.run.log({'Recall': r}, step=steps)
                    self.run.log({'F1': f1}, step=steps)
                # Distinct N-grams
                elif k == str(DistN.__name__):
                    pass
                # Rouge N
                elif k == str(RougeN.__name__):
                    pass
            # other metrics
            else:
                self.run.log({k: v}, step=steps)
