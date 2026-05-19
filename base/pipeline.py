from abc import ABC, abstractmethod
import os
import pickle


class Pipeline(ABC):

    def __init__(self, dataset_config, dataset, trainer):
        """
        constructor for class pipeline
        :param dataset_config: the configuration of the dataset
        :param dataset: an instance of the dataset class
        :param trainer: an instance of the trainer class
        """
        self.trainer = trainer
        self.game_config = self.trainer.game_config
        self.dataset_config = dataset_config
        self.model_config = self.trainer.model_config
        self.game = self.trainer.game
        self.model = self.trainer.model
        self.device = self.model_config.device
        self.dataset = dataset

        # creating the log dir
        log_dir = os.path.join(
            self.game_config.log_dir,
            self.dataset_config.dataset_name,
            str(self.model.__class__.__name__),
        )
        os.makedirs(log_dir, exist_ok=True)
        self.model_config.log_dir = log_dir

        # creating the saved dir
        saved_dir = os.path.join(
            self.game_config.saved_dir,
            self.dataset_config.dataset_name,
            str(self.model.__class__.__name__) + "_" + str(self.game.game_config.seed),
        )
        os.makedirs(saved_dir, exist_ok=True)

        self.model_config.saved_dir = saved_dir

        # simulators
        self.dev_simulators = None
        self.test_simulators = None

    def run_sft(self):
        """
        This method runs the whole supervised fine-tuning pipeline including model training, selection and evaluation
        :return: the results of the current run
        """
        raise NotImplementedError("Please implement this method")

    def run_rlt(self):
        """
        This method run the whole reinforcement learning tuning process
        :return: None
        """
        raise NotImplementedError("This method should be implemented")

    @abstractmethod
    def inference(self, instance):
        """
        method that predict the output for a given particular input instance
        :param instance: the given input instance
        :return: the corresponding output
        """
        raise NotImplementedError("This method must be implemented")

    def load_pretrained_model(self, is_rl=False, is_last=False):
        """
        method that laod the model checkpoint to the current model class
        :param is_rl: if we load the rl pretrained model
        :return: None
        """
        # create the model path
        # this is the path for sft model
        if not is_rl:
            saved_model_path = os.path.join(self.model_config.saved_dir, "model.pth")
        # this is the path for the rlt model
        else:
            saved_model_path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        if not os.path.exists(saved_model_path):
            raise Exception("There is no pretrained model.")
        # load the model from the checkpoint
        self.model = self.trainer.load_model(saved_model_path)

    def execute(self):
        """
        method that execute the whole pipeline of a model
        :return: None
        """
        raise NotImplementedError("This method should be implemented")

    def save_user_simulators(self, user_simulators):
        """
        method that saves the generated user profile
        :return: None
        """
        # create the model path
        saved_simulator_path = os.path.join(self.model_config.saved_dir, "simulators.pkl")
        if not os.path.exists(saved_simulator_path):
            raise Exception("There is no pretrained model.")

        # load the model from the checkpoint
        with open(saved_simulator_path, 'wb') as f:
            pickle.dump(user_simulators, f)

    def set_user_simulators(self, dev_simulators, test_simulators):
        """
        method that set user simulators to the pipeline
        :param dev_simulators: list of dev simulators
        :param test_simulators: list of test simulators
        :return: None
        """
        self.dev_simulators = dev_simulators
        self.test_simulators = test_simulators

    def get_user_simulators(self):
        """
        method that return the user simulators
        :return: list of dev and test user simulators
        """
        return self.dev_simulators, self.test_simulators

    def run_online_test(self):
        """
        method run online evaluation on the test set
        :return:
        """
        raise NotImplementedError("This method must be implemented")

    def save_results(self, results, file_path):
        """
        method that save the results of the current run
        :param results: a dictionary that contains the results of the current run
        :param file_path: the path to the result file
        :return:
        """
        raise NotImplementedError("This method must be implemented")
