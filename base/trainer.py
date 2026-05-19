import copy
from abc import ABC, abstractmethod
import torch

loggers = None


class Trainer(ABC):

    def __init__(self, game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                 loggers):
        """

        :param game_config:
        :param model_config:
        :param accelerator:
        :param game:
        :param model:
        :param offline_evaluator:
        :param online_evaluator:
        :param loggers:
        """
        self.accelerator = accelerator
        self.model_config = model_config
        self.offline_evaluator = offline_evaluator
        self.online_evaluator = online_evaluator
        self.loggers = loggers
        self.game_config = game_config
        self.game = game
        self.model = model
        self.device = self.accelerator.device

        # the global optimization step
        self.global_step = 0

        # progress bar
        self.progress_bar = None

    @abstractmethod
    def process_dataset(self, dataset):
        """
        method that return processed data instances.
        :return: processed data instances
        """
        raise NotImplementedError("PLease implement this method")

    @abstractmethod
    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1):
        """
        Method that constructs dataloaders using given processed data instances
        :param data_instances: the given processed data instances
        :param batch_size: number of batch size
        :param shuffle: trye if we shuffle the dataset.
        :param num_workers: number of worker used for loading the dataset
        :return:
        """
        raise NotImplementedError("PLease implement this method")

    @abstractmethod
    def create_optimizer(self, learning_rate=1e-5):
        """
        method that creates the optimizer for training the model
        :return: Torch optimizer
        """
        raise NotImplementedError("Please implement this method")

    def create_scheduler(self, optimizer, num_warmup_steps, max_train_steps):
        """
        method that creates the scheduler for training the model
        :return: Torch Scheduler
        """
        raise NotImplementedError("Please implement this method")

    def create_criterion(self):
        """
        method that creates the criterion for training the model
        :return: Torch Loss Function
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def train_epoch(self, data_loader, optimizer, lr_scheduler, criterion, max_train_steps):
        """
        method that trains the model with one epoch
        :param data_loader: data loader for training the model
        :param optimizer: optimizer for training the model
        :param lr_scheduler:  the lr scheduler for training the model
        :param criterion: the loss function that we use to train the model
        :param max_train_steps: the maximum number of training steps
        :return: training loss
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def eval_epoch(self, data_loader, criterion):
        """
        method that eval the model
        :param data_loader: the given data loader
        :param criterion: the given loss function
        :return: eval loss
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def train_sft(self, dataset, device=None):
        """
        method that train the model on a supervised manner
        :param dataset: the dataset that we use to train the model
        :param device: the device used to train the model
        :return: the trained model
        """
        raise NotImplementedError("trainer is not implemented")

    def save_model(self, file_path):
        """
        method that saves the model to a particular file path
        :return: None
        """
        torch.save(self.model, file_path)

    def load_model(self, load_file_path):
        """
        method the load model checkpoint  from a particular file path
        :param load_file_path: the path to the checkpoint
        :return:
        """
        self.model = torch.load(load_file_path, weights_only=False)

    def predict(self, instance):
        """
        method that predict the output given an particular instance
        :param instance: the given instance
        :return: None
        """
        raise NotImplementedError("This method must be implemented")

    def test(self, dataset):
        """
        function that evaluates the performance of the model on the test set
        :param dataset: the dataset that we wish to evaluate the performance.
        :return: None
        """
        raise NotImplementedError("This method must be implemented")
