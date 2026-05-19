import os
import json

from base.logger import Logger


class FileLogger(Logger):

    def __init__(self, game_config, dataset_config, model_config, log_dir, local_time, random_seed, model_name,
                 **kwargs):
        """
        constructor for class File Logger
        :param game_config: the configuration of the current scenario
        :param dataset_config: the configurations of the dataset
        :param model_config: the configurations of the model
        :param log_dir: the log directory path
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

        # creating the log directory if it is not existed
        log_dir = os.path.join(
            self.scenario_config.log_dir,
            self.dataset_config.dataset_name,
            model_name,
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # create the path to the log file
        self.file_path = os.path.join(self.log_dir, f"{self.random_seed}-{self.local_time}")
        self.f = open(self.file_path, 'a')

        # write the scenario configurations to the file
        self.f.write("Scenario Configuration \n")
        for k, v in game_config.get_params():
            self.f.write(f"[{k}]: {v} \n")

        self.f.write('-' * 50 + '\n')

        # write the dataset configurations to the file
        self.f.write("Dataset Configuration \n")
        for k, v in dataset_config.get_params():
            self.f.write(f"[{k}]: {v} \n")

        self.f.write('-' * 50 + '\n')

        # write the model configurations to the file
        self.f.write("Scenario Configuration \n")
        for k, v in model_config.get_params():
            self.f.write(f"[{k}]: {v} \n")

        self.f.write('-' * 50 + '\n')

        # logging information of the current run
        self.f.write("Runtime Information \n")
        self.f.write(f"[Local Time]: {local_time}, [RandomSeed]: {random_seed} ,[Model Name]: {model_name} \n")

        self.f.write('-' * 50 + '\n')

    def record(self, results, step=None):
        """
        method that records the results and log them to a file
        :param results: the current results, in form of a dictionary
        :param step: the current step
        :return: None
        """
        self.f.write(f"[Random Seed]: {self.random_seed} , [Model Name]: {self.model_name}, [Step]: {step} \n")
        for k, v in results.items():
            self.f.write(f"[Metric]: {k}, [Values]: {v} \n")
        self.f.write('-' * 50 + '\n')

    def save_responses(self, list_responses, log_dir, file_name):
        """
        method that save a list of generated conversations to file
        :param list_responses: a list of generated responses
        :param log_dir: the path to the directory that we use to save the generated responses
        :param: file_name: the name of the file
        :return: None
        """
        # create the generated responses folder
        convs_dir_path = os.path.join(log_dir, "responses")
        os.makedirs(convs_dir_path, exist_ok=True)
        # create the file path
        convs_file_path = os.path.join(convs_dir_path, file_name)
        with open(convs_file_path, 'w') as f:
            # loop overall responses records
            for instance in list_responses:
                # make the each record is in dictionary format
                assert isinstance(instance, dict)
                # convert dictionary to string
                json_string = json.dumps(instance)
                # save the json string to file
                f.write(json_string + "\n")
