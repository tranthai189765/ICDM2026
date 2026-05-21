from abc import ABC, abstractmethod
import os

from collections import defaultdict
import itertools
import pickle

from loguru import logger
from config.constants import BIG5_PERSONALITY, DECISION_MAKING_STYLE


class Dataset(ABC):

    def __init__(self, dataset_config, **kwargs):
        """
        constructor for the abstract class Dataset
        :param train_data_path:
        :param dev_data_path:
        :param test_data_path:
        :param save_train_convs:
        """
        self.dataset_config = dataset_config

    @abstractmethod
    def pipeline(self, data_path):
        """
        method that employ the dataset preprocessing pipeline
        :param data_path: the path to the data
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def construct_instances(self, conv_id, conv):
        """
        method that converts a conversation to a list of inputs and their corresponding outputs.
        @param conv_id: the index of the conversation
        @param conv: the conversation
        @return: a list of instances
        """
        raise NotImplementedError()

    @abstractmethod
    def read_data(self, data_path):
        """function that reads the data from input file

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def process_data(self, data):
        """Function that process the data given the read data.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def repurpose_dataset(self, data):
        """Function that convert the original dataset from goal-driven setting to target-driven setting.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def return_infor(self):
        """
        method that return information regarding the dataset
        :return: a dictionary that contains the information of the dataset
        """
        raise NotImplementedError()

    def get_config(self):
        """
        Getter of the dataset class
        :return: None
        """
        return self.dataset_config

    def set_config(self, config):
        """
        Setter of the dataset class
        :param config: the new dataset config
        :return: None
        """
        self.dataset_config = config

    @staticmethod
    def load_binary_file(file_path):
        """
        method that loads the goal-topic mapping from file
        :param file_path: the path to the file
        :return: a dictionary that is a mapping of goal, topic to index
        """
        with open(file_path, 'rb') as f:
            pickle_file = pickle.load(f)
            return pickle_file

    @staticmethod
    def save_binary_file(saved_file, file_path):
        """
        method that saves data to binary file
        :param saved_file: the data that we want to save
        :param file_path: the path to the saved file
        :return:
        """
        with open(file_path, 'wb') as f:
            pickle.dump(saved_file, f)

    def get_user_profiles(self):
        """
        the function that return train, dev and test user profiles
        :return:
        """
        pass


class RecommendationDataset(Dataset):

    def __init__(self, dataset_config, **kwargs):
        super().__init__(dataset_config, **kwargs)

        # common attributes of the recommendation datasets
        self.topics = []
        self.goals = []
        self.save_train_convs = self.dataset_config.save_train_convs
        self.train_convs = None

        # generate train, dev and test instances
        self.train_instances = self.pipeline(self.dataset_config.train_data_path)
        self.dev_instances = self.pipeline(self.dataset_config.dev_data_path)
        self.test_instances = self.pipeline(self.dataset_config.test_data_path)

        # log
        self.log = self.dataset_config.log
        if self.log:
            goal_dict = defaultdict(int)
            # log goal count
            for goal in self.goals:
                goal_dict[goal] += 1
            
            print("Goal statistics:", goal_dict)

            # log target item w.r.t data split
            train_target_items = []
            dev_target_items = []
            test_target_items = []

            # log target w.r.t different domains
            movie_target_items = defaultdict(list)
            music_target_items = defaultdict(list)
            food_target_items = defaultdict(list)
            poi_target_items = defaultdict(list)

            for instance in self.train_instances:
                train_target_items.append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'Movie recommendation':
                    movie_target_items['train'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Music recommendation':
                    music_target_items['train'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Food recommendation':
                    food_target_items['train'].append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'POI recommendation':
                    poi_target_items['train'].append(instance['task_background']['target_topic'])

            for instance in self.dev_instances:
                dev_target_items.append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Movie recommendation':
                    movie_target_items['dev'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Music recommendation':
                    music_target_items['dev'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Food recommendation':
                    food_target_items['dev'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'POI recommendation':
                    poi_target_items['dev'].append(instance['task_background']['target_topic'])

            for instance in self.test_instances:
                test_target_items.append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Movie recommendation':
                    movie_target_items['test'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Music recommendation':
                    music_target_items['test'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Food recommendation':
                    food_target_items['test'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'POI recommendation':
                    poi_target_items['test'].append(instance['task_background']['target_topic'])

            print(
                f"Statistics by data splits: Train: {len(list(set(train_target_items)))}, Dev: {len(list(set(dev_target_items)))}, Test: {len(list(set(test_target_items)))}")

            for t in ['train', 'dev', 'test']:
                print(
                    f"Statistics by domain splits {t}: Movie: {len(list(set(movie_target_items[t])))}, Music: {len(list(set(music_target_items[t])))}, Food: {len(list(set(food_target_items[t])))}, POI: {len(list(set(poi_target_items[t])))}")

        # reformat goals and topics
        self.goals = list(set(self.goals))
        self.topics = list(set(self.topics))

        # sort the goals and topics to make sure we have an unique order
        # self.goals = sorted(self.goals)
        # self.topics = sorted(self.topics)

        self.n_goals = len(self.goals)
        self.n_topics = len(self.topics)

        # save goal and topic to file
        if self.dataset_config.save_action:
            print("Saving the goal, topic to files .......")
            ### movie domain
            if self.dataset_config.domain == 'movie':
                if not os.path.exists(self.dataset_config.save_movie_goal_path):
                    RecommendationDataset.save_binary_file(self.goals, self.dataset_config.save_movie_goal_path)
                    RecommendationDataset.save_binary_file(self.topics, self.dataset_config.save_movie_topic_path)
                else:
                    print("Goal, topic files already exist, skip saving")
            # music domain
            elif self.dataset_config.domain == 'music':
                if not os.path.exists(self.dataset_config.save_music_goal_path):
                    RecommendationDataset.save_binary_file(self.goals, self.dataset_config.save_music_goal_path)
                    RecommendationDataset.save_binary_file(self.topics, self.dataset_config.save_music_topic_path)
                else:
                    print("Goal, topic file already exits, skip saving")
            # food domain
            elif self.dataset_config.domain == 'food':
                if not os.path.exists(self.dataset_config.save_food_goal_path):
                    RecommendationDataset.save_binary_file(self.goals, self.dataset_config.save_food_goal_path)
                    RecommendationDataset.save_binary_file(self.topics, self.dataset_config.save_food_topic_path)
                else:
                    print("Goal, topic file already exits, skip saving")
            # poi domain
            elif self.dataset_config.domain == 'poi':
                if not os.path.exists(self.dataset_config.save_poi_goal_path):
                    RecommendationDataset.save_binary_file(self.goals, self.dataset_config.save_poi_goal_path)
                    RecommendationDataset.save_binary_file(self.topics, self.dataset_config.save_poi_topic_path)
                else:
                    print("Goal, topic file already exits, skip saving")
            # all domains
            elif self.dataset_config.domain == 'all':
                if not os.path.exists(self.dataset_config.save_all_goal_path):
                    RecommendationDataset.save_binary_file(self.goals, self.dataset_config.save_all_goal_path)
                    RecommendationDataset.save_binary_file(self.topics, self.dataset_config.save_all_topic_path)
                else:
                    print("Goal, topic file already exits, skip saving")

        # load goal and topic from files
        # this makes sure that every run using the same goal, topic mapping
        if self.dataset_config.load_action:
            print("Loading the goal, topic from files ......")
            # movie domain
            if self.dataset_config.domain == 'movie':
                self.goals = RecommendationDataset.load_binary_file(self.dataset_config.save_movie_goal_path)
                self.topics = RecommendationDataset.load_binary_file(self.dataset_config.save_movie_topic_path)
            # music domain
            elif self.dataset_config.domain == 'music':
                self.goals = RecommendationDataset.load_binary_file(self.dataset_config.save_music_goal_path)
                self.topics = RecommendationDataset.load_binary_file(self.dataset_config.save_music_topic_path)
            # food domain
            elif self.dataset_config.domain == 'food':
                self.goals = RecommendationDataset.load_binary_file(self.dataset_config.save_food_goal_path)
                self.topics = RecommendationDataset.load_binary_file(self.dataset_config.save_food_topic_path)
            # poi domain
            elif self.dataset_config.domain == 'poi':
                self.goals = RecommendationDataset.load_binary_file(self.dataset_config.save_poi_goal_path)
                self.topics = RecommendationDataset.load_binary_file(self.dataset_config.save_poi_topic_path)
            # all domains
            elif self.dataset_config.domain == 'all':
                self.goals = RecommendationDataset.load_binary_file(self.dataset_config.save_all_goal_path)
                self.topics = RecommendationDataset.load_binary_file(self.dataset_config.save_all_topic_path)             
        
        print(self.goals)

        
    def pipeline(self, data_path):
        """method that employs that data pipeline including read_data, repurpose_data and progress_data
        """
        data = self.read_data(data_path=data_path)
        data = self.repurpose_dataset(data)
        if self.dataset_config.save_train_convs and 'train' in data_path:
            self.train_convs = data
        data = self.process_data(data)
        return data

    def return_infor(self):
        """function that returns information about the dataset

        Returns:
            _type_: dictionary
        """
        infor_dict = {
            "num_topics": len(self.topics),
            "num_goals": len(self.goals),
            "train_instances": len(self.train_instances),
            "dev_instances": len(self.dev_instances),
            "test_instances": len(self.test_instances)

        }
        return infor_dict

    def construct_action_mapping(self, combine=False):
        """
        method that create a action mapping, (goal, topic) ->id or goal ->id, topic -> id
        :param combine: true we combine goal, topic to from an action
        :return: a dictionary in case of comine otherwise two dictionaries
        """
        # seperating goals from topics
        if not combine:
            goal2id = {k: v for v, k in enumerate(self.goals)}
            topic2id = {k: v for v, k in enumerate(self.topics)}
            return goal2id, topic2id
        # combining goals and topics
        else:
            goal2id = itertools.product(self.goals, self.topics)
            goal2id = {k: v for v, k in enumerate(goal2id)}
            return goal2id

    def get_user_profiles(self):
        """
        method that return user profiles from the training, development and testing datasets
        :return: train_user_profiles, dev_user_profiles, test_user_profiles
        """
        train_profiles = []
        dev_profiles = []
        test_profiles = []

        # collect user profiles from the training dataset
        for instance in self.train_instances:
            profile = instance['task_background']['user_profile']
            if profile not in train_profiles:
                train_profiles.append(profile)

        # collect user profile from the development dataset
        for instance in self.dev_instances:
            profile = instance['task_background']['user_profile']
            if profile not in dev_profiles:
                dev_profiles.append(profile)

        # collect user profile from the testing dataset
        for instance in self.test_instances:
            profile = instance['task_background']['user_profile']
            if profile not in test_profiles:
                test_profiles.append(profile)

        return train_profiles, dev_profiles, test_profiles


class NegotiationDataset(Dataset):

    def __init__(self, dataset_config, **kwargs):
        """
        constructor for class negotiation datasets
        :param dataset_config: the configuration of the dataset
        :param kwargs: other keywords arguments
        """
        super().__init__(dataset_config, **kwargs)

        # common attributes for negotiation datasets
        self.goals = []
        self.save_train_convs = self.dataset_config.save_train_convs
        self.train_convs = []
        self.dev_convs = []
        self.test_convs = []

        # generate train, dev and test instances
        self.train_instances = self.pipeline(self.dataset_config.train_data_path)
        self.dev_instances = self.pipeline(self.dataset_config.dev_data_path)
        self.test_instances = self.pipeline(self.dataset_config.test_data_path)

        # self.goals = list(set(self.goals))

        # log
        self.log = self.dataset_config.log
        if self.log:
            goal_dict = defaultdict(int)
            # log goal count
            for goal in self.goals:
                goal_dict[goal] += 1

            print(f"Number of goals: {len(self.goals)}")
            print(
                f"Num train instances: {len(self.train_instances)}, Num dev instances: {len(self.dev_instances)}, Num test instances: {len(self.dev_instances)}")

            print(
                f"Num train convs: {len(self.train_convs)}, Num dev convs: {len(self.dev_convs)}, Num test convs: {len(self.test_convs)}")
            
            print("Statistics of goals:", goal_dict)

        # showing the amounts of individual goals.
        # reformat goals and topics
        self.goals = list(set(self.goals))
        self.n_goals = len(self.goals)

        # save goal and topic to file
        if self.dataset_config.save_action:
            print("Saving the goal, topic to files .......")
            NegotiationDataset.save_binary_file(self.goals, self.dataset_config.save_goal_path)

        # load goal and topic from files
        # this makes sure that every run using the same goal, topic mapping
        if self.dataset_config.load_action:
            print("Loading the goal, topic from files ......")
            self.goals = NegotiationDataset.load_binary_file(self.dataset_config.save_goal_path)

    def pipeline(self, data_path):
        """
        method that employs the data pipeline
        :param data_path: the path to the data path
        :return:
        """
        # read the dataset from file
        data = self.read_data(data_path=data_path)
        data = self.repurpose_dataset(data)
        # saving the conversations
        # the saved conversations will be used for rl training and performance evaluation
        if self.dataset_config.save_train_convs:
            # saving the training conversations
            if 'train' in data_path:
                self.train_convs = data
            # saving the development conversations
            elif 'valid' in data_path:
                self.dev_convs = data
            # saving the test conversations
            elif 'test' in data_path:
                self.test_convs = data
            else:
                raise Exception("Something is wrong here ....")
        data = self.process_data(data)
        return data

    def return_infor(self):
        """function that returns information about the dataset
        Returns:
            _type_: dictionary
        """
        infor_dict = {
            "num_goals": len(self.goals),
            "train_instances": len(self.train_instances),
            "dev_instances": len(self.dev_instances),
            "test_instances": len(self.test_instances)

        }
        return infor_dict

    def get_user_profiles(self):
        """
        method that create the user profiles for the negotiation scenario
        :return: a list of train, dev, test user profiles
        """
        # for negotiation dialogues,
        # we user the same set of user profiles for validation and testing
        # there is no user profiles for training
        train_user_profiles = []
        dev_user_profiles = []
        # create in total 300 user profiles
        for i in range(15):
            # loop over personalities
            for persona in BIG5_PERSONALITY:
                # loop over decision making styles
                for decision_type in DECISION_MAKING_STYLE:
                    dev_user_profiles.append((persona, decision_type))
        # return train user profiles
        # dev = test user profiles
        return train_user_profiles, dev_user_profiles, dev_user_profiles

    # def get_user_profiles(self):
    #     """
    #     method that get the train, dev and test user profiles for the data
    #     :return:train, dev and test user profiles
    #     """
    #     train_profiles = []
    #     dev_profiles = []
    #     test_profiles = []
    #
    #     # train user profiles
    #     for instance in self.train_instances:
    #         profile = {
    #             'seller_item_description': instance['task_background']["seller_item_description"],
    #             'seller_price': instance['task_background']['seller_price'],
    #             "buyer_price": instance['task_background']['buyer_price'],
    #             "buyer_item_description": instance['task_background']['buyer_item_description']
    #         }
    #         train_profiles.append(profile)
    #
    #     # dev user profiles
    #     for instance in self.dev_instances:
    #         profile = {
    #             'seller_item_description': instance['task_background']["seller_item_description"],
    #             'seller_price': instance['task_background']['seller_price'],
    #             "buyer_price": instance['task_background']['buyer_price'],
    #             "buyer_item_description": instance['task_background']['buyer_item_description']
    #         }
    #         dev_profiles.append(profile)
    #
    #     # test user profiles
    #     for instance in self.test_instances:
    #         profile = {
    #             'seller_item_description': instance['task_background']["seller_item_description"],
    #             'seller_price': instance['task_background']['seller_price'],
    #             "buyer_price": instance['task_background']['buyer_price'],
    #             "buyer_item_description": instance['task_background']['buyer_item_description']
    #         }
    #         test_profiles.append(profile)
    #
    #     return train_profiles, dev_profiles, test_profiles

    def construct_action_mapping(self, combine=False):
        """
        method that create a action mapping, (goal, topic) ->id or goal ->id, topic -> id
        :param combine: true we combine goal, topic to from an action
        :return: a dictionary in case of comine otherwise two dictionaries
        """
        # seperating goals from topics
        goal2id = {k: v for v, k in enumerate(self.goals)}
        return goal2id


class EmotionalSupportDataset(Dataset):

    def __init__(self, dataset_config, **kwargs):
        """
        constructor for class emotional support dataset
        :param dataset_config:
        """
        super().__init__(dataset_config, **kwargs)

        # common attributes for negotiation datasets
        self.goals = []
        self.save_train_convs = self.dataset_config.save_train_convs
        self.train_convs = []
        self.dev_convs = []
        self.test_convs = []

        # generate train, dev and test instances
        self.train_instances = self.pipeline(self.dataset_config.train_data_path)
        self.dev_instances = self.pipeline(self.dataset_config.dev_data_path)
        self.test_instances = self.pipeline(self.dataset_config.test_data_path)

        #
        self.goals = list(set(self.goals))
        # log
        self.log = self.dataset_config.log
        if self.log:
            goal_dict = defaultdict(int)
            # log goal count
            for goal in self.goals:
                goal_dict[goal] += 1

            print(f"Number of goals: {len(self.goals)}")
            print(
                f"Num train instances: {len(self.train_instances)}, Num dev instances: {len(self.dev_instances)}, Num test instances: {len(self.dev_instances)}")

            print(
                f"Num train convs: {len(self.train_convs)}, Num dev convs: {len(self.dev_convs)}, Num test convs: {len(self.test_convs)}")

        # reformat goals and topics
        self.goals = list(set(self.goals))
        self.n_goals = len(self.goals)

        # save goal and topic to file
        if self.dataset_config.save_action:
            print("Saving the goal, topic to files .......")
            EmotionalSupportDataset.save_binary_file(self.goals, self.dataset_config.save_goal_path)

        # load goal and topic from files
        # this makes sure that every run using the same goal, topic mapping
        if self.dataset_config.load_action:
            print("Loading the goal, topic from files ......")
            self.goals = EmotionalSupportDataset.load_binary_file(self.dataset_config.save_goal_path)

    def pipeline(self, data_path):
        """
        method that employs the pipeline to process the dataset
        :param data_path: the path to the dataset
        :return:
        """
        # read the dataset from file
        data = self.read_data(data_path=data_path)
        data = self.repurpose_dataset(data)
        # saving the conversations
        # the saved conversations will be used for rl training and performance evaluation
        if self.dataset_config.save_train_convs:
            # saving the training conversations
            if 'train' in data_path:
                self.train_convs = data
            # saving the development conversations
            elif 'valid' in data_path:
                self.dev_convs = data
            # saving the test conversations
            elif 'test' in data_path:
                self.test_convs = data
            else:
                raise Exception("Something is wrong here ....")
        data = self.process_data(data)
        return data

    def get_user_profiles(self):
        """
        method that create training, development and testing user profiles
        :return:
        """
        train_profiles = []
        dev_profiles = []
        test_profiles = []
        # train user profiles
        for instance in self.train_instances:
            profile = {
                "emotion_type": instance['task_background']['emotion_type'],
                'problem_type': instance['task_background']['problem_type'],
                'situation': instance['task_background']['situation']
            }
            if profile not in train_profiles:
                train_profiles.append(profile)
        # dev user profiles
        for instance in self.dev_instances:
            profile = {
                "emotion_type": instance['task_background']['emotion_type'],
                'problem_type': instance['task_background']['problem_type'],
                'situation': instance['task_background']['situation']
            }
            if profile not in dev_profiles:
                dev_profiles.append(profile)
        # test user profiles
        for instance in self.test_instances:
            profile = {
                "emotion_type": instance['task_background']['emotion_type'],
                'problem_type': instance['task_background']['problem_type'],
                'situation': instance['task_background']['situation']
            }
            if profile not in test_profiles:
                test_profiles.append(profile)

        return train_profiles, dev_profiles, test_profiles

    def return_infor(self):
        """function that returns information about the dataset
        Returns:
            _type_: dictionary
        """
        infor_dict = {
            "num_goals": len(self.goals),
            "train_instances": len(self.train_instances),
            "dev_instances": len(self.dev_instances),
            "test_instances": len(self.test_instances)

        }
        return infor_dict

    def construct_action_mapping(self, combine=False):
        """
        method that create a action mapping, (goal, topic) ->id or goal ->id, topic -> id
        :param combine: true we combine goal, topic to from an action
        :return: a dictionary in case of comine otherwise two dictionaries
        """
        # seperating goals from topics
        goal2id = {k: v for v, k in enumerate(self.goals)}
        return goal2id
