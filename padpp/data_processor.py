from collections import defaultdict
import torch
import numpy as np
import re

from base.data_processor import DataProcessorForRecommendation, DataProcessorForNegotiation, \
    DataProcessorForEmotionalSupport
from base.torch_dataset import BaseTorchDataset
from utils.game import random_weights
from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, PATH_TOKEN, TARGET, \
    CONTEXT_TOKEN, IGNORE_INDEX, SEEKER_TOKEN, SUPPORTER_TOKEN, BUYER_TOKEN, SELLER_TOKEN, PRICE_BEARING_STRATEGIES


def generate_bins(low, high, n):
    if n <= 0:
        return []
    bin_width = (high - low) / n
    bins = [(low + i * bin_width, low + (i + 1) * bin_width) for i in range(n)]
    return bins


class PADPPDataProcessorForRecommendation(DataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, n_objectives=3):
        """
        
        :param tokenizer:
        :param instance:
        :param max_sequence_length:
        :param action_to_id:
        :param n_objectives:
        :return:
        """
        # the features that we consider.
        # dialogue context, previous goals, topics and the target item
        dialogue_context = instance['dialogue_context']
        prev_paths = instance['pre_goals']
        prev_topics = instance['pre_topics']
        target = instance['task_background']['target_topic']
        target_goal = instance['task_background']['target_goal']

        # using a predefine weight vector
        if 'w' in instance:
            w = instance['w']
        # sampled the weight here
        else:
            # unfiorm weight
            x = 1 / n_objectives
            w = np.array([x for t in range(n_objectives)])

        # processing the dialogue context
        dialogue_str = ""
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += USER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += SYSTEM_TOKEN
            dialogue_str += utt['content']

        # processing the previous planned path
        path_str = ""
        for goal in prev_paths:
            path_str += GOAL_TOKEN
            path_str += goal
            # path_str += TOPIC_TOKEN
            # path_str += topic
            path_str += SEP_TOKEN

        # convert features to token ids
        input_str = f"{PATH_TOKEN}: {path_str} {TARGET}: {target_goal} {target} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        goals_to_ids, _ = action_to_id
        # we only predict the goal
        
        label = goals_to_ids[instance['goal']]
        # # the ground truth label
        # label = action_to_id[(instance['goal'], instance['topic'])]

        # compute the feature vector for the next state
        # calling this method recursively to process the features for the next dialogue state.
        # return the input_ids, label and next_ids
        if 'next_state' in instance:
            next_ids, _, _, _ = self.__call__(tokenizer, instance['next_state'], max_sequence_length, action_to_id)
            return input_ids, w, label, next_ids
        else:
            return input_ids, w, label, None


class PADPPTorchDataset(BaseTorchDataset):

    def collate_fn(self, batch):
        """
        collate function that converts a batch of data features to batched tensor
        :param batch: a batch of data features
        :return:
        """
        # dialogue context
        input_features = defaultdict(list)
        # mdp contextual feature
        mdp_input_features = defaultdict(list)
        labels = []
        next_input_features = defaultdict(list)
        # preference vectors
        weights = []
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            labels.append(instance['label'])
            weights.append(instance['w'])
            # the feature of the next state
            if instance['next_input_ids'] is not None:
                next_input_features['input_ids'].append(instance['next_input_ids'])

        # padding the input features
        # for the current state
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                # input_features[k] = torch.as_tensor(v)
                input_features[k] = torch.as_tensor(v, device=self.device)

        # the label and the preference weights
        labels = torch.LongTensor(labels).to(self.device)
        # Convert via np.array first to silence the "creating a tensor from a
        # list of numpy.ndarrays is extremely slow" warning and actually speed
        # up the conversion (single contiguous array vs. per-element copy).
        weights = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(self.device)

        if len(next_input_features) > 0:
            # padding the input features
            # for the next state
            next_input_features = self.tokenizer.pad(
                next_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.max_sequence_length
            )
            # convert features to torch tensors
            for k, v in next_input_features.items():
                if not isinstance(v, torch.Tensor):
                    # next_input_features[k] = torch.as_tensor(v)
                    next_input_features[k] = torch.as_tensor(v, device=self.device)

            new_batch = {
                "context": input_features,
                "w": weights,
                "next_state": next_input_features,
                "labels": labels,
            }
        else:
            new_batch = {
                "context": input_features,
                "w": weights,
                "labels": labels,
            }
        return new_batch

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            if not self.is_gen:
                # data processor for policy/reward training
                input_ids, weight, label, next_ids = convert_example_to_feature(self.tokenizer, instance,
                                                                                self.max_sequence_length,
                                                                                self.goal2id,
                                                                                self.n_objectives)
                new_instance = {
                    "input_ids": input_ids,
                    "w": weight,
                    "label": label,
                    "next_input_ids": next_ids

                }

            else:
                input_ids, label = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length,
                                                              self.max_target_length, self.is_test)
                new_instance = {
                    "input_ids": input_ids,
                    "label": label,
                }
            processed_instances.append(new_instance)
        return processed_instances


class PADPPDataProcessorForNegotiation(DataProcessorForNegotiation):
    """
    data processor class for the negotiation scenario
    """

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, n_objectives=3, n_bins=5):
        """
        feature function for the preference PPDPP model
        :param tokenizer: a huggingface tokenizer
        :param instance: a input instance
        :param max_sequence_length: max sequence length
        :param action_to_id:
        :return:
        """
        # the features that we consider.
        # dialogue context, previous goals, topics and the target item
        dialogue_context = instance['dialogue_context']
        prev_paths = instance['pre_goals']

        # using a predefine weight vector
        if 'w' in instance:
            w = instance['w']
        # sampled the weight here
        else:
            # unfiorm weight
            x = 1 / n_objectives
            w = np.array([x for t in range(n_objectives)])

        # processing the dialogue context
        dialogue_str = ""
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += SELLER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += BUYER_TOKEN
            dialogue_str += utt['content']

        # processing the previous planned path
        path_str = ""
        for goal in prev_paths:
            if isinstance(goal, tuple):
                goal = goal[0]
            path_str += GOAL_TOKEN
            path_str += goal
            path_str += SEP_TOKEN

        # convert features to token ids
        input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # the ground truth label
        buyer_price = instance['task_background']['buyer_price']
        seller_price = instance['task_background']['seller_price']

        # extract the proposed price
        res = instance['response']
        prices = re.findall(r"[-+]?\d*\.?\d+", res.replace(",", ""))
        prices = [x for x in prices if float(x) >= buyer_price and float(x) <= seller_price]
        if len(prices) > 0:
            proposed_price = max(prices)
        else:
            proposed_price = buyer_price
        proposed_price = float(proposed_price)

        # computing the bin label
        # quantizing the price range into n bins
        bins = generate_bins(buyer_price, seller_price, n=n_bins)
        bin_label = 0
        for i, bin in enumerate(bins):
            if proposed_price >= bin[0] and proposed_price <= bin[1]:
                bin_label = i

        # Collapse non-price strategies to bin 0. Their generated utterance does
        # not depend on the bin, so (strategy, bin>0) is a duplicate that the
        # action mask hides at inference. Keeping the SFT label at bin 0 makes
        # the supervised target consistent with the masked action space (a
        # 'deny' that happens to mention a price would otherwise be labelled at
        # bin>0 — an action the policy can never select).
        if instance['goal'] not in PRICE_BEARING_STRATEGIES:
            bin_label = 0

        label = action_to_id[(instance['goal'], bin_label)]
        
        # compute the feature vector for the next state
        # calling this method recursively to process the features for the next dialogue state.
        # return the input_ids, label and next_ids
        if 'next_state' in instance:
            next_ids, _, _, _ = self.__call__(tokenizer, instance['next_state'], max_sequence_length, action_to_id)
            return input_ids, w, label, next_ids
        else:
            return input_ids, w, label, None


class PADPPDataProcessorForEmotionalSupport(DataProcessorForEmotionalSupport):
    """
    datas processor class for the emotional support conversation
    """

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, n_objectives=3):
        """
        feature function for the preference PPDPP model
        :param tokenizer: a huggingface tokenizer
        :param instance: a input instance
        :param max_sequence_length: max sequence length
        :param action_to_id:
        :return:
        """
        # the features that we consider.
        # dialogue context, previous goals, topics and the target item
        dialogue_context = instance['dialogue_context']
        prev_paths = instance['pre_goals']

        # using a predefine weight vector
        if 'w' in instance:
            w = instance['w']
        # sampled the weight here
        else:
            # unfiorm weight
            x = 1 / n_objectives
            w = np.array([x for t in range(n_objectives)])

        # processing the dialogue context
        dialogue_str = ""
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += SEEKER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += SUPPORTER_TOKEN
            dialogue_str += utt['content']

        # processing the previous planned path
        path_str = ""
        for goal in prev_paths:
            path_str += GOAL_TOKEN
            path_str += goal
            path_str += SEP_TOKEN

        # convert features to token ids
        input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # the ground truth label
        label = action_to_id[instance['goal']]

        # compute the feature vector for the next state
        # calling this method recursively to process the features for the next dialogue state.
        # return the input_ids, label and next_ids
        if 'next_state' in instance:
            next_ids, _, _, _ = self.__call__(tokenizer, instance['next_state'], max_sequence_length, action_to_id)
            return input_ids, w, label, next_ids
        else:
            return input_ids, w, label, None
