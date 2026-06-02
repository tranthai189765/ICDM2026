import copy
import math
import os
import random
from itertools import count

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions import Categorical

from loguru import logger as loguru_logger
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from padpp.data_processor import PADPPDataProcessorForRecommendation, PADPPTorchDataset, \
    PADPPDataProcessorForNegotiation, PADPPDataProcessorForEmotionalSupport
from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger
from logger.terminal_logger import TerminalLogger
from logger.file_logger import FileLogger

from utils.game import save_conversation_for_human_evaluation


# Only these strategies USE the price bin in response generation (the generated
# utterance changes with the bin). For every other strategy the bin is ignored,
# so (strategy, bin>0) produces the exact same utterance and reward as
# (strategy, 0) — a pure duplicate. _build_action_mask keeps only the 19
# distinct actions (propose/counter x 5 bins + 9 other strategies at bin 0).
_PRICE_BEARING_STRATEGIES = {'propose', 'counter', 'final_offer'}


def _build_action_mask(action_mapping, n_actions, device):
    """Return a 1-D bool mask over the flat action space (True = selectable).

    Masks out bin-redundant duplicates: non-price strategies at bin > 0 are
    identical to their bin-0 form, so only bin 0 is kept for them; price-bearing
    strategies keep all bins.
    """
    if isinstance(action_mapping, tuple):
        action_mapping = action_mapping[0]
    mask = torch.zeros(n_actions, device=device, dtype=torch.bool)
    for (strategy, bin_idx), idx in action_mapping.items():
        if idx >= n_actions:
            continue
        if strategy in _PRICE_BEARING_STRATEGIES or bin_idx == 0:
            mask[idx] = True
    return mask


from utils.game import random_weights

from collections import deque
from collections import defaultdict


from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, ITEM_FREQ, USER_REWARD, DEAL_RATE


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    """
    compute
    :param values:
    :param rewards:
    :param dones:
    :param next_values:
    :param gamma:
    :param lambd:
    :return:
    """
    # Calculate TD errors.
    # deltas = rewards + gamma * next_values * (1 - dones)

    # Initialize gae.
    # gaes = torch.empty_like(rewards)

    gaes = []
    # Calculate gae recursively from behind.
    # gaes[-1] = deltas[-1]
    discounted_reward = 0
    for t in reversed(range(rewards.size(0))):
        # gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
        if dones[t] == 1:
            discounted_reward = 0
        discounted_reward = rewards[t] + gamma * discounted_reward
        gaes.insert(0, discounted_reward)

    gaes = torch.stack(gaes)
    return gaes


class PADPPTrainer(Trainer):

    def __init__(self, game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                 loggers, generation_method=None):
        """
        constructor for class PPDPP pipeline training
        :param game_config: the configuration of the game
        :param model_config: the configuration of the model
        :param accelerator: the accelerator
        :param game: instance of the game class
        :param model: instance of the model class
        :param offline_evaluator: instance of the offline evaluator class
        :param online_evaluator: an instance of the online evaluator class
        :param loggers: a set of loggers
        :param generation_method: an instance of the generation method
        """
        super().__init__(game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                         loggers)
        self.generation_method = generation_method

        # global steps for training the actor-critic model and the preference model
        self.ppo_global_step = 1
        self.preference_global_step = 0

        # the corresponding progress bars
        self.ppo_progress_bar = None
        self.preference_progress_bar = None

        self.model = accelerator.unwrap_model(self.model)
        self.tokenizer = self.model.tokenizer

    def process_dataset(self, dataset):
        """
        method that process the given dataset and return processed data instances
        :return: processed data instances.
        """
        return dataset.train_instances, dataset.dev_instances, dataset.test_instances

    def construct_dataloaders(self, data_instances, batch_size, goal2id, shuffle=True, num_workers=1):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param goal2id: a dictionary that map categorical goals to indexes
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :return: a instance of torch dataloader class
        """

        # the data processor for the  recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            convert_example_to_feature = PADPPDataProcessorForRecommendation
        # the data processor for the negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            convert_example_to_feature = PADPPDataProcessorForNegotiation
        # the data processor for the emotional support conversation
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            convert_example_to_feature = PADPPDataProcessorForEmotionalSupport
        else:
            raise Exception("Invalid scenario....")
        # construct the torch dataset
        torch_dataset = PADPPTorchDataset(
            tokenizer=self.tokenizer,
            instances=data_instances,
            goal2id=goal2id,
            max_sequence_length=self.model_config.max_sequence_length,
            device=self.device,
            n_objectives=self.model_config.n_objectives,
            convert_example_to_feature=convert_example_to_feature()
        )
        # construct the data loader
        dataloader = DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=torch_dataset.collate_fn,
        )
        return dataloader

    def create_criterion(self):
        """
        method that create the loss function to train the model
        :return: a torch.nn.CrossEntropyLoss object
        """
        return torch.nn.CrossEntropyLoss()

    def create_optimizer(self, model, learning_rate=1e-5):
        """
        method that create the optimizer to train the model
        :return: a torch.optim.Optimizer
        """
        modules = [model]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        loguru_logger.debug(f"Number trainable params: {count_parameters(model)}")
        return optimizer

    def create_scheduler(self, optimizer, num_warmup_steps, max_train_steps):
        """
        method that create the lr scheduler for training the model
        :param optimizer: the optimizer that we use to train the model
        :param num_warmup_steps: number of worm up steps
        :param max_train_steps: number of training steps.
        :return: a torch.optim.lr_scheduler
        """
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, max_train_steps)
        return lr_scheduler

    def train_epoch(self, data_loader, optimizer, lr_scheduler, criterion, max_train_steps):
        """
        method that trains the model on one epoch
        :param data_loader: data loader used to train the model
        :param optimizer: the optimizer used to train the model
        :param lr_scheduler:  the lr scheduler used to train the model
        :param criterion: the loss function that we use to train the model
        :param max_train_steps: the maximum number of training steps
        :return: the training loss in the current epoch
        """
        stop = False
        train_loss = []
        for step, batch in enumerate(data_loader):
            logits = self.model(batch)
            loss = criterion(logits, batch['labels']) / \
                self.model_config.gradient_accumulation_steps
            self.accelerator.backward(loss)
            # .item() detaches from the autograd graph; using float(loss) on a
            # requires_grad tensor triggers a PyTorch warning and can keep the
            # graph alive longer than needed.
            train_loss.append(loss.item())

            self.progress_bar.update(1)
            self.global_step += 1

            # optim step
            if step % self.model_config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                if self.model_config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.model_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if self.global_step >= max_train_steps:
                stop = True
                break

        # compute average train loss
        train_loss = np.mean(train_loss) * \
            self.model_config.gradient_accumulation_steps
        return train_loss, stop

    def eval_epoch(self, data_loader, criterion):
        """
        method that evaluates the model on the validation set.
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :return: evaluation loss
        """
        dev_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
                with torch.no_grad():
                    logits = self.model(batch)
                    loss = criterion(logits, batch['labels'])
                    self.offline_evaluator.record(logits, batch['labels'])
                    dev_loss.append(loss.item())

        dev_loss = np.mean(dev_loss) * \
            self.model_config.gradient_accumulation_steps
        self.accelerator.wait_for_everyone()
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss
        return results

    def train_sft(self, dataset, device=None):
        """
        method that train the model in an supervised tuning manner
        :param device: the device we use to train the model
        :return: None
        """

        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(
            dataset)

        # construct the goal, topic mapping
        action_mapping = dataset.construct_action_mapping(
            combine=self.model_config.combined_action)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances,
                                                  batch_size=self.model_config.per_device_train_batch_size,
                                                  goal2id=action_mapping,
                                                  shuffle=True, num_workers=self.model_config.num_workers)

        dev_loader = self.construct_dataloaders(dev_instances, batch_size=self.model_config.per_device_eval_batch_size,
                                                goal2id=action_mapping,
                                                shuffle=False, num_workers=self.model_config.num_workers)

        best_loss = math.inf
        # create the optimizer
        optimizer = self.create_optimizer(
            self.model, self.model_config.learning_rate)

        # prepare the model
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_loader)

        # compute the maximum number of training steps
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.model_config.gradient_accumulation_steps)
        max_train_steps = self.model_config.num_train_epochs * num_update_steps_per_epoch

        # create the learning rate scheduler
        lr_scheduler = self.create_scheduler(
            optimizer, self.model_config.num_warmup_steps, max_train_steps)

        # create the loss function
        self.criterion = self.create_criterion()

        # progress bar
        self.progress_bar = tqdm(
            range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        # train the model
        self.model.to(device)
        for epoch in range(self.model_config.num_train_epochs):
            self.model.train()

            # reset the offline evaluator before each training epoch
            self.offline_evaluator.reset()

            # train the model
            train_loss, stop = self.train_epoch(data_loader=train_loader, optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                criterion=self.criterion,
                                                max_train_steps=max_train_steps
                                                )
            # evaluate the performance
            results = self.eval_epoch(dev_loader, self.criterion)

            # logging the results
            for logger in self.loggers:
                logger.record(results, epoch + 1)

            # saving the model if needed
            # for generation, we use the loss as the saving criterion
            if results['loss'] < best_loss:
                loguru_logger.info(
                    "Performance improved. Saving the model .....")
                best_loss = results['loss']

                # pretrained sft for recommendation
                if self.game_config.name == RECOMMENDATION:
                    file_path = os.path.join(
                        self.model_config.saved_dir, f"model_{self.model_config.domain}.pth")
                # pretrained sft for negotiation
                elif self.game_config.name == NEGOTIATION:
                    file_path = os.path.join(
                        self.model_config.saved_dir, f"model.pth")
                self.save_model(file_path)

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def train_rl_step(self, buffer, action_mapping, optimizer, scheduler):
        """
        method that perform the rl training part
        :param buffer: the buffer that stores data instances for training
        :param action_mapping: a dictionary that maps goal, topic to ids
        :param optimizer: the optimizer for training the q network
        :param scheduler: the scheduler for training the q network
        :return:
        """
        # # freezing the preference block
        # # only update the policy part
        # self.model.manipulate_gradient_update(is_preference_block=True, flag=True)
        # self.model.manipulate_gradient_update(is_preference_block=False, flag=True)
        # print(self.model.critic.requires_grad_)

        # progress bar
        progress_bar = tqdm(range(self.model_config.num_train_q_network_epochs),
                            disable=not self.accelerator.is_local_main_process)

        # loop over the number of ppo epochs
        mean_loss = []

        # cloning model used for KL regularization Q learning
        self.kl_model = copy.deepcopy(self.model)

        # constructing a memory buffer
        # target_network
        # used in deep double Q learning
        self.target_model = copy.deepcopy(self.model)

        # define a variable capturing ids of previous batch data.
        for i in tqdm(range(self.model_config.num_train_q_network_epochs)):

            # otherwise we sample a batch of data from prev_step to next_step
            # to train the model
            indices = np.random.choice(
                len(buffer), self.model_config.train_rl_batch_size)
            batch_instances = [buffer[i] for i in indices.tolist()]

            # states
            states = [x[0] for x in batch_instances]

            # action
            if isinstance(action_mapping, tuple):
                batch_act = [action_mapping[0][state['act']]
                             for state in states]
            else:
                batch_act = [action_mapping[state['act']] for state in states]

            batch_act = torch.LongTensor(batch_act).to(self.device)

            # get the rewards
            rewards = [x[1] for x in batch_instances]
            rewards = torch.cat(rewards, dim=0)

            # get the done flags
            batch_done = [x[3] for x in batch_instances]
            batch_done = torch.Tensor(batch_done).to(self.device)

            train_loader = self.construct_dataloaders(states,
                                                      batch_size=self.model_config.train_rl_batch_size,
                                                      shuffle=False,
                                                      goal2id=action_mapping,
                                                      num_workers=self.model_config.num_workers)

            # loop over the train loader
            for batch in train_loader:
                # sample a batch of preference weights
                # sampled preferences from the memory buffer
                # sampled_preferences = random.choices(self.preference_memory, k=self.model_config.n_preferences)
                # sampled_preferences = torch.Tensor(sampled_preferences).to(self.device)
                # if we draw preferences from a uniform distribution
                # w_{current}
                # NOTE: we start the training process by training the model with extreme preferences
                # e.g: [1, 0], [0, 1 ], [0.5, 0.5]
                # if self.ppo_global_step < self.model_config.n_warmup_epochs:
                #     sampled_preferences = list(self.model_config.obj_to_weight.values())
                #     sampled_preferences = [x for x in sampled_preferences if x is not None]
                #     sampled_preferences = np.array(sampled_preferences)
                # # NOTE: if we draw preferences from a uniform distribution
                # else:

                sampled_preferences = random_weights(
                    self.model_config.n_objectives, n=self.model_config.n_preferences)
                self.memory_buffer.extend(sampled_preferences)

                # we add the sampled preferences to the memory buffer
                # self.preference_memory.extend(sampled_preferences)
                sampled_preferences = torch.Tensor(sampled_preferences).to(
                    self.device).requires_grad_(False)

                # computing feature representations
                state, next_state, w_embedding = self.model.compute_state_resp(
                    batch, sampled_preferences)
                bs = state.size(0)

                # n_preferences = self.model_config.n_preferences
                n_preferences = len(sampled_preferences)

                # extend state, next state, w_embedding
                w_embedding = w_embedding.repeat(
                    1, bs).view(-1, w_embedding.size(-1))

                # state
                state = state.repeat(n_preferences, 1).view(-1, state.size(-1))

                # next state
                next_state = next_state.repeat(
                    n_preferences, 1).view(-1, next_state.size(-1))

                # action
                action = batch_act.repeat(
                    sampled_preferences.size(0), 1).view(-1)

                # construct the feature vector
                feature = torch.cat([
                    state,
                    w_embedding
                ], dim=-1)

                # computing the logit using the actor network
                # Q(s,a,w)
                Q = self.model.actor(feature)

                # computing the logit using the actor network
                # Q(s,a,w)
                action_size = Q.view(
                    Q.size(0), -1, self.model_config.n_objectives).size(1)
                Q = Q.view(Q.size(0), -1, self.model_config.n_objectives)

                # shape = [bs, n_tasks, n_objectives]
                Q1 = Q.gather(1, action.view(-1, 1, 1).expand(Q.size(0), 1, self.model_config.n_objectives)).view(-1,
                                                                                                                  self.model_config.n_objectives)

                # NOTE: Successor Feature Training
                # Policy Improvement
                with torch.no_grad():

                    self.model.eval()
                    next_feature = torch.cat([
                        next_state,
                        w_embedding
                    ], dim=-1)

                    # computing the logit using the actor network
                    # Q(s',a,w)
                    Q_next = self.target_model.actor(next_feature).detach()
                    Q_next = Q_next.view(-1, action_size,
                                         self.model_config.n_objectives)

                    # calculate repeated sample_preferences.
                    repeated_sample_preference = sampled_preferences.repeat(
                        1, bs).view(-1, self.model_config.n_objectives)
                    repeated_sample_preference = repeated_sample_preference.unsqueeze(
                        1).repeat(1, action_size, 1)

                    # cosine distance
                    cosim = torch.nn.functional.cosine_similarity(
                        repeated_sample_preference, Q_next, dim=-1)

                    # computing the logit using the actor network
                    scalarized_scores = \
                        (repeated_sample_preference *
                         Q_next).sum(dim=-1).view(-1, action_size)

                    scalarized_scores = scalarized_scores * cosim
                    # get the action-value function of the best action.
                    idx = scalarized_scores.max(1)[1]
                    Q_next_target = Q_next.gather(1, idx.view(-1, 1, 1).expand(idx.size(0),
                                                                               1,
                                                                               self.model_config.n_objectives)).squeeze()

                # NOTE: Generalized Policy Imrpovement
                # compute target Q(s',a',w) here
                # no gradient step here
                with torch.no_grad():
                    self.model.eval()
                    # sample update preferences from the memory buffer
                    # w_{prev}
                    if len(self.memory_buffer) < self.model_config.n_preferences:
                        prev_sampled_preferences = sampled_preferences
                    # sampled previous learned preferences from the memory
                    else:
                        prev_sampled_preferences = random.choices(
                            self.memory_buffer, k=self.model_config.n_preferences)

                    prev_sampled_preferences = torch.Tensor(
                        prev_sampled_preferences).to(self.device).requires_grad_(False)
                    # update the memory buffer
                    # compute the objective embedding for previous updated preferences
                    _, _, prev_w_embedding = self.model.compute_state_resp(
                        batch, prev_sampled_preferences)
                    prev_w_embedding = prev_w_embedding.repeat(
                        1, bs).view(-1, prev_w_embedding.size(-1))

                    # construct the next feature
                    # concatenate next state and prev_w_embedding
                    next_feature = torch.cat([
                        next_state,
                        prev_w_embedding
                    ], dim=-1)

                    Q_prime = self.target_model.actor(next_feature).detach()
                    Q_prime = Q_prime.view(-1, action_size,
                                           self.model_config.n_objectives)

                    tmp_Q_prime = self.model.actor(next_feature).detach()
                    tmp_Q_prime = tmp_Q_prime.view(-1,
                                                   self.model_config.n_objectives)

                    # bs, n_sample * n_action, 2
                    # 1, 2, n_sample
                    repeated_sample_preference = sampled_preferences.repeat(
                        bs, 1)
                    repeated_sample_preference = repeated_sample_preference.unsqueeze(2).repeat(1, action_size,
                                                                                                1).view(-1,
                                                                                                        self.model_config.n_objectives)

                    # computing the cosine similarity between the sampled preferences and the computed Q target
                    # encouraging similar q targets and preferences
                    co_sim = torch.nn.functional.cosine_similarity(repeated_sample_preference, tmp_Q_prime).view(-1,
                                                                                                                 action_size)
                    # computing scalarized Q targets
                    scalarized_scores = \
                        torch.bmm(repeated_sample_preference.unsqueeze(1), tmp_Q_prime.unsqueeze(2)).view(-1,
                                                                                                          action_size)
                    # convex envelope Q ids
                    if self.game_config.name == NEGOTIATION:
                        idx = (co_sim * scalarized_scores).max(1)[1]

                    elif self.game_config.name == RECOMMENDATION:
                        idx = scalarized_scores.max(1)[1]

                    # Bs, n_sample, n_sample, 2 x Bs, n_sample, n_sample, n_action, 2
                    # scalarized_Q_prime = torch.matmul(repeated_sample_preference, Q_prime.permute(0, 1, 3))
                    # scalarized_Q_prime = scalarized_Q_prime.view(state.shape[0], sampled_preferences.shape[0], -1)
                    # print(idx)
                    # collect target Q
                    Q2 = Q_prime.gather(1, idx.view(-1, 1, 1).expand(idx.size(0),
                                                                     1,
                                                                     self.model_config.n_objectives)).squeeze()

                # compute target_q
                # compute the target TD error.
                # this is used to update the value function of the current policies.
                w_batch = sampled_preferences.repeat(
                    1, bs).view(-1, self.model_config.n_objectives)
                rewards = rewards.repeat(sampled_preferences.size(0), 1).view(-1,
                                                                              self.model_config.n_objectives)

                dones = batch_done.repeat(sampled_preferences.size(0), 1).view(-1,
                                                                               1)

                wQ = torch.bmm(w_batch.unsqueeze(1), Q1.unsqueeze(2)).squeeze()

                # computing TD targets
                # use standard policy improvement
                if not self.model_config.use_gpi:
                    # TD target
                    TQ = rewards + self.model_config.gamma * \
                        (1 - dones) * Q_next_target
                    # NOTE: PI TD error
                    wTQ_next = torch.bmm(w_batch.unsqueeze(
                        1), TQ.unsqueeze(2)).squeeze()
                    loss = F.mse_loss(
                        wQ.view(-1), wTQ_next.view(-1), reduction='mean')
                # use GPI-based policy improvement
                else:
                    # TD target
                    TQ = rewards + self.model_config.gamma * (1 - dones) * Q2
                    # scalarization
                    # wQ = torch.bmm(w_batch.unsqueeze(1), Q1.unsqueeze(2)).squeeze()
                    wTQ = torch.bmm(w_batch.unsqueeze(
                        1), TQ.unsqueeze(2)).squeeze()

                    # NOTE: GPI TD error
                    loss = self.model_config.alpha * \
                        F.mse_loss(wQ.view(-1), wTQ.view(-1), reduction='mean')
                    loss += (1 - self.model_config.alpha) * \
                        F.mse_loss(Q1.view(-1), TQ.view(-1), reduction='mean')

                # update the parameters of actor and critic
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update the learning rate
                scheduler.step()

                # update the target network
                if (i + 1) % 1 == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                # collect the mean actor and critic loss
                mean_loss.append(loss)
                progress_bar.update(1)

        # compute the mean actor and critic loss
        mean_loss = sum(mean_loss) / len(mean_loss)

        results = {
            "loss": mean_loss
        }

        loguru_logger.info(f"[RL] step={self.ppo_global_step} loss={mean_loss.item():.6f}")

        # log the results to terminal or file
        for logger in self.loggers:
            logger.record(results, self.ppo_global_step)

        self.ppo_global_step += 1

    def train_rlt(self, cases, dev_cases=None, device=None, simulators=None, dev_simulators=None, action_mapping=None):
        """
        method that train the model in a reinforcement learning manner
        :param cases: a list of situations, target_items, e.g....
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param action_mapping: a dictionary that map (goa, topic) to index
        :return: None
        """
        self.model = self.accelerator.unwrap_model(self.model)

        # compute the total actor and critic training step
        max_training_steps = self.model_config.num_train_rl_epochs * int(
            self.model_config.buffer_length // self.model_config.train_rl_batch_size)

        # create the optimizer for actor and critic
        loguru_logger.debug(f"Actor Learning Rate: {self.model_config.actor_learning_rate}")
        optimizer = self.create_optimizer(self.model,
                                          self.model_config.actor_learning_rate)

        # create the actor and critic lr schedulers
        scheduler = self.create_scheduler(optimizer,
                                          num_warmup_steps=self.model_config.warmup_steps,
                                          max_train_steps=max_training_steps
                                          )

        # best metric
        best_metric = - math.inf
        # using accelerator to prepare the model and optimizer
        # self.model = self.accelerator.prepare(self.model)
        self.model.to(self.device)

        # create a buffer to store experienced interactions for ppo training
        buffer = deque(maxlen=self.model_config.buffer_length)

        # create a memory buffer to record past trained preferences
        self.memory_buffer = deque(
            maxlen=self.model_config.preference_buffer_length)

        # loop for the number of epoch
        # number of training episode / n_episode each epoch
        for train_step in range(0, self.model_config.num_train_rl_epochs + 1):

            self.model.train()

            # using the current policy model
            # this is the execution phase in the algorithm.
            for i_episode in tqdm(range(self.model_config.sampled_times), desc='sampling'):
                # randomly sample one case
                # sample 1 item
                case = np.random.choice(cases)
                # # NOTE: we inititialy start the training with extreme corner weights
                # if self.ppo_global_step <= self.model_config.n_warmup_epochs:
                #     objs = list(self.model_config.obj_to_weight.keys())
                #     objs.remove("uniform")
                #     choice = np.random.randint(low=0, high=len(objs))
                #     w = self.model_config.obj_to_weight[objs[choice]]
                # # after that we train the model with randomly sampled weights
                # else:
                #     #     # sample a preference vector using the trained preference params
                #     #     # this step is not differentiable
                w = random_weights(self.model_config.n_objectives)

                # randomly sample persona information
                # i.e a random user is sampled
                simulator = np.random.choice(simulators)
                loguru_logger.info(
                    '\n================New Episode:{}===================='.format(i_episode))

                # reset the game state
                # sampled initial state s(0)
                # construct a new game state based on the given case and the current simulator
                state = self.game.reset(case, simulator)

                # assign the preference weight vector
                state['w'] = w

                # store the preference to the mem buffer
                # self.mem_buffer.append(w)

                # recommendation scenario
                if self.game_config.name == RECOMMENDATION:
                    loguru_logger.info(
                        f"[Target Item]: {state['task_background']['target_topic']}")
                    loguru_logger.info(
                        f"[Target Goal]: {state['task_background']['target_goal']}")

                # negotiation scenario
                elif self.game_config.name == NEGOTIATION:
                    loguru_logger.info(
                        f"[Item Name]: {state['task_background']['item_name']}")
                    loguru_logger.info(
                        f"[Seller Desired Price]: {state['task_background']['seller_price']}")
                    loguru_logger.info(
                        f"[Buyer Desired Price]: {state['task_background']['buyer_price']}")

                loguru_logger.info(
                    f"[System]: {state['dialogue_context'][0]['content']}")
                loguru_logger.info(
                    f"[USER]: {state['dialogue_context'][1]['content']}")

                # more than 1 objectives, therefore the reward is a vector
                done = False

                # trajectories to store simulated interactions
                trajectory = []
                rewards = []
                loguru_logger.info(f"Objective Weight: [{w}]")

                # interactive simulations
                # executing step: c
                for t in count():  # user dialog

                    # old state
                    old_state = copy.deepcopy(state)

                    # predict the action using the sampled w and the trained model
                    # a ~ \pi(a|s,w)
                    action, _, _ = self.predict(state, torch.FloatTensor(w).to(self.device), action_mapping,
                                                is_computing_reward=False, use_gpi=False)

                    # employing the action to observe the next state
                    # and the corresponding rewards
                    state, reward, done, _ = self.game.step(
                        state, action, self.generation_method, simulator)

                    # storing the reward
                    # this is the reward obtain via Monte-Carlo sampling
                    reward = torch.tensor(
                        [reward], device=device, dtype=torch.float)
                    rewards.append(reward)

                    # collect information
                    # s, s', a, a', done to compute the TD error loss
                    old_state['next_state'] = copy.deepcopy(state)
                    old_state['act'] = action
                    old_state['done'] = 1 if done in (1, -1) else done
                    trajectory.append(old_state)

                    # storing the experiences to the ppo buffer
                    # each experience is (state, next_state, r, log_prob)
                    if train_step >= 0:
                        # computing the estimated reward function
                        # _, _, estimated_reward = self.predict(old_state, torch.FloatTensor(w).to(self.device),
                        #                                       action_mapping, is_computing_reward=True, use_gpi=False)

                        # a failed case
                        # but for training if only consider 0, 1 for on-going or terminated conversation.
                        if done == -1:
                            done = 1

                        buffer.append([
                            # state, next_state, reward, log_prob, done
                            # old_state, estimated_reward, log_prob, done
                            old_state, reward, 1, abs(done)
                        ])

                    if done:
                        break

                # # calculating the accumulated return for one episode with the current simulator
                # accumulated_return = 0
                # for i, reward in enumerate(rewards):
                #     accumulated_return = accumulated_return + \
                #         (self.model_config.gamma) ** i * reward

            # update the USFA
            # updating the parameters of universal successor features
            # loguru_logger.warning(f"Global epoch: {train_step}, Training the preference model ....")
            # self.train_preference(preference_buffer, action_mapping, preference_optimizer, preference_scheduler)
            # update the ppo
            # starting from the second iterations
            if train_step >= 0:
                # make sure the numbef or dev items equal to the number of dev simulators
                assert len(dev_cases) == len(dev_simulators)

                # update the policy model
                loguru_logger.warning(
                    f"Global epoch: {train_step}, Training the Q network ....")
                self.train_rl_step(buffer,
                                   action_mapping,
                                   optimizer=optimizer,
                                   scheduler=scheduler
                                   )

                # # update the ppo buffer length
                # # get the most recent experiences
                # if len(buffer) >= self.model_config.buffer_length:
                #     buffer = buffer[-self.model_config.buffer_length:]

                # evaluate the results on the dev set
                # computationally
                if dev_cases is not None and (train_step + 1) % 10 == 0:
                    # uniform evaluation
                    if self.game_config.name == RECOMMENDATION:
                        # uniform evaluation
                        loguru_logger.warning(
                            f"Global epoch: {train_step}, Uniform Weight Evaluation ....")
                        results = self.online_test(dev_cases, device, dev_simulators, action_mapping, stage='dev',
                                                   obj='uniform')

                    elif self.game_config.name == NEGOTIATION:
                        loguru_logger.warning(
                            f"Global epoch: {train_step}, Uniform Weight Evaluation ....")
                        results = self.online_test(dev_cases, device, dev_simulators, action_mapping, stage='dev',
                                                   obj='uniform')

                    # logging the results
                    for logger in self.loggers:
                        logger.record(results, self.ppo_global_step)

                    # saving the model checkpoitns
                    # we choose the checkpoint with the best SR
                    if results[SUCCESS_RATE] > best_metric:
                        best_metric = results[SUCCESS_RATE]

                        # saving the rl fine-tuning model
                        # save the rl pretrained model
                        loguru_logger.info(
                            "Saving the RL fine-tuned model .....")
                        if self.game_config.name == RECOMMENDATION:
                            file_path = os.path.join(
                                self.model_config.saved_dir, f"rl_model_{self.model_config.domain}.pth")

                        # pretrained sft for negotiation
                        elif self.game_config.name == NEGOTIATION:
                            file_path = os.path.join(
                                self.model_config.saved_dir, f"rl_model.pth")
                        self.save_model(file_path)

        loguru_logger.info(
            "Saving the last checkpoint of the RL fine-tuned model .....")
        file_path = os.path.join(
            self.model_config.saved_dir, "rl_model_last.pth")
        self.save_model(file_path)

        # return data for preference training
        return None

    def predict(self, instance, w, action_mapping=None, is_test=False, is_computing_reward=False, use_gpi=True, n=10):
        """
        method that predict the action given an input instance
        :param instance: the given input instance
        :param action_mapping: a dictionary that maps action to index
        :param is_test: True if it is inference time else False
        :param is_computing_reward: True if we are computing the estimated reward function
        :return: an predicted action
        """
        # create the inverse action mapping
        if isinstance(action_mapping, tuple):
            # create the inverse action mapping
            inverse_action_mapping = {
                v: k for k, v in action_mapping[0].items()}
        else:
            inverse_action_mapping = {v: k for k, v in action_mapping.items()}

        # create the data loader
        data_loader = self.construct_dataloaders([instance],
                                                 batch_size=1,
                                                 goal2id=action_mapping,
                                                 shuffle=True,
                                                 num_workers=self.model_config.num_workers)

        # # prepare the dataloader and model using accelerator
        # data_Loader, self.model = self.accelerator.prepare(data_loader, self.model)

        # evaluation phase
        # computing the estimated reward
        if is_computing_reward:
            self.model.eval()
            # make sure no gradient pass through here.
            with torch.no_grad():
                # predict the action
                for batch in data_loader:
                    # predict the action
                    # we also compute the reward using the next state
                    reward = None
                    if is_computing_reward:
                        # computing the estimated reward
                        reward = self.model.compute_features(batch)
                        action = None
                        log_prob = None
        else:
            # predict the action
            for batch in data_loader:
                w_gpi = random_weights(self.model_config.n_objectives, n=n)
                w_gpi = torch.Tensor(w_gpi).unsqueeze(1).to(self.device)

                # compute the state representation
                reward = None
                state_resp, _, w_embedding = self.model.compute_state_resp(
                    batch, w)
                # feature = self.model.projector(state_resp)
                feature = state_resp
                # construct the feature vector
                feature = torch.cat(
                    [feature, w_embedding.unsqueeze(0)], dim=-1)
                # computing the logit using the actor network
                # Q(s,a,w)

                logits = self.model.actor(feature)
                logits = logits.view(1, -1, self.model_config.n_objectives)
                # applying GPI
                if use_gpi:
                    # sample a batch of w
                    logits = logits.repeat(n, 1, 1)
                    logits = torch.bmm(logits, w_gpi.permute(0, 2, 1))
                    logits = logits.max(dim=0)[0]
                    logits = logits.permute(1, 0)
                # Inference
                else:
                    logits = torch.bmm(logits, batch['w'].unsqueeze(
                        1).permute(0, 2, 1)).squeeze(-1)
                    loguru_logger.debug(f"w={batch['w']} logits={logits}")

                # Mask bin-redundant duplicate actions (non-price strategies at
                # bin > 0 produce the same utterance/reward as bin 0). Keeps the
                # 19 distinct actions. Toggle with model_config.mask_redundant_actions.
                if getattr(self.model_config, 'mask_redundant_actions', True):
                    action_mask = _build_action_mask(
                        action_mapping, logits.size(-1), logits.device)
                    logits = logits.masked_fill(~action_mask, float('-inf'))

                action, log_prob = self.select_action(logits, is_test=is_test)
                action = inverse_action_mapping[action]
                loguru_logger.debug(f"action={action} log_prob={log_prob}")

        # return action and log prob
        return action, log_prob, reward

    def select_action(self, logits, is_test=True, eps=0.1):
        """
        method that select an action from the output logits
        :param logits: the logits output by a model
        :param is_test: True if it is inference time else false
        """
        # convert logits to probabilities
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)

        # compute policy with policy model.
        if is_test:
            # greedy sampling
            action = logits.argmax()
            return action.item(), None
        else:
            # epsilon greedy. self.current_eps (set by the curriculum/RL loop)
            # overrides the fixed default so exploration can be annealed across
            # epochs; falls back to the eps argument when no schedule is set.
            cur = getattr(self, 'current_eps', None)
            if cur is not None:
                eps = cur
            p = np.random.random()
            if p < eps:
                action = np.random.randint(0, logits.size(-1))
            else:
                action = logits.argmax().item()
            return action, None

    def test(self, dataset):
        """
        method that evaluate the performance of the model on the test set
        :param dataset: the dataset that we want to evaluate the model performance.
        :return: the results on the test setz
        """

        # create the action mapping
        action_mapping = dataset.construct_action_mapping(
            combine=self.model_config.combined_action)

        # create the data loader
        test_loader = self.construct_dataloaders(dataset.test_instances,
                                                 batch_size=self.model_config.per_device_eval_batch_size,
                                                 goal2id=action_mapping,
                                                 shuffle=False,
                                                 num_workers=self.model_config.num_workers)

        # get the model performance on the test set.
        results = self.eval_epoch(test_loader, self.create_criterion())
        return results

    def online_test(self, cases, device=None, simulators=None, action_mapping=None, stage='dev', obj='uniform'):
        """
        method that evaluate the rl-finetuned model on the test set
        :param cases: a list of situations, target_items, e.g....
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param action_mapping: a dictionary that map (goa, topic) to index
        :return:
        """

        loguru_logger.warning(f"Online Testing on Target Item Set ......")
        loguru_logger.warning(
            f"Num Target Items: {len(cases)}, Num Simulators: {len(simulators)}")

        # a default dict to store turn-level results
        turn_level_results = defaultdict(list)

        # a default dict to store turn-level strategies
        strategy_statistics = defaultdict(int)

        # success rate and average number of conversation turns.
        SR, total_reward = 0., 0.

        # avg_sub_reward, avg_obj_reward, avg_turn_reward
        # avg_sub_reward, avg_obj_reward, avg_turn_reward = 0.0, 0.0, 0.0

        # a tensor which is used to store the loss
        # select a particular simulator
        # and promote items to this simulator
        self.model.to(device)
        # simulator = simulators[0]

        # conversations
        convs = []

        # loop over the item set
        for idx, (case, simulator) in tqdm(enumerate(list(zip(cases[:10], simulators)))):

            loguru_logger.info(
                '\n================Item Num:{}===================='.format(idx))

            # IMPORTANT: the preference weight determine which aspects should be prioritied during inference
            # e.g: [0.7,0.2,0.1]: this preference favours the success rate
            # e.g: [0.2,0.7,0.1]: this preference favours other utitilies, such as fairness, item_freq or toxicity
            # e.g: [0.2, 0.2, 0.7]: this preference favours the avg turn metric.
            if stage == 'test':
                # specified objective weight
                # for the adatability experiment
                if self.model_config.objective_weight is not None:
                    w = np.array(self.model_config.objective_weight)
                # specified objective
                else:
                    # uniform sampled weight
                    if self.model_config.prioritized_objective == "uniform":
                        w = random_weights(
                            self.model_config.n_objectives, dist="uniform")
                    # evaluate using a given objective weight
                    # set the weight for the prioritized objective
                    # e.g: if the prioritized objective is sl_ratio then the corresponding weight is [1.0, 0.0, 0.0]
                    else:
                        w = np.array(
                            self.model_config.obj_to_weight[self.model_config.prioritized_objective.strip()])

            # during development, we randomly sample a weight vector
            elif stage == 'dev':
                # unfirm sampling
                if obj == 'uniform':
                    w = random_weights(
                        self.model_config.n_objectives, dist="uniform")
                else:
                    w = np.array(self.model_config.obj_to_weight[obj.strip()])

            # reset the game state
            # construct a new game state based on the given case and the current simulator
            state = self.game.reset(case, simulator)

            loguru_logger.info(f"Objective Weight: [{w}]")

            # recommendation scenario
            if self.game_config.name == RECOMMENDATION:
                loguru_logger.info(
                    f"[Target Item]: {state['task_background']['target_topic']}")
                loguru_logger.info(
                    f"[Target Goal]: {state['task_background']['target_goal']}")

            # negotiation scenario
            elif self.game_config.name == NEGOTIATION:
                loguru_logger.info(
                    f"[Item Name]: {state['task_background']['item_name']}")
                loguru_logger.info(
                    f"[Seller Desired Price]: {state['task_background']['seller_price']}")
                loguru_logger.info(
                    f"[Buyer Desired Price]: {state['task_background']['buyer_price']}")

            loguru_logger.info(
                f"[System]: {state['dialogue_context'][0]['content']}")
            loguru_logger.info(
                f"[USER]: {state['dialogue_context'][1]['content']}")

            # episode-level reward
            # more than 1 objectives, therefore the reward is a vector
            epi_reward = []
            done = False

            # flag for checking if the target is mentioned during the conversation
            o_flag = False

            # assign the preference weight vector
            state['w'] = w

            # a flag to check if the conversation is successful
            # for computing the success rate
            is_successful = False
            conv_turn = 0

            # interactive simulation
            prev_reward = 0

            # execution phase
            for t in count():  # user  dialog

                # predict the action
                action, log_prob, _ = self.predict(state, torch.FloatTensor(w).to(self.device), action_mapping,
                                                   is_computing_reward=False, use_gpi=False, is_test=True)

                #  count the number of each action
                strategy_statistics[action] += 1

                # employing the action to observe the next state
                # and the corresponding rewards
                state, reward, done, o_done = self.game.step(
                    state, action, self.generation_method, simulator)

                # storing the reward
                # this is reward obtained using monte-carlo sampling
                reward = torch.tensor(
                    [reward], device=device, dtype=torch.float)
                epi_reward.append(reward)

                # recommendation: check if target is mentioned during the conversation
                # negotiation: check if there is a deal between user and system
                # emotional support conversation: ......
                if o_done == 1:
                    o_flag = True

                # current turn reward + past reward
                tmp_reward = (reward + prev_reward).tolist()[0]

                # cummulated reward
                turn_level_results[t].append(tmp_reward)
                prev_reward = reward

                # evaluate the outcome of the conversation
                if done:
                    # successful case
                    # if the llm_reward is greater than epsilon.
                    # recommendation: check if target is mentioned during the conversation.
                    # negotiation: check if there is a deal between user and system.
                    # emotional support conversation: check if the mental problem of the seeker is resolved.
                    if done == 1 and o_flag:
                        # increase the SR
                        SR += 1
                        is_successful = True
                    conv_turn = len(state['dialogue_context'])
                    # total_reward += epi_reward
                    break

            # store the conversations
            convs.append(state)

            # compute the cumulative reward at each turn
            # cumsum_turn_results = np.cumsum(epi_reward, axis=0)

            # construct the epi reward tensor
            # a tensor storing reward obtained during the conversation
            epi_reward = torch.cat(epi_reward, dim=0)

            # exception case, length of the conversation is 1
            if len(epi_reward.shape) == 1:
                epi_reward.unsqueeze(0)

            # objective-based epi reward
            objective_based_reward = epi_reward.sum(dim=0)

            # update the online evaluator
            # recommendation scenario
            if self.game_config.name == RECOMMENDATION:

                # three objectives
                # i.e user reward, item_freq, turn_reward
                if len(objective_based_reward) == 3:
                    item_freq = objective_based_reward[-2].item()
                    # turn_reward = objective_based_reward[-1].item()

                # two objectives
                # user_reward, item_freq
                elif len(objective_based_reward) == 2:
                    item_freq = objective_based_reward[-1].item()
                    # turn_reward = -1

                # for recommendation
                # the first objective is subjective reward
                user_reward = objective_based_reward[0].item()

                # the second objective is target item frequency
                # dialogue-level results
                self.online_evaluator.record(
                    {
                        # use to compute the success rate and avg conv turn.
                        SUCCESS_RATE: int(is_successful),
                        AVG_TURN: [conv_turn, 0],

                        USER_REWARD: user_reward,
                        # rewards on objectives of interest
                        # target item frequency for recommendation
                        ITEM_FREQ: item_freq

                    }
                )
            # negotiation scenario
            if self.game_config.name == NEGOTIATION:
                # for the negotiation
                # the first objective is sl_ratio
                # the second objective is fairness
                epi_reward = epi_reward.mean(dim=0)
                sl_ratio_reward = epi_reward[0].item()
                fairness_reward = epi_reward[1].item()

                # i.e shoud be deal reward
                deal_rate_reward = epi_reward[-1].item()

                # three objectives
                # i.e user reward, item_freq, turn_reward
                if len(objective_based_reward) == 2:
                    turn_reward = -1

                # we do not consider sl_ratio and fairness score for failed conversations
                # if not is_successful:
                #     sl_ratio_reward = 0
                #     fairness_reward = 0

                # dialogue-level results
                self.online_evaluator.record(
                    {
                        # use to compute the success rate and avg conv turn.
                        SUCCESS_RATE: is_successful,
                        AVG_TURN: [conv_turn, 0],

                        # deal rate
                        DEAL_RATE: deal_rate_reward,

                        # rewards on objectives of interest
                        # this can be used to compute the SL_ratio, Fairness Score for negotiation
                        SL_RATIO: sl_ratio_reward,
                        FAIRNESS: fairness_reward
                    }
                )

            # emotional support conversation
            elif self.game_config.name == EMOTIONAL_SUPPORT:
                # the first objective is sl_ratio
                toxicity = objective_based_reward[1].item()
                # user reward
                user_reward = objective_based_reward[0].item(
                ) / epi_reward.shape[0]
                # the second objective is toxicity
                self.online_evaluator.record(
                    {
                        # use to compute the success rate and avg conv turn.
                        SUCCESS_RATE: is_successful,
                        AVG_TURN: [conv_turn, turn_reward],

                        # user-oriented reward
                        USER_REWARD: user_reward,

                        # rewards on objectives of interest
                        # toxicity for emotional support conversation
                        TOXICITY: toxicity
                    }
                )

        # # final result dict
        # final_result_turns = defaultdict(list)

        # for k, v in turn_level_results.items():
        #     final_result_turns[k] = defaultdict(list)
        #     final_result_turns[k] = defaultdict(list)
        #     final_result_turns[k] = defaultdict(list)

        # for k,v in turn_level_results.items():
        #     for l in v:
        #         final_result_turns[k]['gain'].append(l[0])
        #         final_result_turns[k]['fair'].append(l[1])
        #         final_result_turns[k]['deal'].append(l[2])

        # for k, v in final_result_turns.items():
        #     final_result_turns[k]['gain'] = np.mean(final_result_turns[k]['gain'])
        #     final_result_turns[k]['fair'] = np.mean(final_result_turns[k]['fair'])
        #     final_result_turns[k]['deal'] = np.mean(final_result_turns[k]['deal'])

        # for k, v in final_result_turns.items():
        #     print(f"turn {k}, values: {v}")

        # strategy statistics
        loguru_logger.debug(f"strategy_statistics={strategy_statistics}")

        # compute the results using the evaluator
        results = self.online_evaluator.report()

        # log the results to terminal or file
        for logger in self.loggers:
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Testing")

            # save conversations for human evaluation
            if isinstance(logger, FileLogger):
                for idx, conv in enumerate(convs):
                    save_conv_path = os.path.join(
                        logger.log_dir, f"conversation_{idx}.txt")
                    save_conversation_for_human_evaluation(
                        save_conv_path, conv)

        # return the results of the online evaluation
        return results
