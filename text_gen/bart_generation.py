import math
import os

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger as loguru_logger

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BartForConditionalGeneration

from base.text_gen import PLMGeneration
from base.pipeline import Pipeline
from base.trainer import Trainer
from base.torch_dataset import BaseTorchDataset
from base.data_processor import DataProcessorForGeneration

from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT
from utils.generation import convert_example_to_feature_for_generation_recommendation, IGNORE_INDEX, \
    convert_example_to_feature_for_generation_negotiation, convert_example_to_feature_for_generation_emotional_support

from base.model import Model
from config.config import GenerationConfig, ModelConfig

from logger.file_logger import FileLogger
from logger.wandb_logger import WanDBLogger


class BARTGenerationConfig(GenerationConfig, ModelConfig):
    """
    BART Generation config class
    """
    no_repeat_ngram_size = 3
    max_target_length = 50
    pass


class BARTDataProcessorForGeneration(DataProcessorForGeneration):

    def __call__(self, *args, **kwargs):
        """
        function that convert instances to data features for the generation task
        :param args:  arguments
        :param kwargs:  keywords arguments
        :return:
        """
        # recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            input, label = convert_example_to_feature_for_generation_recommendation(*args,
                                                                                    **kwargs)
        # negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            input, label = convert_example_to_feature_for_generation_negotiation(
                *args, **kwargs)
        # emotional support scenario
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            input, label = convert_example_to_feature_for_generation_emotional_support(
                *args, **kwargs)
        # invalid scenario
        else:
            raise Exception("Invalid Scenario !!")
        return input, label


class BARTTorchDatasetForGeneration(BaseTorchDataset):
    """
    BART Torch Dataset for generation class
    """
    pass


class BARTTrainerForGeneration(Trainer):

    def __init__(self, dataset_config, game_config, model_config, accelerator, game, model, offline_evaluator,
                 online_evaluator,
                 loggers):
        """
        constructor for class BartTrainerForGeneration
        :param dataset_config: the configuration of the dataset
        :param game_config: the game configuration
        :param model_config: the model configuration
        :param accelerator: the accelerator
        :param game: the game
        :param model: the model
        :param offline_evaluator: an instance of class offline evaluator
        :param online_evaluator: an instance of class online evaluator
        :param loggers: a set of loggers.
        """
        super().__init__(game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                         loggers)
        self.dataset_config = dataset_config

    def process_dataset(self, dataset):
        """
        method that process the given dataset and return processed data instances
        :param dataset: an instance of the Dataset class
        :return: processed data instances.
        """
        return dataset.train_instances, dataset.dev_instances, dataset.test_instances

    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1, is_test=False):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :param is_test: True if we're in inference time, else False
        :return: a instance of torch dataloader class
        """

        # construct the torch dataset
        torch_dataset = BARTTorchDatasetForGeneration(

            tokenizer=self.model.tokenizer,
            instances=data_instances,
            max_sequence_length=self.model_config.max_sequence_length,
            device=self.device,
            convert_example_to_feature=BARTDataProcessorForGeneration(
                self.game_config, self.dataset_config),

            # for generation task
            is_gen=True,
            is_test=is_test,
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

    def create_optimizer(self):
        """
        method that create the optimizer to train the model
        :return: a torch.optim.Optimizer
        """
        modules = [self.model]
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
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.model_config.learning_rate)
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
            logits, loss = self.model(batch)
            self.accelerator.backward(loss)
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

    def eval_epoch(self, data_loader, criterion, return_responses=False):
        """
        method that evaluates the model on the validation set.
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :param: return_responses: True if we with to save the generated responses
        :return: evaluation loss
        """
        self.model.eval()
        # development loss
        dev_loss = []

        # ground truth responses
        # loop over the validation step
        for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):

            # compute validation loss
            with torch.no_grad():
                _, loss = self.model(batch)
                dev_loss.append(loss.item())

            # generate the output sequence
            gen_seqs = self.accelerator.unwrap_model(self.model.plm).generate(
                **batch['context'],
                max_new_tokens=self.model_config.max_gen_length,
                no_repeat_ngram_size=self.model_config.no_repeat_ngram_size
            )
            # pre-processing the output sequences
            gen_resp_ids = []
            for gen_seq in gen_seqs:
                gen_seq = [token_id for token_id in gen_seq if token_id !=
                           self.model.tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq)

            # pre-processing the labels
            label_resp_ids = []
            for label_seq in batch['labels']:
                label_seq = [
                    token_id for token_id in label_seq if token_id != IGNORE_INDEX]
                label_resp_ids.append(label_seq)

            # decoding the prediction and labels
            # convert sequences of ids to sequences of text
            decoded_preds = self.model.tokenizer.batch_decode(
                gen_resp_ids, skip_special_tokens=True)
            # for other kind of generation
            # this block of code should be modified
            decoded_preds = [decoded_pred.replace('</s>', '').replace('<s>', '') for decoded_pred in
                             decoded_preds]
            decoded_preds = [pred.strip() for pred in decoded_preds]

            decoded_labels = self.model.tokenizer.batch_decode(
                label_resp_ids, skip_special_tokens=True)
            decoded_labels = [decoded_label.replace('</s>', '').replace('<s>', '') for decoded_label in
                              decoded_labels]
            decoded_labels = [label.strip() for label in decoded_labels]

            # evaluate the performance.
            self.offline_evaluator.record(decoded_preds, decoded_labels)

        dev_loss = np.mean(dev_loss) * \
            self.model_config.gradient_accumulation_steps
        # compute the values of metrics
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss

        # saving the generated responses and labels
        if return_responses:
            response_records = []
            for pred_res, label_res in list(zip(self.offline_evaluator.preds, self.offline_evaluator.labels)):
                response_records.append({
                    "pred": pred_res,
                    "label": label_res
                })
            return results, response_records
        # only return the values of metrics
        return results

    def train_sft(self, dataset, device=None):
        """
        method that train the model
        :param dataset: the dataset that we wish to train the model
        :param device: the device we use to train the model
        :return: the trained model
        """

        best_loss = math.inf

        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(
            dataset)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances,
                                                  batch_size=self.model_config.per_device_train_batch_size,
                                                  shuffle=True, num_workers=self.model_config.num_workers,
                                                  is_test=False)

        dev_loader = self.construct_dataloaders(dev_instances, batch_size=self.model_config.per_device_eval_batch_size,
                                                shuffle=False, num_workers=self.model_config.num_workers, is_test=False)

        # create the optimizer
        self.optimizer = self.create_optimizer()
        model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader)

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
        model.to(device)
        for epoch in range(self.model_config.num_train_epochs):
            model.train()

            # reset the offline evaluator before each training epoch
            self.offline_evaluator.reset()

            # train the model
            train_loss, stop = self.train_epoch(data_loader=train_loader, optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                criterion=self.criterion,
                                                max_train_steps=max_train_steps
                                                )
            # evaluate the performance
            results, response_records = self.eval_epoch(
                dev_loader, self.criterion, return_responses=True)

            # logging the results
            for logger in self.loggers:
                logger.record(results, epoch + 1)

                # saving the generated response
                if isinstance(logger, FileLogger):
                    logger.save_responses(response_records, self.model_config.log_dir,
                                          file_name=f"{logger.model_name}-{logger.local_time}-{epoch + 1}-Dev.txt")

            # saving the model if needed
            # for generation, we use the loss as the saving criterion
            if results['loss'] < best_loss:
                loguru_logger.info(
                    "Performance improved. Saving the model .....")
                best_loss = results['loss']
                file_path = os.path.join(
                    self.model_config.saved_dir, "model.pth")
                self.save_model(file_path)

            # terminal if needed
            if stop:
                print("Training process is completed.")
                break

    def predict(self, instance):
        """
        method that predict the output response given an particular input instance
        :param instance: the input instance
        :return: a generated response in form of text
        """

        # construct the data loader
        # set is test to True for inference.
        data_loader = self.construct_dataloaders([instance], batch_size=1, is_test=True,
                                                 num_workers=self.model_config.num_workers, shuffle=False)
        self.model.to(self.device)
        # data_loader, self.model = self.accelerator.prepare(data_loader, self.model)

        self.model.eval()
        # loop over the validation step
        for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):

            # generate the output sequence
            gen_seqs = self.model.plm.generate(
                **batch['context'],
                max_new_tokens=self.model_config.max_gen_length,
                no_repeat_ngram_size=self.model_config.no_repeat_ngram_size
            )

            # decoding the prediction and labels
            # pre-processing the output sequences
            gen_resp_ids = []
            for gen_seq in gen_seqs:
                gen_seq = [token_id for token_id in gen_seq if token_id !=
                           self.model.tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq)

            # convert sequences of ids to sequences of text
            decoded_preds = self.model.tokenizer.batch_decode(
                gen_resp_ids, skip_special_tokens=True)

            # for other kind of generation
            # this block of code should be modified
            decoded_preds = [decoded_pred.replace('</s>', '').replace('<s>', '') for decoded_pred in
                             decoded_preds]
            decoded_preds = [pred.strip() for pred in decoded_preds]

            return decoded_preds[0]

    def test(self, dataset):
        """
        method that evaluate the model performance on the test set,
        :param dataset: the given dataset that we want to evaluate the performance
        :return: the results of metrics and generated responses
        """
        # construct the data loader
        test_loader = self.construct_dataloaders(dataset.test_instances,
                                                 batch_size=self.model_config.per_device_eval_batch_size,
                                                 shuffle=False, num_workers=self.model_config.num_workers,
                                                 is_test=False)

        # evaluate the perfomance.
        results, response_records = self.eval_epoch(
            test_loader, self.criterion, return_responses=True)
        return results, response_records


class BARTModelForGeneration(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for Class BERT-based policy model
        :param model_config: the model configuration class
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        # create the tokenizer and the backbone pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)

        # for generation, we need to use class bart for conditional generation
        self.plm = BartForConditionalGeneration.from_pretrained(self.model_config.plm,
                                                                cache_dir=self.model_config.cached_dir)

        # prepend special tokens to the vocabulary and resize the embedding matrix of the PLM
        self.tokenizer.add_special_tokens(
            self.model_config.special_tokens_dict)
        self.plm.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        """
        Forward function
        :param batch: a batched tensor data
        :return: logits and loss
        """
        outputs = self.plm(**batch['context'],
                           labels=batch['labels'], return_dict=True)
        logits = outputs["logits"]
        loss = outputs["loss"]
        return logits, loss


class BARTPipelineForGeneration(Pipeline):

    def inference(self, instance):
        """
        method that predict the output response given an particular input instance
        :param instance: the given input instance
        :return: a generated response in the format of text
        """
        generated_response = self.trainer.predict(instance)
        return generated_response

    def run_sft(self):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :return: the results of the current run.
        """

        # train and select the best model
        self.trainer.train_sft(self.dataset, self.device)

        # evaluate the performance on the test set
        # reset the offline evaluator
        self.trainer.offline_evaluator.reset()

        # load the pretrained model
        self.load_pretrained_model()

        # compute the performance on test set and generated responses
        results, response_records = self.trainer.test(self.dataset)

        # logging the test results
        for logger in self.trainer.loggers:

            # log the results to file or terminal
            # currently it is not available for Wandb Logger
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Test Set")

            # saving the generated response
            if isinstance(logger, FileLogger):
                logger.save_responses(response_records, self.model_config.log_dir,
                                      file_name=f"{logger.model_name}-{logger.local_time}-Test.txt")

        # return the results of the current run
        return results

    def execute(self):
        """
        method that execute the training pipeline
        :return:
        """
        loguru_logger.info(
            "Fine tuning the generation model on the background dataset.....")
        sft_result = self.run_sft()
        return sft_result


class BARTGeneration(PLMGeneration):

    def __init__(self, generation_config, pipeline, is_test=False):
        """
        constructor for class Bart Generation
        :param generation_config: an instance of the generation config
        :param pipeline: a particular pipeline
        :param is_test: True if we are using the generation class at inference time
        """
        super().__init__()
        self.genertation_config = generation_config
        self.pipeline = pipeline
        self.is_test = is_test

        # creating the log dir for the scenario if it is not existed
        os.makedirs(self.pipeline.game_config.log_dir, exist_ok=True)

        # utilizing this class at inference time
        if is_test:
            loguru_logger.info(
                "Testing phase. Loading model checkpoint ......")
            # load the pretrained generation model
            self.pipeline.load_pretrained_model()

    def generate_response(self, instance):
        """
        function that generate the response
        :return:
        """
        generated_response = self.pipeline.inference(instance)
        return generated_response

    def prepare(self):
        """
        method that prepare the text generation model
        :return: the results of fine-tuning the generation model
        """
        results = self.pipeline.execute()
        return results
