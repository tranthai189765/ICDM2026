import random
import json

import torch.random
import yaml
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from fastchat.model import add_model_args

# datasets for recommendation
from dataset.rec_datasets.durecdial import DuRecdial
from dataset.rec_datasets.inspired import Inspired
# datasets for negotiation
from dataset.neg_datasets.bargain import CraiglistBargain

# datasets for emotional support
from dataset.es_datasets.esc import ESConv

from config.constants import *

from config.config import RecommendationGameConfig, DatasetConfigForRecommendation, DatasetConfigForNegotiation, \
    NegotiationGameConfig, EmotionalSupportGameConfig, DatasetConfigForEmotionalSupport

from base.game import RecommendationGame, NegotiationGame, EmotionalSupportGame

# # policy models
# # BERT
# from baselines.BERT.pipeline import BERTPipelineForRecommendation
# from baselines.BERT.model import BERTModel
# from baselines.BERT.trainer import BERTTrainer
# from baselines.BERT.config import BERTConfig
# # BART
# from baselines.BART.pipeline import BARTPipelineForRecommendation
# from baselines.BART.model import BARTModel
# from baselines.BART.trainer import BARTTrainer
# from baselines.BART.config import BARTConfig
# # PPDPP
# from baselines.PPDPP.pipeline import PPDPPPipelineForRecommendation, PPDPPPipelineForNegotiation, \
#     PPDPPPipelineForEmotionalSupport
# from baselines.PPDPP.model import PPDPPModel
# from baselines.PPDPP.trainer import PPDPPTrainer
# from baselines.PPDPP.config import PPDPPConfigForRecommendation, PPDPPConfigForNegotiation, \
#     PPDPPConfigForEmotionalSupport

# # TRIP
# from baselines.TRIP.pipeline import TRIPPipelineForRecommendation, TRIPPipelineForNegotiation, \
#     TRIPPipelineForEmotionalSupport
# from baselines.TRIP.model import TRIPModel
# from baselines.TRIP.trainer import TRIPTrainer
# from baselines.TRIP.config import TRIPConfigForRecommendation, TRIPConfigForNegotiation, \
#     TRIPConfigForEmotionalSupport

# # DPDP
# from baselines.DPDP.pipeline import DPDPPipelineForRecommendation, DPDPPipelineForNegotiation, \
#     DPDPPipelineForEmotionalSupport
# from baselines.DPDP.model import DPDPModel
# from baselines.DPDP.trainer import DPDPTrainer
# from baselines.DPDP.config import DPDPConfigForRecommendation, DPDPConfigForNegotiation, \
#     DPDPConfigForEmotionalSupport


# # UNIMIND
# from baselines.UNIMIND.pipeline import UNIMINDPipelineForRecommendation
# from baselines.UNIMIND.model import UNIMINDModel
# from baselines.UNIMIND.trainer import UNIMINDTrainer
# from baselines.UNIMIND.config import UNIMINDConfigForRecommendation

# # RTCP
# from baselines.RTCP.pipeline import RTCPPipelineForRecommendation, RTCPPipelineForNegotiation, \
#     RTCPPipelineForEmotionalSupport
# from baselines.RTCP.model import RTCPModel
# from baselines.RTCP.trainer import RTCPTrainer
# from baselines.RTCP.config import RTCPConfigForRecommendation, RTCPConfigForNegotiation, \
#     RTCPConfigForEmotionalSupport

# # COLOR
# from baselines.COLOR.pipeline import COLORPipelineForRecommendation
# from baselines.COLOR.model import COLORModel
# from baselines.COLOR.trainer import COLORTrainer
# from baselines.COLOR.config import COLORConfigForRecommendation

# # MODPL
# from modpl.config import MODPLConfigForRecommendation, MODPLConfigForNegotiation, MODPLConfigForEmotionalSupport
# from modpl.trainer import MODPLTrainer
# from modpl.model import MODPLModel
# from modpl.pipeline import MODPLPipelineForRecommendation, MODPLPiplineForNegotiation, MODPLPiplineForEmotionalSupport

# # Contextual NODPL
# from modpl_new.config import ContextualMODPLConfigForRecommendation, ContextualMODPLConfigForNegotiation, \
#     ContextualMODPLConfigForEmotionalSupport
# from modpl_new.trainer import ContextualMODPLTrainer
# from modpl_new.model import ContextualMODPLModel
# from modpl_new.pipeline import ContextualMODPLPipelineForRecommendation, ContextualMODPLPipelineForNegotiation, \
#     ContextualMODPLPipelineForEmotionalSupport

# Contextual NODPL
from padpp.config import PADPPConfigForRecommendation, PADPPConfigForNegotiation, \
    PADPPConfigForEmotionalSupport
from padpp.trainer import PADPPTrainer
from padpp.model import PADPPModel
from padpp.pipeline import PADPPPipelineForRecommendation, PADPPPipelineForNegotiation, \
    PADPPPipelineForEmotionalSupport

# DMORL
from dmorl.config import DMORLConfigForRecommendation, DMORLConfigForNegotiation, \
    DMORLConfigForEmotionalSupport
from dmorl.trainer import DMORLTrainer
from dmorl.model import DMORLModel
from dmorl.pipeline import DMORLPipelineForRecommendation, DMORLPipelineForNegotiation, \
    DMORLPipelineForEmotionalSupport

# # SetMax PADPP
# from baselines.PADPP_smp.config import SetMaxPADPPConfigForRecommendation, SetMaxPADPPConfigForNegotiation, \
#     SetMaxPADPPConfigForEmotionalSupport
# from baselines.PADPP_smp.trainer import SetMaxPADPPTrainer
# from baselines.PADPP_smp.model import SetMaxPADPPModel
# from baselines.PADPP_smp.pipeline import SetMaxPADPPPipelineForNegotiation, SetMaxPADPPPipelineForRecommendation, \
#     SetMaxPADPPPipelineForEmotionalSupport


# # Min Dist PADPP
# from baselines.PADPP_min_dist.config import MinDistPADPPConfigForRecommendation, MinDistPADPPConfigForNegotiation, \
#     MinDistPADPPConfigForEmotionalSupport
# from baselines.PADPP_min_dist.trainer import MinDistPADPPTrainer
# from baselines.PADPP_min_dist.model import MinDistPADPPModel
# from baselines.PADPP_min_dist.pipeline import MinDistPADPPPipelineForNegotiation, MinDistPADPPPipelineForRecommendation, \
#     MinDistPADPPPipelineForEmotionalSupport


# # Deep Double Q Learning
# from baselines.DDQL.config import DDQLConfigForRecommendation, DDQLConfigForNegotiation, \
#     DDQLConfigForEmotionalSupport
# from baselines.DDQL.trainer import DDQLTrainer
# from baselines.DDQL.model import DDQLModel
# from baselines.DDQL.pipeline import DDQLPipelineForRecommendation, DDQLPipelineForNegotiation, \
#     DDQLPipelineForEmotionalSupport


# # Envelope MOQ
# from baselines.Envelope.config import EnvelopeConfigForRecommendation, EnvelopeConfigForNegotiation, \
#     EnvelopeConfigForEmotionalSupport
# from baselines.Envelope.trainer import EnvelopeTrainer
# from baselines.Envelope.model import EnvelopeModel
# from baselines.Envelope.pipeline import EnvelopePipelineForRecommendation, EnvelopePipelineForNegotiation, \
#     EnvelopePipelineForEmotionalSupport


# # Pro_active Chain-of-thought (ProCOT)
# from baselines.ProCOT.config import ProCOTConfigForNegotiation, ProCOTConfigForEmotionalSupport
# from baselines.ProCOT.trainer import ProCOTTrainer
# from baselines.ProCOT.model import ProCOTModel
# from baselines.ProCOT.pipeline import ProCOTPipelineForNegotiation, ProCOTPipelineForEmotionalSupport

# # Standard Prompting (Standard)
# from baselines.Standard.config import StandardPromptConfigForNegotiation, StandardPromptConfigForEmotionalSupport
# from baselines.Standard.trainer import StandardPromptTrainer
# from baselines.Standard.model import StandardPromptModel
# from baselines.Standard.pipeline import StandardPromptPipelineForNegotiation, StandardPromptPipelineForEmotionalSupport

# # ICL_AIF model
# from baselines.ICL_AIF.config import ICLAIFConfigForNegotiation, ICLAIFConfigForEmotionalSupport
# from baselines.ICL_AIF.trainer import ICLAIFTrainer
# from baselines.ICL_AIF.model import ICLAIFModel
# from baselines.ICL_AIF.pipeline import ICLAIFPipelineForNegotiation, ICLAIFPipelineForEmotionalSupport

# # Proactive
# from baselines.Proactive.config import ProactiveConfigForNegotiation, ProactiveConfigForEmotionalSupport
# from baselines.Proactive.trainer import ProactiveTrainer
# from baselines.Proactive.model import ProactiveModel
# from baselines.Proactive.pipeline import ProactivePipelineForNegotiation, ProactivePipelineForEmotionalSupport

# # Ask-an-Expert
# from baselines.AnE.config import AnEConfigForNegotiation, AnEConfigForEmotionalSupport
# from baselines.AnE.trainer import AnETrainer
# from baselines.AnE.model import AnEModel
# from baselines.AnE.pipeline import AnEPipelineForNegotiation, AnEPipelineForEmotionalSupport

# # GDP-Zero
# from baselines.GDP_Zero.config import GDPZeroConfigForNegotiation, GDPZeroConfigForEmotionalSupport
# from baselines.GDP_Zero.trainer import GDPZeroTrainer
# from baselines.GDP_Zero.model import GDPZeroModel
# from baselines.GDP_Zero.pipeline import GDPZeroPipelineForNegotiation, GDPZeroPipelineForEmotionalSupport

# metrics for evaluation
from eval.metric import Accuracy, PrecisionRecallF1, Item_Freq, SR, OfflineMetric, OnlineMetric, DistN, AverageTurn, \
    RougeN, BleuN, Fairness, SL_Ratio, Toxicity, User_Reward, DealRate

# user simulators
from simulator.rec_simulator import RecommendationSimulator
from simulator.neg_simulator import NegotiationSimulator
from simulator.es_simulator import EmotionalSupportSimulator

# utility classes for BART generation
from text_gen.bart_generation import BARTGeneration, BARTGenerationConfig, BARTTrainerForGeneration, \
    BARTModelForGeneration, BARTPipelineForGeneration

from text_gen.chatgpt_generation import ChatGPTGeneration, ChatGPTConfigForGeneration
from text_gen.vicuna_generation import VicunaGeneration, VicunaGenerationConfig
from text_gen.llama3_generation import Llama3Generation, Llama3ConfigForGeneration
from text_gen.fpt_generation import FPTGeneration, FPTConfigForGeneration

# loggers
from logger.file_logger import FileLogger
from logger.terminal_logger import TerminalLogger
from logger.wandb_logger import WanDBLogger


def set_seed(seed):
    """
    control the random seed for each run
    :param seed: the random seed
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    """
    function that parse arguments from the command line
    :return: a set of keywords arugments
    """
    parser = argparse.ArgumentParser()

    # hyper_parameters for runnning the pipelines
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--project_name", type=str,
                        default="MODPL", help="The project's name")
    parser.add_argument("--exp_name", type=str, default="",
                        help="The experiment's name")
    parser.add_argument("--log_dir", type=str,
                        default="logs", help="the log dir")
    parser.add_argument("--loggers", type=str,
                        default="terminal", help="names of loggers")
    parser.add_argument("--scenario", type=str,
                        default="recommendation", help="the scenario of interest")
    parser.add_argument("--datasets", type=str, help="names of datasets.")
    parser.add_argument("--models", type=str,
                        default='bert', help="names of models")
    parser.add_argument("--gen_models", type=str,
                        default='bart', help="names of models")
    parser.add_argument("--metrics", type=str, help="names of metrics")
    parser.add_argument('--overwrite_sim', action='store_true',
                        help='if we overwrite the saved user simulators')

    # arguments fro recommendation training
    parser.add_argument("--domain", default='movie', type=str,
                        help="the name of the domain of consideration")

    # arguments for analyses
    parser.add_argument('--ablation', type=str, default='',
                        help='Running ablation study')
    parser.add_argument('--objective_weight', type=str,
                        default=None, help='The objective weight')
    parser.add_argument('--model_type', type=str,
                        default='llama3', help='The type of the llm model')
    parser.add_argument('--use_persona', action='store_true',
                        help='if using persona for simulator')
    parser.add_argument('--prioritized_objective', type=str,
                        default='uniform', help='The prioritized objective')
    parser.add_argument('--test_phase', action='store_true',
                        help='if we are in the testing phase')
    parser.add_argument('--num_train_rl_epochs', type=int,
                        default=50, help='the number of RL training epochs')
    parser.add_argument('--n_preferences', type=int, default=128,
                        help='the number of updated preferences used for GPI')

    # default is using generalized policy improvement
    parser.add_argument('--use_gpi', type=int, default=1,
                        help='1 if we use GPI else 0')
    parser.add_argument('--num_test_cases', type=int,
                        default=0, help='The number of test cases')
    # add vicuna training params
    add_model_args(parser)

    args, _ = parser.parse_known_args()
    return args


def load_binary_file(file_path):
    """
    function thats load a binary file
    @param file_path: the path to the saved file
    @return: a pickle object
    """
    with open(file_path, 'rb') as f:
        object = pickle.load(f)
        return object


def save_binary_file(object, file_path):
    """
    function that save an object to a binary file
    @param object: an python object
    @param file_path: the path to the saved file
    @return: None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)


def convert_list_to_str(knowledge):
    """
    function that convert a list of 3 elements to a text string
    @param knowledge: a list of 3 element where each element is a string
    @return: a text string
    """
    if len(knowledge) == 0:
        return ""
    return f"{knowledge[0]} {knowledge[1]} {knowledge[2]}"


def convert_dict_to_str(profile):
    """
    function that convert a dictionary to a text string
    @param profile: a dictionary which contains information about the user.
    @return: a text string
    """
    out_str = ""
    for k, v in profile.items():
        if isinstance(v, list):
            value_str = ""
            for e in v:
                value_str += e
                value_str += " "
            out_str += f"{k}: {value_str}"
        elif isinstance(v, str):
            out_str += f"{k}: {v}"
        out_str += " "
    return out_str


def load_config_from_yaml_file(file_path):
    """
    function that load a dictionary of parameters from a yaml file
    :param file_path: the string indicates the file path
    :return: a dictionary of parameters
    """
    with open(file_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        return params


def convert_string_to_list(string):
    """
    function that converts a string to a list
    :param string: a text string which contains several elements
    :return: a list of elements
    """
    return string.strip().split(',')


def reformat_args(args):
    """
    function that reformats the arguments
    :param args: a dictionary that contains arguments and their values
    :return: a reformated args
    """
    for k, v in args.items():
        if isinstance(v, str):
            if ',' in v:
                args[k] = convert_string_to_list(v)
    return args


def get_datasets_by_names(scenario, dataset_names):
    """
    function that returns a list of dataset class and their configuration file path by their names
    :param scenario: the scenario that we are considering
    :param dataset_names: a list containing the names of datasets
    :return: a list of dataset classes and their configuration file path.
    """
    # pre-processing for coding convenience
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]

    if scenario == RECOMMENDATION:
        dataset_dict = {
            DURECDIAL: [
                DURECDIAL_CONFIG_PATH,
                DuRecdial,
                DatasetConfigForRecommendation

            ],
            INSPIRED: [
                INSPIRED_CONFIG_PATH,
                Inspired,
                DatasetConfigForRecommendation
            ]
        }
        # collecting datasets
        datasets = []
        for name in dataset_names:
            datasets.append(dataset_dict[name])
        return datasets
    # negotiation scenario
    elif scenario == NEGOTIATION:
        dataset_dict = {
            CRAIGSLIST_BARGAIN: [
                CRAIGSLIST_BARGAIN_CONFIG_PATH,
                CraiglistBargain,
                DatasetConfigForNegotiation
            ]
        }
        # collecting datasets
        datasets = []
        for name in dataset_names:
            datasets.append(dataset_dict[name])
        return datasets
    # emotional support scenario
    elif scenario == EMOTIONAL_SUPPORT:
        dataset_dict = {
            ES_CONV: [
                ES_CONV_CONFIG_PATH,
                ESConv,
                DatasetConfigForEmotionalSupport
            ]
        }
        # collecting datasets
        datasets = []
        for name in dataset_names:
            datasets.append(dataset_dict[name])
        return datasets
    else:
        raise Exception("Invalid scenario !!!")


def get_model_by_names(scenario, model_names):
    """
    function that returns a set of models by their names
    :param scenario: the scenario we are considering
    :param model_names: a list of model names
    :return: a list of model classes
    """
    # pre-processing for coding convenience
    if not isinstance(model_names, list):
        model_names = [model_names]
    # recommendation scenario
    if scenario == RECOMMENDATION:
        model_dict = {
            # BERT: [
            #     BERT_CONFIG_PATH,
            #     BERTConfig,
            #     BERTModel,
            #     BERTPipelineForRecommendation,
            #     BERTTrainer,
            # ],
            # BART: [
            #     BART_CONFIG_PATH,
            #     BARTConfig,
            #     BARTModel,
            #     BARTPipelineForRecommendation,
            #     BARTTrainer,
            # ],
            # # PPDPP model
            # PPDPP: [
            #     PPDPP_CONFIG_PATH_FOR_RECOMMENDATION,
            #     PPDPPConfigForRecommendation,
            #     PPDPPModel,
            #     PPDPPPipelineForRecommendation,
            #     PPDPPTrainer
            # ],
            # # DPDP
            # DPDP: [
            #     DPDP_CONFIG_PATH_FOR_RECOMMENDATION,
            #     DPDPConfigForRecommendation,
            #     DPDPModel,
            #     DPDPPipelineForRecommendation,
            #     DPDPTrainer
            # ],
            # # TRIP model
            # TRIP: [
            #     TRIP_CONFIG_PATH_FOR_RECOMMENDATION,
            #     TRIPConfigForRecommendation,
            #     TRIPModel,
            #     TRIPPipelineForRecommendation,
            #     TRIPTrainer
            # ],
            # # unimind model
            # UNIMIND: [
            #     UNIMIND_CONFIG_PATH_FOR_RECOMMENDATION,
            #     UNIMINDConfigForRecommendation,
            #     UNIMINDModel,
            #     UNIMINDPipelineForRecommendation,
            #     UNIMINDTrainer
            # ],
            # # color model
            # COLOR: [
            #     COLOR_CONFIG_PATH_FOR_RECOMMENDATION,
            #     COLORConfigForRecommendation,
            #     COLORModel,
            #     COLORPipelineForRecommendation,
            #     COLORTrainer
            # ],
            # # RTCP model
            # RTCP: [
            #     RTCP_CONFIG_PATH_FOR_RECOMMENDATION,
            #     RTCPConfigForRecommendation,
            #     RTCPModel,
            #     RTCPPipelineForRecommendation,
            #     RTCPTrainer
            # ],
            # # preference-enhanced model
            # MODPL: [
            #     MODPL_CONFIG_PATH_FOR_RECOMMENDATION,
            #     MODPLConfigForRecommendation,
            #     MODPLModel,
            #     MODPLPipelineForRecommendation,
            #     MODPLTrainer
            # ],
            # contextual MODPL
            PADPP: [
                PADPP_CONFIG_PATH_FOR_RECOMMENDATION,
                PADPPConfigForRecommendation,
                PADPPModel,
                PADPPPipelineForRecommendation,
                PADPPTrainer
            ],
            # DMORL – Dynamic Multi-Objective RL
            DMORL: [
                DMORL_CONFIG_PATH_FOR_RECOMMENDATION,
                DMORLConfigForRecommendation,
                DMORLModel,
                DMORLPipelineForRecommendation,
                DMORLTrainer
            ],
            # # DDQL
            # DDQL: [
            #     DDQL_CONFIG_PATH_FOR_RECOMMENDATION,
            #     DDQLConfigForRecommendation,
            #     DDQLModel,
            #     DDQLPipelineForRecommendation,
            #     DDQLTrainer
            # ]
        }
        # collect model packages
        models = []
        for name in model_names:
            models.append(model_dict[name])
        return models
    # the negotiation scenario
    # remember to change the model class to negotiation
    elif scenario == NEGOTIATION:
        model_dict = {
            # # PPDPP model
            # PPDPP: [
            #     PPDPP_CONFIG_PATH_FOR_NEGOTIATION,
            #     PPDPPConfigForNegotiation,
            #     PPDPPModel,
            #     PPDPPPipelineForNegotiation,
            #     PPDPPTrainer
            # ],
            # # trip
            # TRIP: [
            #     TRIP_CONFIG_PATH_FOR_NEGOTIATION,
            #     TRIPConfigForNegotiation,
            #     TRIPModel,
            #     TRIPPipelineForNegotiation,
            #     TRIPTrainer
            # ],
            # # DPDP
            # DPDP: [
            #     DPDP_CONFIG_PATH_FOR_NEGOTIATION,
            #     DPDPConfigForNegotiation,
            #     DPDPModel,
            #     DPDPPipelineForNegotiation,
            #     DPDPTrainer
            # ],
            # # ddql
            # DDQL: [
            #     DDQL_CONFIG_PATH_FOR_NEGOTIATION,
            #     DDQLConfigForNegotiation,
            #     DDQLModel,
            #     DDQLPipelineForNegotiation,
            #     DDQLTrainer
            # ],
            # # envelope
            # ENVELOPE: [
            #     ENVELOPE_CONFIG_PATH_FOR_NEGOTIATION,
            #     EnvelopeConfigForNegotiation,
            #     EnvelopeModel,
            #     EnvelopePipelineForNegotiation,
            #     EnvelopeTrainer
            # ],
            # # RTCP model
            # RTCP: [
            #     RTCP_CONFIG_PATH_FOR_NEGOTIATION,
            #     RTCPConfigForNegotiation,
            #     RTCPModel,
            #     RTCPPipelineForNegotiation,
            #     RTCPTrainer
            # ],
            # # preference-enhanced model
            # # multi-objective dialogue policy learning
            # MODPL: [
            #     MODPL_CONFIG_PATH_FOR_NEGOTIATION,
            #     MODPLConfigForNegotiation,
            #     MODPLModel,
            #     MODPLPiplineForNegotiation,
            #     MODPLTrainer
            # ],
            # contextual modpl for negotiation
            PADPP: [
                PADPP_CONFIG_PATH_FOR_NEGOTIATION,
                PADPPConfigForNegotiation,
                PADPPModel,
                PADPPPipelineForNegotiation,
                PADPPTrainer
            ],
            # DMORL – Dynamic Multi-Objective RL
            DMORL: [
                DMORL_CONFIG_PATH_FOR_NEGOTIATION,
                DMORLConfigForNegotiation,
                DMORLModel,
                DMORLPipelineForNegotiation,
                DMORLTrainer
            ],
            # # Set max PADPP
            # SMP_PADPP: [
            #     SMP_PADPP_CONFIG_PATH_FOR_NEGOTIATION,
            #     SetMaxPADPPConfigForNegotiation,
            #     SetMaxPADPPModel,
            #     SetMaxPADPPPipelineForNegotiation,
            #     SetMaxPADPPTrainer
            # ],
            # # Min dist PADPP
            # MIN_DIST_PADPP: [
            #     MIN_DIST_PADPP_CONFIG_PATH_FOR_NEGOTIATION,
            #     MinDistPADPPConfigForNegotiation,
            #     MinDistPADPPModel,
            #     MinDistPADPPPipelineForNegotiation,
            #     MinDistPADPPTrainer
            # ],

            # # proactive chain of though (ProCOT) for negotiation
            # PRO_COT: [
            #     PRO_COT_CONFIG_PATH,
            #     ProCOTConfigForNegotiation,
            #     ProCOTModel,
            #     ProCOTPipelineForNegotiation,
            #     ProCOTTrainer
            # ],
            # # standard prompting for negotiation
            # STANDARD: [
            #     STANDARD_CONFIG_PATH,
            #     StandardPromptConfigForNegotiation,
            #     StandardPromptModel,
            #     StandardPromptPipelineForNegotiation,
            #     StandardPromptTrainer
            # ],
            # # ICL_AIF for negotiation
            # ICL_AIF: [
            #     ICL_AIF_CONFIG_PATH,
            #     ICLAIFConfigForNegotiation,
            #     ICLAIFModel,
            #     ICLAIFPipelineForNegotiation,
            #     ICLAIFTrainer
            # ],
            # # Proactive prompting for negotiation
            # PROACTIVE: [
            #     PROACTIVE_CONFIG_PATH,
            #     ProactiveConfigForNegotiation,
            #     ProactiveModel,
            #     ProactivePipelineForNegotiation,
            #     ProactiveTrainer
            # ],
            # # Ask-an-expert for negotiation
            # ANE: [
            #     ANE_CONFIG_PATH,
            #     AnEConfigForNegotiation,
            #     AnEModel,
            #     AnEPipelineForNegotiation,
            #     AnETrainer
            # ],
            # # GDP-Zero for negotiation
            # GDP_ZERO: [
            #     GDP_ZERO_CONFIG_PATH_FOR_NEGOTIATION,
            #     GDPZeroConfigForNegotiation,
            #     GDPZeroModel,
            #     GDPZeroPipelineForNegotiation,
            #     GDPZeroTrainer
            # ]
        }

        # collect model packages
        models = []
        for name in model_names:
            models.append(model_dict[name])
        return models
    # emotional support conversation
    elif scenario == EMOTIONAL_SUPPORT:
        model_dict = {
            # # PPDPP model
            # PPDPP: [
            #     PPDPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT,
            #     PPDPPConfigForEmotionalSupport,
            #     PPDPPModel,
            #     PPDPPPipelineForEmotionalSupport,
            #     PPDPPTrainer
            # ],
            # # RTCP model
            # RTCP: [
            #     RTCP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT,
            #     RTCPConfigForEmotionalSupport,
            #     RTCPModel,
            #     RTCPPipelineForEmotionalSupport,
            #     RTCPTrainer
            # ],
            # # multi-objective dialogue policy learning
            # MODPL: [
            #     MODPL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT,
            #     MODPLConfigForEmotionalSupport,
            #     MODPLModel,
            #     MODPLPiplineForEmotionalSupport,
            #     MODPLTrainer
            # ],
            # # contextual modpl for emotional support conversation

            PADPP: [
                PADPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT,
                PADPPConfigForEmotionalSupport,
                PADPPModel,
                PADPPPipelineForEmotionalSupport,
                PADPPTrainer
            ],
            # DMORL – Dynamic Multi-Objective RL
            DMORL: [
                DMORL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT,
                DMORLConfigForEmotionalSupport,
                DMORLModel,
                DMORLPipelineForEmotionalSupport,
                DMORLTrainer
            ],
            # # Proactive chain of thought for emotional support conversation
            # PRO_COT: [
            #     PRO_COT_CONFIG_PATH,
            #     ProCOTConfigForEmotionalSupport,
            #     ProCOTModel,
            #     ProCOTPipelineForEmotionalSupport,
            #     ProCOTTrainer
            # ],

            # # Standard prompting for emotional support
            # STANDARD: [
            #     STANDARD_CONFIG_PATH,
            #     StandardPromptConfigForEmotionalSupport,
            #     StandardPromptModel,
            #     StandardPromptPipelineForEmotionalSupport,
            #     StandardPromptTrainer
            # ],
            # # ICL_AIF for emotional support conversation
            # ICL_AIF: [
            #     ICL_AIF_CONFIG_PATH,
            #     ICLAIFConfigForEmotionalSupport,
            #     ICLAIFModel,
            #     ICLAIFPipelineForEmotionalSupport,
            #     ICLAIFTrainer
            # ],
            # # Proactive prompting for emotional support
            # PROACTIVE: [
            #     PROACTIVE_CONFIG_PATH,
            #     ProactiveConfigForEmotionalSupport,
            #     ProactiveModel,
            #     ProactivePipelineForEmotionalSupport,
            #     ProactiveTrainer
            # ],
            # # Ask an Expert for emotional support
            # ANE: [
            #     ANE_CONFIG_PATH,
            #     AnEConfigForEmotionalSupport,
            #     AnEModel,
            #     AnEPipelineForEmotionalSupport,
            #     AnETrainer
            # ],
            # # GDP_Zero for emotional support
            # GDP_ZERO: [
            #     GDP_ZERO_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT,
            #     GDPZeroConfigForEmotionalSupport,
            #     GDPZeroModel,
            #     GDPZeroPipelineForEmotionalSupport,
            #     GDPZeroTrainer

            # ]
        }

        # collect model packages
        models = []
        for name in model_names:
            models.append(model_dict[name])
        return models

    else:
        raise Exception("Invalid scenario !!!")


def get_metrics_by_names(scenario, metric_names):
    """
    function that returns offline and online metrics
    :param scenario: the scenario that we're runing the models
    :param metric_names: a list contains metric's names
    :return: two lists which are offline and online metrics
    """
    # recommendation scenario
    if scenario == RECOMMENDATION:
        metric_dict = {
            # offline metrics
            ACCURACY: Accuracy(),
            PRF1: PrecisionRecallF1(),
            DIST_N: DistN(Ns=[2, 3, 4]),
            BLEU_N: BleuN(Ns=[2, 3, 4]),
            ROUGE_N: RougeN(Ns=['1', '2', 'l']),

            # online metricvs
            ITEM_FREQ: Item_Freq(),
            SUCCESS_RATE: SR(),
            # user oriented metric
            # used to log the user reward
            USER_REWARD: User_Reward(),
            AVG_TURN: AverageTurn()
        }

        # pre-processing for coding convenience
        if not isinstance(metric_names, list):
            metric_names = [metric_names]

        # offline and online metrics
        offline_metrics = []
        online_metrics = []

        # loop overall metric names
        for name in metric_names:
            metric = metric_dict[name]
            if isinstance(metric, OfflineMetric):
                offline_metrics.append(metric)
            elif isinstance(metric, OnlineMetric):
                online_metrics.append(metric)
            else:
                raise Exception("invalid metric !!!")
        return offline_metrics, online_metrics

    # metrics for negotiation
    elif scenario == NEGOTIATION:
        metric_dict = {
            # offline metrics
            ACCURACY: Accuracy(),
            PRF1: PrecisionRecallF1(),
            DIST_N: DistN(Ns=[2, 3, 4]),
            BLEU_N: BleuN(Ns=[2, 3, 4]),
            ROUGE_N: RougeN(Ns=['1', '2', 'l']),

            # online metrics
            SUCCESS_RATE: SR(),
            SL_RATIO: SL_Ratio(),
            FAIRNESS: Fairness(),
            DEAL_RATE: DealRate(),
            AVG_TURN: AverageTurn()
        }

        # pre-processing for coding convenience
        if not isinstance(metric_names, list):
            metric_names = [metric_names]

        # offline and online metrics
        offline_metrics = []
        online_metrics = []

        # loop overall metric names
        for name in metric_names:
            metric = metric_dict[name]
            if isinstance(metric, OfflineMetric):
                offline_metrics.append(metric)
            elif isinstance(metric, OnlineMetric):
                online_metrics.append(metric)
            else:
                raise Exception("invalid metric !!!")
        return offline_metrics, online_metrics

    # emotional support conversation
    elif scenario == EMOTIONAL_SUPPORT:
        metric_dict = {
            # offline metrics
            ACCURACY: Accuracy(),
            PRF1: PrecisionRecallF1(),
            DIST_N: DistN(Ns=[2, 3, 4]),
            BLEU_N: BleuN(Ns=[2, 3, 4]),
            ROUGE_N: RougeN(Ns=['1', '2', 'l']),

            # online metrics
            TOXICITY: Toxicity(),
            SUCCESS_RATE: SR(),

            # the user oriented reward
            USER_REWARD: User_Reward(),
            AVG_TURN: AverageTurn()
        }

        # pre-processing for coding convenience
        if not isinstance(metric_names, list):
            metric_names = [metric_names]
        # offline and online metrics
        offline_metrics = []
        online_metrics = []

        # loop overall metric names
        for name in metric_names:
            metric = metric_dict[name]
            if isinstance(metric, OfflineMetric):
                offline_metrics.append(metric)
            elif isinstance(metric, OnlineMetric):
                online_metrics.append(metric)
            else:
                raise Exception("invalid metric !!!")
        return offline_metrics, online_metrics
    else:
        raise Exception('invalid scenario !!!')


def get_scenario_by_name(name):
    """
    function that returns the scenario config file path and config class
    :param name: the name of the scenario
    :return: a list of config file path and config class
    """
    # target-driven recommendation
    if name == RECOMMENDATION:
        return [RECOMMENDATION_CONFIG_PATH, RecommendationGameConfig, RecommendationGame, RecommendationSimulator]
    # negotiation dialogue
    elif name == NEGOTIATION:
        return [NEGOTIATION_CONFIG_PATH, NegotiationGameConfig, NegotiationGame, NegotiationSimulator]
    # should be replaced with emotional support conversation
    elif name == EMOTIONAL_SUPPORT:
        return [EMOTIONAL_SUPPORT_CONFIG_PATH, EmotionalSupportGameConfig, EmotionalSupportGame,
                EmotionalSupportSimulator]
    else:
        raise Exception("Invalid Scenario")


def get_loggers_by_names(names, **kwargs):
    """
    function that returns a list of loggers by their names
    :param names: list of logger names
    :param kwargs: keywords arguments
    :return: list of loggers
    """
    # pre-processing for coding convenience
    if not isinstance(names, list):
        names = [names]

    # a list of loggers
    loggers = []

    # colleting loggers by their names
    for name in names:
        if name == TERMINAL_LOGGER:
            logger = TerminalLogger(**kwargs)
        elif name == FILE_LOGGER:
            logger = FileLogger(**kwargs)
        elif name == WANDB_LOGGER:
            logger = WanDBLogger(**kwargs)
            pass
        else:
            raise Exception("Invalid loggers ...")
        loggers.append(logger)
    return loggers


def get_text_generation_model_by_name(scenario, names):
    """
    function that return a text generation model by name
    :param scenario: the name of the scenario
    :param names: the names of the generation model
    :return:
    """
    # pre-processing for coding convenience
    if not isinstance(names, list):
        names = [names]
    # recommendation scenario
    if scenario == RECOMMENDATION:
        generation_dict = {
            BART_GENERATION: [
                BART_GENERATION_CONFIG_PATH,
                BARTGenerationConfig,
                BARTModelForGeneration,
                BARTTrainerForGeneration,
                BARTPipelineForGeneration,
                BARTGeneration,
            ],
            CHATGPT: [
                CHATGPT_GENERATION_CONFIG_PATH,
                CHATGPT_PROMPT_FOR_RECOMMENDATION,
                ChatGPTConfigForGeneration,
                ChatGPTGeneration
            ],
            VICUNA: [
                VICUNA_GENERATION_CONFIG_PATH,
                VICUNA_PROMPT_FOR_RECOMMENDATION,
                VicunaGenerationConfig,
                VicunaGeneration
            ],
            LLAMA3: [
                LLAMA3_GENERATION_CONFIG_PATH,
                LLAMA3_PROMPT_FOR_RECOMMENDATION,
                Llama3ConfigForGeneration,
                Llama3Generation
            ],
            FPT: [
                FPT_GENERATION_CONFIG_PATH,
                FPT_PROMPT_FOR_RECOMMENDATION,
                FPTConfigForGeneration,
                FPTGeneration
            ]
        }
        # a list contains generation packages
        generation_packages = []

        # loop over text generation names
        for name in names:
            package = generation_dict[name]
            generation_packages.append(package)

        # return the generaton packages
        return generation_packages

    # negotiation dialogue
    elif scenario == NEGOTIATION:
        generation_dict = {
            BART_GENERATION: [
                BART_GENERATION_CONFIG_PATH,
                BARTGenerationConfig,
                BARTModelForGeneration,
                BARTTrainerForGeneration,
                BARTPipelineForGeneration,
                BARTGeneration,

            ],
            CHATGPT: [
                CHATGPT_GENERATION_CONFIG_PATH,
                CHATGPT_PROMPT_FOR_NEGOTIATION,
                ChatGPTConfigForGeneration,
                ChatGPTGeneration
            ],
            VICUNA: [
                VICUNA_GENERATION_CONFIG_PATH,
                VICUNA_PROMPT_FOR_NEGOTIATION,
                VicunaGenerationConfig,
                VicunaGeneration
            ],
            LLAMA3: [
                LLAMA3_GENERATION_CONFIG_PATH,
                LLAMA3_PROMPT_FOR_NEGOTIATION,
                Llama3ConfigForGeneration,
                Llama3Generation
            ],
            FPT: [
                FPT_GENERATION_CONFIG_PATH,
                FPT_PROMPT_FOR_NEGOTIATION,
                FPTConfigForGeneration,
                FPTGeneration
            ]
        }
        # a list contains generation packages
        generation_packages = []

        # loop over text generation names
        for name in names:
            package = generation_dict[name]
            generation_packages.append(package)

        # return the generaton packages
        return generation_packages

    # emotional support conversation
    elif scenario == EMOTIONAL_SUPPORT:
        generation_dict = {
            BART_GENERATION: [
                BART_GENERATION_CONFIG_PATH,
                BARTGenerationConfig,
                BARTModelForGeneration,
                BARTTrainerForGeneration,
                BARTPipelineForGeneration,
                BARTGeneration,

            ],
            CHATGPT: [
                CHATGPT_GENERATION_CONFIG_PATH,
                CHATGPT_PROMPT_FOR_EMOTIONAL_SUPPORT,
                ChatGPTConfigForGeneration,
                ChatGPTGeneration
            ],
            VICUNA: [
                VICUNA_GENERATION_CONFIG_PATH,
                VICUNA_PROMPT_FOR_EMOTIONAL_SUPPORT,
                VicunaGenerationConfig,
                VicunaGeneration
            ],
            LLAMA3: [
                LLAMA3_GENERATION_CONFIG_PATH,
                LLAMA3_PROMPT_FOR_EMOTIONAL_SUPPORT,
                Llama3ConfigForGeneration,
                Llama3Generation
            ],
            FPT: [
                FPT_GENERATION_CONFIG_PATH,
                FPT_PROMPT_FOR_EMOTIONAL_SUPPORT,
                FPTConfigForGeneration,
                FPTGeneration
            ]
        }
        # a list contains generation packages
        generation_packages = []

        # loop over text generation names
        for name in names:
            package = generation_dict[name]
            generation_packages.append(package)

        # return the generaton packages
        return generation_packages
    else:
        raise Exception('Invalid Scenario ......')


def create_user_simulators(simulator_class, user_profiles, saved_filed_path=None):
    """
    function that create a set of user simulators by using given user profiles and scenario name
    :param simulator_class: the class of the user simulator
    :param user_profiles: list contain user profiles
    :param saved_filed_path: Saved file path
    :return: a list of instances of simulators
    """
    # create a set of simulators
    user_simulators = []

    # create a set of simulator based on sampled dev user profiles
    for profile in tqdm(user_profiles):
        simulator = simulator_class(profile)
        user_simulators.append(simulator)

    # save the user simulators to file
    if saved_filed_path is not None:
        with open(saved_filed_path, 'wb') as f:
            pickle.dump(user_simulators, f)
    return user_simulators


def load_user_simulators(simulator_file_path):
    """
    function that load user simulators from file
    :param simulator_file_path: the path to the simulator file
    :return: a list of instances of simulators
    """
    with open(simulator_file_path, 'rb') as f:
        user_simulators = pickle.load(f)
        return user_simulators


def read_results(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data
