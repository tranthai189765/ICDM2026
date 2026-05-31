USER_TOKEN = "[USER]"
SYSTEM_TOKEN = "[SYSTEM]"
KNOW_TOKEN = "[KNOW]"
PATH_TOKEN = "[PATH]"
SEP_TOKEN = "[SEP]"
PROFILE_TOKEN = "[PROFILE]"
CONTEXT_TOKEN = "[CONTEXT]"
GOAL_TOKEN = "[GOAL]"
TARGET = "[TARGET]"
TOPIC_TOKEN = "[TOPIC]"
PAD_TOKEN = "<pad>"
IGNORE_INDEX = -100

# special tokens for negotiation
BUYER_TOKEN = "[BUYER]"
SELLER_TOKEN = "[SELLER]"

# special tokens for emotional support
SEEKER_TOKEN = "[SEEKER]"
SUPPORTER_TOKEN = "[SUPPORTER]"

rec_special_tokens_dict = {
    'additional_special_tokens': [USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN,
                                  CONTEXT_TOKEN, GOAL_TOKEN, TARGET],
}

neg_special_tokens_dict = {
    'additional_special_tokens': [SELLER_TOKEN, BUYER_TOKEN, PATH_TOKEN, SEP_TOKEN, CONTEXT_TOKEN, GOAL_TOKEN],
}

es_special_tokens_dict = {
    'additional_special_tokens': [SEEKER_TOKEN, SUPPORTER_TOKEN, PATH_TOKEN, SEP_TOKEN, CONTEXT_TOKEN, GOAL_TOKEN],
}

DURECDIAL_TARGET_GOALS = [
    "Movie recommendation",
    "Food recommendation",
    "Music recommendation",
    "POI recommendation",
]

DURECDIALGOALS = {
    'Ask about weather',
    'Play music',
    'Q&A',
    'Music on demand',
    'Movie recommendation',
    'Chat about stars',
    'Say goodbye',
    'Music recommendation',
    'Ask about date',
    'Ask questions',
    'Greetings',
    'POI recommendation',
    'Food recommendation',
}

BIG5_PERSONALITY = [
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
]
DECISION_MAKING_STYLE = [
    "directive", "analytical", "conceptual", "behavioral"
]

USER = 'user'
ASSISTANT = "assistant"

# scenario names
RECOMMENDATION = 'recommendation'
NEGOTIATION = 'negotiation'
EMOTIONAL_SUPPORT = 'emotional_support'

# logger names
TERMINAL_LOGGER = 'terminal'
FILE_LOGGER = 'file'
WANDB_LOGGER = 'wandb'

# datasets for recommendation
DURECDIAL = 'durecdial'
INSPIRED = 'inspired'

# datasets for negotiation
CRAIGSLIST_BARGAIN = 'craigslist_bargain'

# dataset for emotional support
ES_CONV = "es_conv"

# configuration of datasets
DURECDIAL_CONFIG_PATH = 'config/datasets/durecdial.yaml'
INSPIRED_CONFIG_PATH = 'config/datasets/inspired.yaml'
CRAIGSLIST_BARGAIN_CONFIG_PATH = 'config/datasets/craigslist_bargain.yaml'
ES_CONV_CONFIG_PATH = 'config/datasets/es_conv.yaml'

RECOMMENDATION_CONFIG_PATH = 'config/scenario/recommendation.yaml'
NEGOTIATION_CONFIG_PATH = 'config/scenario/negotiation.yaml'
EMOTIONAL_SUPPORT_CONFIG_PATH = "config/scenario/emotional_support.yaml"

BART_GENERATION = 'bart_gen'
BART_GENERATION_CONFIG_PATH = 'config/generation/BART.yaml'

BERT = 'bert'
BERT_CONFIG_PATH = 'config/models/BERT.yaml'

BART = 'bart'
BART_CONFIG_PATH = 'config/models/BART.yaml'

# rtcp model
RTCP = 'rtcp'
RTCP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/RTCP_REC.yaml'
RTCP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/RTCP_NEG.yaml'
RTCP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/RTCP_ES.yaml'

# unimind model
UNIMIND = 'unimind'
UNIMIND_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/UNIMIND_REC.yaml'

# color model
COLOR = 'color'
COLOR_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/COLOR_REC.yaml'

# ppdpp model
PPDPP = 'ppdpp'
PPDPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/PPDPP_REC.yaml'
PPDPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/PPDPP_NEG.yaml'
PPDPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/PPDPP_ES.yaml'

# trip model
TRIP = 'trip'
TRIP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/TRIP_REC.yaml'
TRIP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/TRIP_NEG.yaml'
TRIP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/TRIP_ES.yaml'

# DPDP model
DPDP = 'dpdp'
DPDP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/DPDP_REC.yaml'
DPDP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/DPDP_NEG.yaml'
DPDP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/DPDP_ES.yaml'

PREFERENCE_PPDPP = 'preference_ppdpp'
PREFERENCE_PPDPP_CONFIG_PATH = 'config/models/PREFERENCE_PPDPP.yaml'

# MODPL old version
MODPL = 'modpl'
MODPL_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/MODPL_REC.yaml'
MODPL_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/MODPL_NEG.yaml'
MODPL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/MODPL_ES.yaml'

# contextual MODPL
PADPP = "padpp"
PADPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/PADPP_REC.yaml'
PADPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/PADPP_NEG.yaml'
PADPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/PADPP_ES.yaml'

# Set Max PADPP
SMP_PADPP = "smp_padpp"
SMP_PADPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/SMP_PADPP_REC.yaml'
SMP_PADPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/SMP_PADPP_NEG.yaml'
SMP_PADPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/SMP_PADPP_ES.yaml'

# Min Dist PADPP
MIN_DIST_PADPP = "min_dist_padpp"
MIN_DIST_PADPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/MIN_DIST_PADPP_REC.yaml'
MIN_DIST_PADPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/MIN_DIST_PADPP_NEG.yaml'
MIN_DIST_PADPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/MIN_DIST_PADPP_ES.yaml'

# DDQL
DDQL = "ddql"
DDQL_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/DDQL_REC.yaml'
DDQL_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/DDQL_NEG.yaml'
DDQL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/DDQL_ES.yaml'

# Envelope
ENVELOPE = "envelope"
ENVELOPE_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/ENVELOPE_REC.yaml'
ENVELOPE_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/ENVELOPE_NEG.yaml'
ENVELOPE_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/ENVELOPE_ES.yaml'


# proactive chain-of-thought (ProCOT)
PRO_COT = "pro_cot"
PRO_COT_CONFIG_PATH = "config/models/PRO_COT.yaml"

# standard prompting
STANDARD = "standard"
STANDARD_CONFIG_PATH = "config/models/STANDARD.yaml"

# ICL_AIF
ICL_AIF = "icl_aif"
ICL_AIF_CONFIG_PATH = "config/models/ICL_AIF.yaml"

# Proactive
PROACTIVE = "proactive"
PROACTIVE_CONFIG_PATH = "config/models/PROACTIVE.yaml"

# Ask-an-Expert
ANE = 'ane'
ANE_CONFIG_PATH = "config/models/ANE.yaml"

# GDP-Zero
GDP_ZERO = 'gdp_zero'
GDP_ZERO_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/GDP_ZERO_NEG.yaml'
GDP_ZERO_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/GDP_ZERO_ES.yaml'

# DMORL – Dynamic Multi-Objective Reinforcement Learning
DMORL = 'dmorl'
DMORL_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/DMORL_REC.yaml'
DMORL_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/DMORL_NEG.yaml'
DMORL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/DMORL_ES.yaml'

# H-MOD – buyer-objective H-MOD training/evaluation on top of DMORL/PADPP
HMOD = 'hmod'
HMOD_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/HMOD_NEG.yaml'

# types of evaluators
OFFLINE = 'offline'
ONLINE = 'online'

# metrics
ACCURACY = 'acc'
PRF1 = 'prf1'
BLEU_N = 'bleu_n'
ROUGE_N = 'rouge_n'
AVG_TURN = 'avg_turn'
DIST_N = 'dist_n'

# success rate
SUCCESS_RATE = 'sr'

# H-MOD buyer-agent evaluation metrics
LLM_SUCCESS_RATE = 'llm_sr'
GOAL_SUCCESS_RATE = 'gsr'
T2DA = 't2da'
CVR = 'cvr'
BLOCKED_CVR = 'blocked_cvr'
ACTUAL_CVR = 'actual_cvr'

# deal rate
DEAL_RATE = 'deal_rate'

# User statisfaction
USER_REWARD = 'user_reward'

# objectives for recommendation
ITEM_FREQ = 'item_freq'

# objectives for negotiation
SL_RATIO = 'sl_ratio'
FAIRNESS = 'fairness'

# objectives for emotional support
TOXICITY = 'toxicity'

# llama 3
LLAMA3 = "llama3"
LLAMA3_GENERATION_CONFIG_PATH = 'config/generation/LLAMA3.yaml'
LLAMA3_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# FPT-hosted Llama-3.3-70B (OpenAI-compatible API; credentials from .env)
FPT = "fpt"
FPT_GENERATION_CONFIG_PATH = 'config/generation/FPT.yaml'

# Local Qwen2.5-14B-Instruct (Alibaba). Sweet spot ~14B for A100 40GB in BF16
# without needing quantization. Uses the same response-generation prompts as
# Llama-3 (they're Instruct templates handled by the tokenizer's chat template).
QWEN = "qwen"
QWEN_GENERATION_CONFIG_PATH = 'config/generation/QWEN.yaml'
QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# prompts for llm generation
CHATGPT = 'chatgpt'
CHATGPT_GENERATION_CONFIG_PATH = 'config/generation/CHATGPT.yaml'

VICUNA = 'vicuna'
VICUNA_GENERATION_CONFIG_PATH = 'config/generation/VICUNA.yaml'

# the prompts for chat gpt
# including prompts for recommendation, negotiation and emotional support
CHATGPT_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": "You are the recommender who is trying to recommend an item to the user. "
                "Please reply with only one short and succinct sentence. {}."
     },
]

CHATGPT_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a Buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the Buyer who is trying to buy the {} with the price of {}. Product description: {} \nPlease "
                "reply with only one short and succinct sentence. {}"
     }
]

CHATGPT_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},

    {"role": "user",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {}."
     }
]

# prompts for the llama 3 mode
# including the recommendation, negotiation and emotional support
LLAMA3_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": f"You are the recommender who is trying to recommend an item to the user. "
                "Your topic sets: {}."
                "Please reply with only one short and succinct sentence. {}"
     },
]

LLAMA3_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the buyer who is trying to buy the {} with the price of {}. Product description: {} \n . "
                "Please reply with only one short and succinct sentence. {}"
     }
]

LLAMA3_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},

    {"role": "user",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {}"
     }
]

# FPT model reuses the same prompts as LLaMA3 (it also serves a Llama-family model)
FPT_PROMPT_FOR_RECOMMENDATION = LLAMA3_PROMPT_FOR_RECOMMENDATION
FPT_PROMPT_FOR_NEGOTIATION = LLAMA3_PROMPT_FOR_NEGOTIATION
FPT_PROMPT_FOR_EMOTIONAL_SUPPORT = LLAMA3_PROMPT_FOR_EMOTIONAL_SUPPORT

# Qwen2.5 Instruct uses standard system/user prompts; reuse the Llama-3 prompts
# (the chat template handles model-specific formatting at tokenization time).
QWEN_PROMPT_FOR_RECOMMENDATION = LLAMA3_PROMPT_FOR_RECOMMENDATION
QWEN_PROMPT_FOR_NEGOTIATION = LLAMA3_PROMPT_FOR_NEGOTIATION
QWEN_PROMPT_FOR_EMOTIONAL_SUPPORT = LLAMA3_PROMPT_FOR_EMOTIONAL_SUPPORT

# the prompts for vicuna model
# including prompts for recommendation, negotiation and emotional support
VICUNA_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode."
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": "Please reply with only one short and succinct sentence. {} . Now start the game."
     }
]

VICUNA_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the buyer who is trying to buy the {} with the price of {}. Product description: {}\nPlease "
                "reply with only one short and succinct sentence. {} Now start the game."
     }
]

VICUNA_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},
    {"role": "assistant",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {} .Are you ready to play the game?"},
    {"role": "assistant", "content": "Yes, I'm ready to play the game!"}

]

LLM_MODEL = "gpt-3.5-turbo"

# a mapping from goal to textual description
# for the negotiation task
NEGOTIATION_GOAL2DESCRIPTION = {'greet': 'Please say hello or chat randomly.',
                                'inquire': 'Please ask any question about product, year, price, usage, etc.',
                                'inform': 'Please provide information about the product, year, usage, etc.',
                                'propose': 'Please initiate a price or a price range for the product.',
                                'counter': 'Please propose a new price or a new price range.',
                                'counter-noprice': 'Please propose a vague price by using comparatives with existing price.',
                                'confirm': 'Please ask a question about the information to be confirmed.',
                                'affirm': 'Please give an affirmative response to a confirm.',
                                'deny': 'Please give a negative response to a confirm.',
                                'agree': 'Please agree with the proposed price.',
                                'disagree': 'Please disagree with the proposed price.',
                                # Option B additions (paper-specific buyer tactics)
                                'walk_away': 'Please announce that you are walking away from the negotiation.',
                                'final_offer': 'Please make a take-it-or-leave-it final price offer.',

                                # for standard prompting. There is no instruction for dialogue strategy
                                'Standard': ""}

# a mapping from goal to textual description
# for the recommendation task
DURECDIAL_GOAL2DESCRIPTION = {'Ask about weather': 'Please provide information about the weather.',
                              'Play music': 'Please select an appropriate song from your given topic set and reply that song is playing.',
                              'Music recommendation': 'Please recommend the song \"{}\" to the user',
                              'Q&A': 'Please answer questions asked by the user.',
                              'Chat about stars': "Please select an appropriate movie star from your given topic set and provide information about the movie star.",
                              'Music on demand': 'Please select an appropriate song from your given topic set and reply that song is suitable for the user demand',
                              'Movie recommendation': 'Please recommend the movie \"{}\" to the user.',
                              'Say goodbye': 'Please say goodbye to the user.',
                              'Ask about date': 'Please provide information regarding date.',
                              'Ask questions': 'Please select an appropriate topic from your given topic set and ask questions regarding that topic',
                              'Greetings': 'Please say hello or chat randomly.',
                              'POI recommendation': 'Please recommend the restaurant \"{}\" to the user.',
                              'Food recommendation': 'Please recommend the food \"{}\" to the user.'}


# a mapping from goal to textual description
# for the emotional support task
ES_CONV_GOAL2DESCRIPTION = {"Question": "Please ask the Patient to elaborate on the situation they just described.",
                            "Self-disclosure": "Please provide a statement relating to the Patient about the situation they just described.",
                            "Affirmation and Reassurance": "Please provide affirmation and reassurance to the Patient on the situation they just described.",
                            "Providing Suggestions": "Please provide suggestion to the Patient on the situation they just described.",
                            "Others": "Please chat with the Patient.",
                            "Reflection of feelings": "Please acknowledge the Patient's feelings about the situation they described.",
                            "Information": "Please provide factual information to help the Patient with their situation.",
                            "Restatement or Paraphrasing": "Please acknowledge the Patient's feelings by paraphrasing their situation.",

                            # for standard prompting. There is no instruction for dialogue strategy
                            "Standard": ""}

# a mapping from goal to textual description
# for the recommendation task, inspired dataset
INSPIRED_GOAL2DESCRIPTION = {
    'opinion_inquiry': 'Please asks for user’s opinion on the \"{}\" movie-related attributes.',
    'acknowledgment': 'Please acknowledge the user preference.',
    'no_strategy': 'Please randomly respond the user.',
    'encouragement': 'Please praise of the user’ movie taste and encouragement to watch \"{}\"',
    'personal_opinion': "Please express your subjective opinion about \"{}\", including its plot, actors, or other movie attributes.",
    'rephrase_preference': 'Please rephrase the user preference for confirmation.',
    'transparency': 'Please disclose your thinking process of understanding the user’ preference.',
    'offer_help': 'Please disclose explicit intention to help the user or being transparent.',
    'personal_experience': 'Please share your personal experience related to a movie.',
    'experience_inquiry': 'Please ask for user’s experience on movie watching, such as whether the user has watched a certain movie or not.',
    'self_modeling': 'Please becomes a role model to do something first so that the user would follow.',
    'preference_confirmation': 'Please ask or rephrase the user’s preference.',
    'similarity': 'Please being like-minded toward the users about their movie preference to produce similarity among them.',
    'credibility': 'provide provide factual information about movie attributes, such as the plot, actors, or awards that \"{}\" has'
}
