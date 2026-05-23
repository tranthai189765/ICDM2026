import copy
import re

import numpy as np
from loguru import logger

from config.constants import *


def convert_list_to_str(knowledge):
    """
    function that convert a list of 3 elements to a text string
    @param knowledge: a list of 3 element where each element is a string
    @return: a text string
    """
    if len(knowledge) == 0:
        return ""
    return f"{knowledge[0]} {knowledge[1]} {knowledge[2]}"


def convert_example_to_feature_for_generation_recommendation(tokenizer, instance, max_sequence_length=512,
                                                             max_target_length=50,
                                                             is_test=False, dataset=DURECDIAL):
    """
    function that convert an instance to input and labels for a response generation model (under the recommendation scenario)
    @param tokenizer: a huggingface tokenizer
    @param instance: an instance from the data.
    @param max_sequence_length: the maximum length of the input sequence.
    @param max_target_length: the maximum length of the target response
    @param is_test: True if inference or False if training.
    @return: an input sequence and its corresponding labels.
    """
    # construct the dialogue context
    dialogue_context = instance['dialogue_context']
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    # training the model, using ground truth data
    if not is_test:
        # ground truth goal for training the model
        goal = instance['goal']
        topic = instance['topic']
        knowledge = instance['knowledge']
    else:
        # predicted goal for the inference step
        goal = instance['pred_goal']
        topic = instance['pred_topic']
        # knowledge = instance['pred_know']
        knowledge = ''

    # construct the knowledge part
    if not isinstance(knowledge, str):
        knowledge_str = convert_list_to_str(knowledge)
    else:
        knowledge_str = knowledge

    # the targeted item
    target = instance['task_background']['target_topic']

    # construct the input sequence for response generation task
    if dataset == DURECDIAL:
        input_str = f"{GOAL_TOKEN}: {goal} {TOPIC_TOKEN}: {topic} {KNOW_TOKEN}: {knowledge_str}  {CONTEXT_TOKEN}: {dialogue_str}"
    # for inspired dataset
    elif dataset == INSPIRED:
        input_str = f"{GOAL_TOKEN}: {goal} {TARGET}: {target} {CONTEXT_TOKEN}: {dialogue_str}"

    # tokenize and convert the input sequence to input ids
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # construct the label for response generation task
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: " + instance['response']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_generation_negotiation(tokenizer, instance, max_sequence_length=512,
                                                          max_target_length=50,
                                                          is_test=False, dataset=CRAIGSLIST_BARGAIN):
    # construct the dialogue context
    dialogue_context = instance['dialogue_context']
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += SELLER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += BUYER_TOKEN
        dialogue_str += utt['content']

    # training the model, using ground truth data
    if not is_test:
        # ground truth goal for training the model
        goal = instance['goal']
    else:
        # predicted goal for the inference step
        goal = instance['pred_goal']

    # construct the input sequence for response generation task
    if dataset == CRAIGSLIST_BARGAIN:
        input_str = f"{GOAL_TOKEN}: {goal} {CONTEXT_TOKEN}: {dialogue_str}"
    # for inspired dataset
    else:
        raise Exception('The dataset is invalid')

    # tokenize and convert the input sequence to input ids
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # construct the label for response generation task
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{BUYER_TOKEN}: " + instance['response']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_generation_emotional_support(tokenizer, instance, max_sequence_length=512,
                                                                max_target_length=50,
                                                                is_test=False, dataset=ES_CONV):
    # construct the dialogue context
    dialogue_context = instance['dialogue_context']
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += SEEKER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SUPPORTER_TOKEN
        dialogue_str += utt['content']

    # training the model, using ground truth data
    if not is_test:
        # ground truth goal for training the model
        goal = instance['goal']
    else:
        # predicted goal for the inference step
        goal = instance['pred_goal']

    # construct the input sequence for response generation task
    if dataset == ES_CONV:
        input_str = f"{GOAL_TOKEN}: {goal} {CONTEXT_TOKEN}: {dialogue_str}"
    # for inspired dataset
    else:
        raise Exception('The dataset is invalid')

    # tokenize and convert the input sequence to input ids
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # construct the label for response generation task
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SUPPORTER_TOKEN}: " + instance['response']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def construct_prompt_for_chat_gpt_response_generation_recommendation(state, prompt, dataset='durecdial'):
    """
    emthod that constructs the prompt for chatgpt response generation for the recommendation scenario
    :param state: the current state of the conversation
    :param prompt: the give prompt, should be different for different scenario
    :return:
    """
    
    new_prompt = copy.deepcopy(prompt)
    # predicted goal and topic
    pred_goal = state['pred_goal']
    
    # target goal and topic
    target_topic = state['task_background']['target_topic']
    target_goal = state['task_background']['target_goal']
    topic_set = state['task_background']['topic_set']
        
    if dataset == DURECDIAL:
        goal2description = DURECDIAL_GOAL2DESCRIPTION
        target_goals = DURECDIAL_TARGET_GOALS
    elif dataset == INSPIRED:
        goal2description = INSPIRED_GOAL2DESCRIPTION
    
    # handling exception cases
    try:
        # one of recommendation goals
        if pred_goal.strip() in target_goals:
            # the pred goal is the target goal of this case.
            if pred_goal.strip().lower() == target_goal.strip().lower() :
                strategy_description = goal2description[pred_goal.strip()].format(target_topic)
            # wrong target goal.
            else:
                # the goal is food recommendation
                if pred_goal.strip() == "Food recommendation":
                    strategy_description = "Based on the conversation and your topic set, please recommend an appropriate food to the user."
                # the goal is music recommendation
                elif pred_goal.strip() == "Music recommendation":
                    strategy_description = "Based on the conversation and your topic set, please recommend an appropriate song to the user."
                # the goal is movie recommendation
                elif pred_goal.strip() == "Movie recommendation":
                    strategy_description = "Based on the conversation and your topic set, please recommend an appropriate movie to the user."
                # the goal is POI recommendation
                elif pred_goal.strip() == "POI recommendation":
                    strategy_description = "Based on the conversation and your topic set, please recommend an appropriate restaurant to the user."
                else:
                    raise Exception("Something is wrong here.")
        # other strategies.
        else:
            strategy_description = goal2description[pred_goal.strip()]
    except:
        strategy_description = ''
    
    # constructing the prompt
    # using the predicted goal and topic
    new_prompt[-1]['content'] = new_prompt[-1]['content'].format(topic_set, strategy_description)
    # print(new_prompt)
    return new_prompt, strategy_description


_PRICE_PATTERN = re.compile(r"\$?\d[\d,]*\.?\d*")


def _seller_has_offered_price(state) -> bool:
    """Return True only if the seller has made a real counter-offer DURING
    negotiation. The game initialises dialogue_context with a fixed pair
    [assistant: 'how much?', user: 'this is a good X and price is $N'] —
    that opening listing is dataset boilerplate, not a real negotiated
    offer. Treating it as one lets 'agree' at the very first agent turn
    capitulate to the seller's listed price (sl_ratio ~ 0).

    We skip the first two boilerplate turns and only count a user turn as
    "a real offer" if it appears at index >= 2 (i.e. user response AFTER
    the agent's first negotiating move)."""
    dialogue = state.get('dialogue_context', [])
    for turn in dialogue[2:]:
        if turn.get('role') == 'user':
            content = turn.get('content', '') or ''
            if _PRICE_PATTERN.search(content):
                return True
    return False


def construct_prompt_for_chat_gpt_response_generation_negotiation(state, prompt):
    """
    method that construct the prompt for chatgpt response generation for the negotiation scenario
    :param state: the current state of the conversation
    :param prompt: the given prompt, should be different for different scenarios
    :return:
    """
    new_prompt = copy.deepcopy(prompt)
    pred_goal = state['pred_goal']
    # coarse-grained dialogue strategy:
    # PPDPP, ProCOT, MODPL, etc.
    # those models does not explicitly incorporate an explicit price.
    if pred_goal in NEGOTIATION_GOAL2DESCRIPTION:
        goal_description = NEGOTIATION_GOAL2DESCRIPTION[pred_goal]
    # other model such as ICL_AIF
    else:
        # response generation for MODPL
        if isinstance(pred_goal, tuple):
            strategy, bin_pred = pred_goal
            bin_num = 5
            # t = np.random.randn(1)
            t = 0
            bin_width = (state['task_background']['seller_price'] - state['task_background']['buyer_price']) / bin_num

            # proposing a price based on the predicted bin and a random ratio
            proposed_priced = int(state['task_background']['buyer_price'] + bin_width * (bin_pred + t))
            goal_description = NEGOTIATION_GOAL2DESCRIPTION[strategy]
            
            # following https://arxiv.org/pdf/2010.09954
            # 4 actions propose, counter, agree and disagree must to be followed by a particular price.
            if strategy == "propose":
                goal_description = (
                    f". Please propose the price of ${proposed_priced} explicitly. "
                    "Do not accept any other price."
                )
            elif strategy == "counter":
                goal_description = (
                    f". Please counter the seller with the price of ${proposed_priced} explicitly. "
                    "Do not accept any other price."
                )
            elif strategy == "agree":
                # 'agree' must accept the seller's most recent offered price.
                # BUT if no seller offer exists yet (e.g. turn 0), the LLM would
                # hallucinate a price (often the seller's listed asking price)
                # and the simulator would confirm it — giving the agent a
                # zero-effort deal at the seller's full ask (sl_ratio = 0).
                # When this happens we fall back to 'propose' so the agent must
                # actually open the negotiation instead of capitulating.
                has_seller_offer = _seller_has_offered_price(state)
                if has_seller_offer:
                    goal_description = (
                        ". Please clearly ACCEPT the seller's most recent offered price. "
                        "State 'I accept your offer of $<their last price>' and do not propose a different price."
                    )
                else:
                    goal_description = (
                        f". Please propose the price of ${proposed_priced} explicitly. "
                        "Do not accept any other price. The seller has not yet stated a price, "
                        "so you cannot accept anything; open the negotiation with your own offer."
                    )
            elif strategy == "disagree":
                goal_description = (
                    ". Please clearly REJECT the seller's offer without proposing a new price. "
                    "State that the price is unacceptable."
                )
            logger.debug(f"goal_description: {goal_description}")
        else:
            goal_description = pred_goal

    new_prompt[1]['content'] = new_prompt[1]['content'].format(state['task_background']['item_name'],
                                                               state['task_background']['buyer_price'],
                                                               state['task_background']['buyer_item_description'],
                                                               goal_description)
    return new_prompt, goal_description


def construct_prompt_for_chat_gpt_response_generation_emotional_support(state, prompt):
    """
    method that construct the prompt for chatgpt response generation for the emotional support conversation
    :param state: the current state of the conversation
    :param prompt: the given prompt, should be different for different scenarios
    :return:
    """
    new_prompt = copy.deepcopy(prompt)
    pred_goal = state['pred_goal']
    # coarse-grained dialogue strategy:
    # PPDPP, ProCOT, MODPL, etc.
    if pred_goal in ES_CONV_GOAL2DESCRIPTION:
        goal_description = ES_CONV_GOAL2DESCRIPTION[pred_goal]
    # other model using dialogue-level strategies such as ICL-AIF
    else:
        goal_description = pred_goal

    new_prompt[1]['content'] = new_prompt[1]['content'].format(goal_description)
    return new_prompt, goal_description


def construct_prompt_for_vicuna_response_generation_negotiation(state, prompt):
    """
    method that construct the prompt for vicuna response generation in the negotiation scenario
    :param state: the current state of the conversation
    :param prompt: the given prompt, should be different for different scenarios
    :return:
    """
    # construct the new prompt by filling in specific information
    new_prompt = copy.deepcopy(prompt)
    pred_goal = state['pred_goal']
    new_prompt[-1]['content'] = new_prompt[-1]['content'].format(state['task_background']['item_name'],
                                                                 state['task_background']['buyer_price'],
                                                                 state['task_background']['buyer_item_description'],
                                                                 NEGOTIATION_GOAL2DESCRIPTION[pred_goal])

    # combine prompt and dialogue context
    new_prompt.extend(state['dialogue_context'])

    # construct the input string for the vicuna model
    # using the current dialoge context
    seps = [' ', '</s>']
    role = "assistant"
    ret = new_prompt[0]['content'] + seps[0]
    for i, message in enumerate(new_prompt[1:]):
        role_text = message['role']
        ret += role_text + ": " + message['content'] + seps[i % 2]
    ret += '%s:' % role
    return ret


def construct_prompt_for_vicuna_response_generation_recommendation(state, prompt):
    """
    method that construct the prompt for vicuna response generation in the recommendation scenario
    :param state: the current state of the conversation
    :param prompt: the given prompt, should be different for different scenarios
    :return:
    """
    # construct the new prompt by filling in specific information
    new_prompt = copy.deepcopy(prompt)

    # the predicted goal and topic
    pred_goal = state['pred_goal']
    pred_topic = state['pred_topic']
    new_prompt[-1]['content'] = new_prompt[-1]['content'].format(DURECDIAL_GOAL2DESCRIPTION[pred_goal].format(
        pred_topic))

    # combine prompt and dialogue context
    new_prompt.extend(state['dialogue_context'])

    # construct the input string for the vicuna model
    # using the current dialoge context
    seps = [' ', '</s>']
    role = SYSTEM_TOKEN
    ret = new_prompt[0]['content'] + seps[0]
    for i, message in enumerate(new_prompt[1:]):
        role_text = ''
        if message['role'] == 'user':
            role_text = USER_TOKEN
        elif message['role'] == 'assistant':
            role_text = SYSTEM_TOKEN
        ret += role_text + ": " + message['content'] + seps[i % 2]
    ret += '%s:' % role
    return ret


def construct_prompt_for_vicuna_response_generation_emotional_support(state, prompt):
    """
    method that construct the prompt for vicuna response generation for emotional support
    :param state: the current state of the current conversation
    :param prompt: a predefined  prompt that contains place holders
    :return:
    """
    # construct the new prompt by filling in specific information
    new_prompt = copy.deepcopy(prompt)
    pred_goal = state['pred_goal']
    new_prompt[1]['content'] = new_prompt[1]['content'].format(ES_CONV_GOAL2DESCRIPTION[pred_goal])

    # combine prompt and dialogue context
    new_prompt.extend(state['dialogue_context'])

    # construct the input string for the vicuna model
    # using the current dialoge context
    seps = [' ', '</s>']
    role = "assistant"
    ret = new_prompt[0]['content'] + seps[0]
    for i, message in enumerate(new_prompt[1:]):
        role_text = message['role']
        ret += role_text + ": " + message['content'] + seps[i % 2]
    ret += '%s:' % role
    return ret
