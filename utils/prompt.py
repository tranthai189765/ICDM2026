import os
import re
import copy

from dotenv import load_dotenv
import openai

from googleapiclient import discovery
import json

import transformers
from transformers import pipeline
import torch

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

from config.constants import LLM_MODEL, LLAMA3, CHATGPT, LLAMA3_MODEL, FPT, QWEN, QWEN_MODEL

load_dotenv()


@retry(
    retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError,
                                   openai.APIStatusError, openai.APITimeoutError)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    # openai >= 1.0 client-based call (ChatGPT path is currently disabled via assert)
    _client = openai.OpenAI(api_key=API_KEY)
    return _client.chat.completions.create(**kwargs)


# load the environment variables
load_dotenv()

# The api key, llm model
API_KEY = os.getenv("API_KEY")
MODEL = LLM_MODEL
# openai.api_key is set per-call via OpenAI(api_key=...) in openai >= 1.0

# API for toxicity evaluation
PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_KEY')

# FPT cloud (OpenAI-compatible) — serves Llama-3.3-70B-Instruct
FPT_API_KEY = os.getenv("FPT_API_KEY")
FPT_API_URL = os.getenv("FPT_API_URL", "https://mkp-api.fptcloud.com/v1/chat/completions")
FPT_MODEL = os.getenv("FPT_MODEL", "Llama-3.3-70B-Instruct")
# OpenAI client wants a base_url (no /chat/completions suffix)
_FPT_BASE_URL = FPT_API_URL.rsplit("/chat/completions", 1)[0]
_fpt_client = None


def _get_fpt_client():
    global _fpt_client
    if _fpt_client is None:
        _fpt_client = openai.OpenAI(api_key=FPT_API_KEY, base_url=_FPT_BASE_URL)
    return _fpt_client


def call_fpt_model(messages, temperature=0.0, max_token=256, n_return_sequences=1):
    """Call FPT-hosted Llama-3.3-70B-Instruct via the OpenAI-compatible API."""
    client = _get_fpt_client()
    results = []
    for _ in range(n_return_sequences):
        resp = client.chat.completions.create(
            model=FPT_MODEL,
            messages=messages,
            temperature=max(temperature, 0.01),  # FPT rejects temperature=0 in some cases
            max_tokens=max_token,
        )
        results.append(resp.choices[0].message.content.strip())
    return results[0] if n_return_sequences == 1 else results

# Lazy-loaded globals — not instantiated until first use so that runs using
# DeepInfra/ChatGPT don't pay the cost of downloading large local models.
_local_pipelines = {}      # model_id -> (pipeline, terminators)
_sentiment_pipeline = None


def _detect_terminators(tokenizer):
    """Collect EOS / end-of-turn token ids for common Instruct chat templates
    (Llama-3 uses <|eot_id|>, Qwen uses <|im_end|>, etc.)."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    unk = tokenizer.unk_token_id
    for tok in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != unk:
            ids.add(tid)
    return list(ids)


def _get_local_pipeline(model_id=None, load_mode=None):
    """Load and cache a HF text-generation pipeline for any Instruct model
    (Llama-3, Qwen2.5, Mistral, ...). End-of-turn tokens are auto-detected
    from the tokenizer so the same loader works across families.

    Args:
      model_id: HF model id; defaults to LLAMA3_MODEL.
      load_mode: 'bf16' (default) | '4bit'/nf4 | '8bit'.
                 Overridable via env LLAMA_LOCAL_LOAD.
    """
    global _local_pipelines
    model_id = model_id or LLAMA3_MODEL
    load_mode = (load_mode or os.getenv("LLAMA_LOCAL_LOAD", "bf16")).lower()

    if model_id in _local_pipelines:
        return _local_pipelines[model_id]

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if load_mode in ("4bit", "int4", "nf4"):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_mode in ("8bit", "int8"):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
        device_map="auto",
    )
    terminators = _detect_terminators(pipe.tokenizer)
    _local_pipelines[model_id] = (pipe, terminators)
    return pipe, terminators


def _get_llama_pipeline():
    """Backward-compat shim: load the default Llama-3 pipeline."""
    return _get_local_pipeline(LLAMA3_MODEL)


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
    return _sentiment_pipeline


def _call_local_lm(model_id, prompt, temperature=0.0, max_token=30, n_return_sequences=1):
    """Shared local-LM call path. Works with any Instruct model via auto-detected
    end-of-turn tokens (Llama-3, Qwen2.5, Mistral, ...)."""
    pipe, terminators = _get_local_pipeline(model_id)
    response = pipe(
        prompt,
        max_new_tokens=max_token,
        eos_token_id=terminators,
        do_sample=True,
        temperature=max(temperature, 0.01),  # temp=0 disallowed when do_sample=True
        top_p=0.9,
        num_return_sequences=n_return_sequences,
    )
    if n_return_sequences > 1:
        return [x["generated_text"][-1]["content"] for x in response]
    return response[0]["generated_text"][-1]["content"]


def call_llama3_model(prompt, temperature=0.0, max_token=30, n_return_sequences=1):
    """Call the local Llama-3 pipeline."""
    return _call_local_lm(LLAMA3_MODEL, prompt, temperature, max_token, n_return_sequences)


def call_qwen_model(prompt, temperature=0.0, max_token=30, n_return_sequences=1):
    """Call the local Qwen2.5 pipeline (defaults to QWEN_MODEL from constants)."""
    return _call_local_lm(QWEN_MODEL, prompt, temperature, max_token, n_return_sequences)


def reformat_demonstration(demonstration, is_agent_start=False):
    """
    function that reformat the demonstrative conversation
    @param demonstration: the given conversation
    @param is_agent_start: True if the system starts the conversation else False
    @return: the reformated demonstrative conversation
    """
    new_demonstration = []
    role = 0
    if is_agent_start:
        role = -1
    for utt in demonstration:
        if role % 2 == 0:
            new_demonstration.append({'role': 'user', 'content': utt})
        elif role == -1 or role % 2 != 0:
            new_demonstration.append({'role': 'assistant', 'content': utt})
        role += 1
    return new_demonstration


def call_llm(prompt, n=1, temperature=0.0, max_token=10, model_type='chatgpt'):
    """
    function that calls llm for n times using the given prompt
    :param prompt: the given input prompt
    :param n: number of times we call the llm
    :param temperature: the temperature we use to prompt the llm
    :param max_token: the maximum number of output tokens
    :param model_type: the name of the large language mdoel
    :return:
    """
    responses = []
    # call llm for n times
    for i in range(n):
        # the llm is the chatgpt model
        if model_type == CHATGPT:
            # call the llm with backoff
            assert 1 == 0
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_token
            )

            responses.append(response.choices[0]['message']['content'])
        # the llm is the llama 3 model
        elif model_type == LLAMA3:
            # do something here
            responses.append(call_llama3_model(prompt, temperature, max_token))
        # FPT-hosted Llama-3.3-70B-Instruct
        elif model_type == FPT:
            responses.append(call_fpt_model(prompt, temperature, max_token))
        # local Qwen2.5 (e.g. 14B-Instruct)
        elif model_type == QWEN:
            responses.append(call_qwen_model(prompt, temperature, max_token))
    return responses


def get_llm_based_assessment_for_recommendation(target_topic, simulated_conversation,
                                                demonstration=None,
                                                n=10,
                                                temperature=1.1,
                                                max_tokens=50,
                                                profile_description=None,
                                                model_type='chatgpt'):
    """
    function that computes an target-driven assessment given the current conversation
    :param target_topic: the target item
    :param simulated_conversation: the generated conversation
    :param demonstration: an demonstrative example
    :param n: the number of times we prompt the model
    :param temperature: the temperature used to prompt the llm
    :param max_tokens: the maximal number of tokens used to prompt the llm
    :return:
    """
    # messages = []
    # if demonstration is not None:
    #     system_instruction_1 = ''' This is an example of a {} conversation between an user (you) and the system.
    #     In this conversation, the user (you) accepted  the item : {}
    #     '''.format(demonstration['target_goal'], demonstration['target_topic'])
    #
    #     # the first instruction prompt
    #     messages = [
    #         {"role": "system", "content": system_instruction_1},
    #     ]
    #     # 1-shot demonstration
    #     for utt in reformat_demonstration(demonstration,
    #                                       is_agent_start=demonstration['goal_type_list'][0] == 'Greetings'):
    #         messages.append(utt)

    accept_string = "accept"
    reject_string = "reject"

    system_instruction_2 = f"""
    Based on the given conversation, please decide whether the user accepted the item: {target_topic} at the end of the conversation.
    The conversation is:
    """
    system_instruction_3 = f"""Please decide whether the user accepted the item: {target_topic} at the end of the conversation : {target_topic}. 
    Based on the give conversation, please decide whether the user is happy and willing to accept the target item: {target_topic}. 
    If the user is happy, please only generate the word: {accept_string}.
    If the user is confused or not willing to accept the item :{target_topic}, please only generate the word: {reject_string}.
    """
    # the second instruction prompt
    messages = [
        {"role": "system", "content": system_instruction_2},
    ]
    # simulated conversation
    copied_conv = copy.deepcopy(simulated_conversation)
    for utt in copied_conv:
        # switch role
        if utt['role'] == 'system':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        temp = {'role': utt['role'], 'content': utt['content']}
        messages.append(temp)

    # # llm-based target-driven assessment instruction
    # system_instruction_3 = f"""
    # Based on the given conversation, you need to infer the attitude of the user towards the
    # target item : {target_topic}. You need to infer if the user is happy and willing to accept the target item: {target_topic}.
    # If the user is happy, you need to only generate the word: {accept_string}.
    # If the user is confused or not willing to accept the item :{target_topic}, you need to only generate the word: {reject_string}.
    # """
    #
    messages.append(
        {'role': 'user', 'content': system_instruction_3}
    )

    responses = []

    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    # calling the llama 3
    elif model_type == LLAMA3:
        responses.extend(call_llama3_model(messages, temperature, max_tokens, n_return_sequences=n))
    elif model_type == FPT:
        responses.extend(call_fpt_model(messages, temperature, max_tokens, n_return_sequences=n))
    elif model_type == QWEN:
        responses.extend(call_qwen_model(messages, temperature, max_tokens, n_return_sequences=n))

    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    # convert the text-based assessment to scalar based assessment
    is_successful = 0
    for response in responses:
        if response.lower() == accept_string.lower():
            is_successful += 1

    return float(is_successful) / n


def get_llm_based_assessment_for_negotiation(simulated_conversation,
                                             n=10,
                                             temperature=1.1,
                                             max_tokens=20,
                                             model_type='chatgpt'
                                             ):
    """
    function that assesses if there is a deal between the user and the system in a negotiation conversation
    :param simulated_conversation: 
    :param n:
    :param temperature: 
    :param max_tokens: 
    :return:
    """
    # the reward computation function for negotiation scenario
    # the following code is borrowed from the PPDPP official implementation
    # evaluating the progress at the last two rounds
    dial = ''
    for utt in simulated_conversation:
        if utt['role'] == 'user':
            role = 'Seller'
        else:
            role = 'Buyer'
        dial += f"{role}: {utt['content']}"
        dial += ". "

    # construct the message to prompt the llm
    # following the prompt from PPDPP
    messages = [{"role": "system",
                 "content": "Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal."},
                {"role": "user",
                 "content": f"""You have to follow the instructions below during chat. 
                            1. Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. 
                            2. If they have reached a deal, please extract the deal price as [price]. 
                            You can only reply with one of the following sentences: "They have reached a deal at [price]". "They have not reached a deal."
                            The following is the conversation between a Buyer and a Seller: 
                            Buyer: Can we meet in the middle at 15? 
                            Seller: Deal, let's meet at 15 for this high-quality balloon.
                            Question: Have they reached a deal ? 
                            Answer: They have reached a deal at $15.
                            The following is the conversation between a Buyer and a Seller:
                            Buyer: I'd be willing to pay $5400 for the truck.
                            Seller: I'm still a bit hesitant, but I'm willing to meet you halfway at $5600.
                            Question: Have they reached a deal? 
                            Answer: They have not reached a deal.
                            The following is the conversation: {dial}\n 
                            Question: Have they reached a deal? 
                            Answer: """
                }]
    

    responses = []
    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    elif model_type == FPT:
        responses.extend(call_fpt_model(messages, temperature, max_tokens, n_return_sequences=n))
    elif model_type == QWEN:
        responses.extend(call_qwen_model(messages, temperature, max_tokens, n_return_sequences=n))
    else:
        responses.extend(call_llama3_model(messages, temperature, max_tokens, n_return_sequences=n))
    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    return responses


def get_llm_based_price_extraction_for_negotiation(buyer_utterance,
                                                   seller_price,
                                                   buyer_price,
                                                   temperature=0.0,
                                                   max_tokens=12,
                                                   model_type='chatgpt'):
    """
    Ask the LLM to extract the price the BUYER is committing to in a single
    utterance. Returns a float price or None (no commitment / not a number).

    The agent in this scenario is the Buyer, so when an utterance mentions
    several numbers (the buyer's offer, the seller's listing, a model year,
    a quantity, ...) we want the one the buyer is *offering to pay*. A simple
    max()/min() heuristic mis-fires; this routine delegates the disambiguation
    to the LLM.

    :param buyer_utterance: the agent's latest generated utterance (string)
    :param seller_price:    seller's listed/ask price (for context)
    :param buyer_price:     buyer's target price (for context)
    :param model_type:      'chatgpt' | 'fpt' | 'qwen' | 'llama3'
    :return: float | None
    """
    messages = [
        {"role": "system",
         "content": "You extract the price a Buyer offers to pay from a single chat message in a price negotiation."},
        {"role": "user",
         "content": f"""In this negotiation the Seller lists the item at ${seller_price} and the Buyer's target is around ${buyer_price}.
Read the Buyer's message below and output ONLY the price (a single number, no currency sign) that the Buyer is offering to pay or agreeing to in THIS message.
Rules:
- If the Buyer is rejecting/quoting the seller's price but not offering their own, or makes no price offer, output NONE.
- Ignore numbers that are model years, quantities, dates, or references to the seller's ask.
- Output just the number, or the word NONE. No other text.

Examples:
Buyer message: "I'll pay $200, not your $500."
Answer: 200
Buyer message: "This 2007 Honda is nice, I can do 2500."
Answer: 2500
Buyer message: "That's too expensive, no thanks."
Answer: NONE

Buyer message: "{buyer_utterance}"
Answer:"""}
    ]

    if model_type == CHATGPT:
        resp = chat_completion_with_backoff(
            model=MODEL, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        out = resp.choices[0]['message']['content']
    elif model_type == FPT:
        out = call_fpt_model(messages, temperature, max_tokens, n_return_sequences=1)[0]
    elif model_type == QWEN:
        out = call_qwen_model(messages, temperature, max_tokens, n_return_sequences=1)[0]
    else:
        out = call_llama3_model(messages, temperature, max_tokens, n_return_sequences=1)[0]

    if out is None:
        return None
    out = out.strip()
    if 'none' in out.lower():
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+", out.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def get_llm_based_assessment_for_emotional_support(state,
                                                   simulated_conversation,
                                                   n=10,
                                                   temperature=1.1,
                                                   max_tokens=20,
                                                   model_type='chatgpt'):
    """
    function that assesses if the supporter successfully confront the seeker in a emotional support conversation
    :param simulated_conversation: the simulated conversation between the seeker and the supporter
    :param n: the number of prompting the LLMs
    :param temperature: the temperature used for prompting the LLMs
    :param max_tokens: the maximal number of tokens generated by the LLMs
    :return:
    """
    # the reward computation function for emotional support conversation
    # the following code is borrowed from the PPDPP official implementation
    dial = ''
    for utt in simulated_conversation:
        if utt['role'] == 'user':
            role = 'Patient'
        else:
            role = 'Supporter'
        dial += f"{role}: {utt['content']}"
        dial += ". "

    # construct the message to prompt the llm
    messages = [{"role": "system",
                 "content": "Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation."},
                {"role": "user",
                 "content": "You can only reply with one of the following sentences: No, the Patient feels worse. No, the Patient feels the same. No, but the Patient feels better. Yes, the Patient's issue has been solved.\n\n"
                            "The following is a conversation about %s regarding %s: %s\nQuestion: Has the Patient's issue been solved? Answer: " % (
                                state['task_background']['emotion_type'], state['task_background']['problem_type'],
                                dial)}]

    responses = []
    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    # calling the llama 3
    elif model_type == LLAMA3:
        responses.extend(call_llama3_model(messages, temperature, max_tokens, n_return_sequences=n))
    elif model_type == FPT:
        responses.extend(call_fpt_model(messages, temperature, max_tokens, n_return_sequences=n))
    elif model_type == QWEN:
        responses.extend(call_qwen_model(messages, temperature, max_tokens, n_return_sequences=n))

    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    return responses


def get_toxicity_assessment_for_emotional_support(generated_system_utt):
    """
    method that compute the toxicity score for emotional support conversation
    :param generated_system_utt: the generated system utterance
    :return:
    """
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': generated_system_utt},
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
    return toxicity_score


def get_user_sentiment_for_item_recommendation(generated_user_utterance):
    """
    method that compute the user sentiment for target-driven recommendation
    :param generated_user_utterance: the generated utterance of the user
    :return:
    """
    sentiment = _get_sentiment_pipeline()(generated_user_utterance)
    return sentiment
