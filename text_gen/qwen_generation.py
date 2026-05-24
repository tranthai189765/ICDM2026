import time

from loguru import logger
from base.text_gen import LLMGeneration
from config.config import GenerationConfig
from utils.generation import (
    construct_prompt_for_chat_gpt_response_generation_negotiation,
    construct_prompt_for_chat_gpt_response_generation_emotional_support,
    construct_prompt_for_chat_gpt_response_generation_recommendation,
)
from utils.prompt import call_llm
from config.constants import EMOTIONAL_SUPPORT, RECOMMENDATION, NEGOTIATION, QWEN


class QwenConfigForGeneration(GenerationConfig):
    prompt = "This is the prompt and subjected to be changed"
    temperature = 0.0


class QwenGeneration(LLMGeneration):
    """Response generation backed by a local Qwen2.5 Instruct model
    (default Qwen/Qwen2.5-14B-Instruct, see config.constants.QWEN_MODEL)."""

    def __init__(self, generation_config, pipeline=None, is_test=False):
        super().__init__()
        self.generation_config = generation_config
        self.pipeline = pipeline
        self.is_test = is_test

    def generate_response(self, instance):
        dialogue_context = instance['dialogue_context']
        if self.generation_config.scenario_name == RECOMMENDATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_recommendation(
                instance, self.generation_config.prompt)
        elif self.generation_config.scenario_name == NEGOTIATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_negotiation(
                instance, self.generation_config.prompt)
        elif self.generation_config.scenario_name == EMOTIONAL_SUPPORT:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_emotional_support(
                instance, self.generation_config.prompt)
        else:
            raise Exception("Invalid Scenario ...")

        messages.extend(dialogue_context)
        messages.append(
            {'role': 'user', 'content': f"{goal_description}. "
                                        'Please reply with only one short and succinct sentence.'}
        )

        t = time.time()
        response = call_llm(messages, n=1,
                            temperature=0.001,
                            max_token=self.generation_config.max_gen_length,
                            model_type=QWEN)
        logger.debug(f"Qwen Response Generation Time: {time.time() - t:.2f}s")
        return response[0]
