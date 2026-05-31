from base.text_gen import LLMGeneration
from config.config import GenerationConfig
import re
from utils.generation import construct_prompt_for_chat_gpt_response_generation_negotiation, \
    construct_prompt_for_chat_gpt_response_generation_emotional_support, \
    construct_prompt_for_chat_gpt_response_generation_recommendation
from utils.prompt import call_llm

from config.constants import EMOTIONAL_SUPPORT, RECOMMENDATION, NEGOTIATION


class ChatGPTConfigForGeneration(GenerationConfig):
    # the prompt used for
    prompt = "This is the prompt and subjected to be changed"


class ChatGPTGeneration(LLMGeneration):

    def __init__(self, generation_config, pipeline=None, is_test=False):
        """
        constructor for class Chatgpt generation
        :param generation_config: the configuration of the generation method
        :param pipeline: pipeline used to prepare the generation method, for chatgpt, we do not need any pipeline
        :param is_test: True if we are using the generation method at inference time
        """
        super().__init__()
        self.generation_config = generation_config
        self.pipeline = pipeline
        self.is_test = is_test

    def generate_response(self, instance):
        """
        method that generates the response using chatgpt.
        :param instance: the current state of the conversation
        :return:
        """
        if getattr(self.generation_config, "model_type", None) == "rule":
            return self._rule_response(instance)

        dialogue_context = instance['dialogue_context']

        # the recommendation scenario
        if self.generation_config.scenario_name == RECOMMENDATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_recommendation(instance,
                                                                                                          self.generation_config.prompt,
                                                                                                          dataset=self.generation_config.dataset)
        # the negotiation scenario
        elif self.generation_config.scenario_name == NEGOTIATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_negotiation(instance,
                                                                                                       self.generation_config.prompt)
        # the emotional support conversation
        elif self.generation_config.scenario_name == EMOTIONAL_SUPPORT:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_emotional_support(instance,
                                                                                                             self.generation_config.prompt)
        else:
            raise Exception("Invalid Scenario ...")

        messages.extend(dialogue_context)

        # calling the llm for response generation
        # Incorporating strategy description at the later of the prompt improve the alignment
        # between the predicted dialogue strategy and the generated response.
        messages.append(
            {'role': 'user', 'content': f"{goal_description}. "
                                        'Please reply with only one short and succinct sentence.'}
        )

        response = call_llm(messages,
                            n=1,
                            temperature=self.generation_config.temperature,
                            max_token=self.generation_config.max_gen_length,
                            model_type="chatgpt"
                            )
        # returning the response
        return response[0]

    def _rule_response(self, instance):
        """Deterministic buyer utterance used only for smoke/debug training."""
        strategy = instance.get('pred_goal', 'counter')
        topic = instance.get('pred_topic', 2)
        try:
            bin_pred = int(topic)
        except (TypeError, ValueError):
            bin_pred = 2
        buyer_price = float(instance['task_background']['buyer_price'])
        seller_price = float(instance['task_background']['seller_price'])
        span = max(seller_price - buyer_price, 1.0)
        bin_width = span / 5.0
        proposed_price = int(buyer_price + bin_width * max(0, min(bin_pred, 4)))

        if strategy == "agree":
            seller_offer = self._last_user_price(instance)
            if seller_offer is not None:
                return f"I accept your offer of ${seller_offer:.0f}."
            return f"I can offer ${proposed_price}."
        if strategy == "propose":
            return f"I can offer ${proposed_price}."
        if strategy == "counter":
            return f"Could you do ${proposed_price}?"
        if strategy == "final_offer":
            return f"${proposed_price} is my final offer."
        if strategy == "walk_away":
            return "I will pass for now and look at other options."
        if strategy in {"disagree", "deny"}:
            return "That price is too high for me."
        if strategy == "inquire":
            return "Can you tell me more about the condition?"
        if strategy == "greet":
            return "Hi, I am interested in this item."
        return f"I am interested, but my budget is around ${proposed_price}."

    def _last_user_price(self, instance):
        for turn in reversed(instance.get('dialogue_context', [])):
            if turn.get('role') != 'user':
                continue
            prices = re.findall(r"[-+]?\d*\.?\d+", (turn.get('content') or '').replace(",", ""))
            if prices:
                return float(prices[-1])
        return None
