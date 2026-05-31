import time

from loguru import logger
from base.simulator import Simulator
from utils.prompt import call_llm
from utils.generation import PRICE_TAG_INSTRUCTION_SELLER


class NegotiationSimulator(Simulator):

    def __init__(self, user_profile, use_persona=False):
        """
        constructor for class negotiation simulator
        :param user_profile: a tuple of big5 persona and decision making style
        """
        self.use_persona = use_persona
        # generating the profile description
        self.user_profile_description = self.generate_persona_description(user_profile)

    def respond(self, state):
        """
        method that generates the user response for the negotiation scenario
        :param state: the current state of the conversation
        :return: the generated response by the user.
        """
        dialogue_context = state['dialogue_context']
        item_name = state['task_background']['item_name']
        initial_price = state['task_background']['seller_price']
        product_description = state['task_background']['seller_item_description']

        # if we employ the persona for the user simulator
        if self.use_persona:
            prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Seller in a price
            bargaining game. The task background is as bellow.
            Your persona: {self.user_profile_description}
            You must follow the instructions below during chat.
            1. Your utterances and bargain behavior need to strictly follow your persona. Varying your wording
            and avoid repeating yourself verbatim!
            2. You can decide to change your target price flexibly based on your persona and the conversation.
            Your Strategy:
            1. "Source Derogation": Attacks the other party or questions the item.
            2. "Counter Argument": Provides a non-personal argument/factual response to refute a previous claim
            or to justify a new claim.
            3. "Personal Choice": Provides a personal reason for disagreeing with the current situation or chooses
            to agree with the situation provided some specific condition is met.
            4. "Information Inquiry": Requests for clarification or asks additional information about the item or
            situation.
            5. "Self Pity": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
            6. "Hesitance": Stalls for time and is hesitant to commit; specifically, they seek to further the
            conversation and provide a chance for the other party to make a better offer
            7. "Self-assertion": Asserts a new claim or refutes a previous claim with an air of finality/ confidence.
            8. "Others": Do not explicitly foil the negotiation attempts.
            You are the seller who is trying to sell the {item_name} with the initial price of {initial_price}. 
            Product description: {product_description}. 
            The conversation history is as bellow:
            """
        # we ignore the persona for the user simulator
        else:
            prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Seller in a price
            bargaining game. The task background is as bellow.
            You must follow the instructions below during chat.
            You can decide to change your target price flexibly based on the conversation.
            Your Strategy:
            1. "Source Derogation": Attacks the other party or questions the item.
            2. "Counter Argument": Provides a non-personal argument/factual response to refute a previous claim
            or to justify a new claim.
            3. "Personal Choice": Provides a personal reason for disagreeing with the current situation or chooses
            to agree with the situation provided some specific condition is met.
            4. "Information Inquiry": Requests for clarification or asks additional information about the item or
            situation.
            5. "Self Pity": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
            6. "Hesitance": Stalls for time and is hesitant to commit; specifically, they seek to further the
            conversation and provide a chance for the other party to make a better offer
            7. "Self-assertion": Asserts a new claim or refutes a previous claim with an air of finality/ confidence.
            8. "Others": Do not explicitly foil the negotiation attempts.
            You are the Seller who is trying to sell the {item_name} with the price of {initial_price}. 
            Product description: {product_description}.  
            The conversation history is as bellow:
            """
        # construct the system instruction prompt
        messages = [
            {"role": "system", "content": prompt}
        ]

        # reformating and prepending the dialogue context to the current prompt
        messages.extend(self.reformat_dialogue_context(dialogue_context))
        
        # COT thinking to respond
        tag_instruction = PRICE_TAG_INSTRUCTION_SELLER if state.get('use_price_tag') else ""
        messages.append(
            {'role': 'user', 'content': f"""Based on the given task background and conversation history,
             first consider the price proposed by the Buyer and your desired price of {initial_price}.
             Then determining if the proposed price is acceptable.
             If yes, please accept the offer by the Buyer.
             if no, please select an appropriate strategy and respond the Buyer accordingly.
             You have to reply with only one short and succinct sentence.{tag_instruction}
            """
             }
        )

        # messages.extend(dialogue_context)
        t = time.time()

        # calling the llm for response generation
        response = call_llm(messages, n=1, temperature=0.00000001, max_token=self.max_gen_token, model_type=self.model_type)
        logger.debug(f"Simulator Generation Time: {time.time() - t:.2f}s")
        return response[0]

    def generate_persona_description(self, user_profile):
        """
        method that generate a persona description given the big5 personality and decision making style (persona, decision_type)
        :return: an user profile description
        """
        assert len(user_profile) == 2
        prompt = f"""
        You need to incorporate the following user profile and generate a cohesive persona description.
        You need to ensure the description is easy to understand.
        ********
        Big-Five Personality: {user_profile[0]}
        Decision-Making Style: {user_profile[1]}
        ********
        """
        messages = [
            {"role": "system", "content": prompt}
        ]
        output = call_llm(messages, n=1, temperature=self.temperature, max_token=self.max_description_tokens,
                          model_type=self.model_type)
        # return the user persona description
        return output[0]
    
