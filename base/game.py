import os.path
import re
import math
import time
from abc import ABC, abstractmethod
from loguru import logger

from config.constants import DURECDIAL, INSPIRED, CRAIGSLIST_BARGAIN, ES_CONV
from utils.prompt import get_llm_based_assessment_for_recommendation, get_llm_based_assessment_for_negotiation, \
    get_llm_based_assessment_for_emotional_support, get_toxicity_assessment_for_emotional_support, \
    get_user_sentiment_for_item_recommendation, get_llm_based_price_extraction_for_negotiation
from config.constants import SUCCESS_RATE, DEAL_RATE, ITEM_FREQ, AVG_TURN, SL_RATIO, FAIRNESS, TOXICITY, USER_REWARD


# Only these buyer strategies actually COMMIT to a price. Numbers that appear
# in any other utterance (deny, disagree, inquire, ...) are referential — e.g.
# the agent restating its budget while denying — and must NOT be credited as a
# secured price, otherwise a 'deny' that echoes the buyer target yields a bogus
# sl_ratio of 1.0 and the policy learns to spam 'deny'.
NEG_PRICE_COMMITTING = {'propose', 'counter', 'final_offer', 'agree'}


class Game(ABC):

    def __init__(self, game_config, dataset_config):
        """
        constructor for class abstract class Scenario
        :param game_config: configuration of the scenario
        """
        self.game_config = game_config
        self.dataset_config = dataset_config
        self.model_type = self.game_config.model_type

        # create the log dir
        os.makedirs(self.game_config.log_dir, exist_ok=True)

        # create the saved dir
        os.makedirs(self.game_config.saved_dir, exist_ok=True)

    @abstractmethod
    def reset(self, case, simulator):
        """
        method that reset the state of the environment
        :param case: a specific circumstance, task background, target item, e.g
        :param simulator: a instance of the simulator class 
        :return: None
        """
        raise NotImplementedError("This method needs to be implemented")

    @abstractmethod
    def is_terminated(self, action, state):
        """
        method that check if the current state if the terminated state
        :return:
        """
        raise NotImplementedError("This method needs to be implemented")

    def step(self, state, action, generation_model, simulator):
        """
        method that update the state of the game
        :param action: the current action by the system
        :param generation_model: the system response generation model
        :param simulator: the user simulator
        :return:
        """
        raise NotImplementedError("This method needs to be implemented")

    def compute_reward(self, state, action, system_response, profile_description):
        """
        method that update the current state of the game
        :param state: the current state of the game
        :param action: the predicted action by the system
        :param system_response: the generated system response
        :param profile_description: the user profile description
        :return: None
        """
        raise NotImplementedError("This method needs to be implemented")


class RecommendationGame(Game):

    def __init__(self, game_config, dataset_config):
        """
        constructor for class recommendation game
        :param dataset_config: the dataset config
        """
        super().__init__(game_config, dataset_config)

    def reset(self, case, simulator):
        """
        method that reset the state of the scenario
        :param case: a particular case, for recommendation, it is a target item
        :param simulator: a simulator used to generate the user's response
        :return: 
        """
        if 'conv' in case['demonstration']:
            del case['demonstration']['conv']

        if self.dataset_config.dataset_name == DURECDIAL:
            goal = "Greetings"
            topic = "Greetings"
        elif self.dataset_config.dataset_name == INSPIRED:
            goal = "no_strategy"
            topic = "no_strategy"
        else:
            raise Exception("Invalid dataset")

        state = {
            "task_background": {
                "target_topic": case['topic'],
                "target_goal": case['goal'],
                "topic_set": case['topic_set']                
            },
            "demonstration": case["demonstration"],
            "dialogue_context": [{"role": "assistant", "content": "Hi, How do I help you ?"}],
            "goal": goal,  # will not affect anything, only including it for coding convenience
            "topic": topic,
            "knowledge": [case['topic'], "", ""],  # will not affect anything, only including it for coding convenience
            "response": "",  # will not affect anything, only including it for coding convenience
            "pre_goals": [''],
            "pre_topics": [''],
            "goal_path": [""],
            "topic_path": [""],

        }
        user_initial_response = simulator.respond(state=state, dataset=self.dataset_config.dataset_name)
        state['dialogue_context'].append({'role': 'user', 'content': user_initial_response})
        return state

    def is_terminated(self, action, state):
        """
        method that check if the game is terminated
        :param action: the current action from the system
        :param state: the current state of the game
        :return: True if the game is terminated else False
        """
        # say goodbye goal
        if action == self.game_config.terminated_action:
            return 1
        # if the length of the conversation exceed a predefined threshold
        if len(state['dialogue_context']) >= self.game_config.max_horizon:
            return 1
        return 0

    def step(self, state, action, generation_model, simulator):
        """
        method that update the current state of the game and return the reward
        :param state: the current state of the game
        :param action: the predicted action by the system
        :param generation_model: the response generation method
        :param simulator: the user simulator
        :return: the new state, reward, and flag indicating if the game is terminated
        """
        if isinstance(action, tuple):
            goal, topic = action
        else:
            goal = action
            topic = ''

        logger.info(f"[Goal]: {goal}")
        logger.info(f"[Topic]: {topic}")

        # prepare state for the response generation
        state['pred_goal'] = goal
        state['pred_topic'] = topic

        # generate the system response
        system_response = generation_model.generate_response(state)

        # update the dialogue context
        state['dialogue_context'].append({"role": "assistant", "content": system_response})

        # generate user response with LLM
        user_response = simulator.respond(state, dataset=self.dataset_config.dataset_name)

        # construct the new state
        # prepend the system and user reponse to the dialogue context
        # prepend the predicted goal, topic to the previous goals, topics
        state['dialogue_context'].append({'role': 'user', 'content': user_response})
        state['pre_goals'].append(goal)
        state['pre_topics'].append(topic)

        logger.info(f"[System]: {system_response}")
        logger.info(f"[USER]: {user_response}")

        # compute the reward
        reward, done, o_done = self.compute_reward(state, action, system_response, simulator.user_profile_description)

        logger.debug(f"reward={reward} done={done} o_done={o_done}")
        # return the new state, intermediate reward, and termination flag.
        return state, reward, done, o_done

    def compute_reward(self, state, action, system_response, profile_description):
        """
        method that compute the intermediate reward r(s_{t},s_{t+1},a)
        :param state: the new state of the game s_{t+1}
        :param action: the predicted acton a
        :param system_response: the generated system response
        :param profile_description: the user profile description
        :return: the reward and the termination flag
        """
        if isinstance(action, tuple):
            goal, topic = action
        else:
            goal = action
            topic = ''
                    
        # the targeted item
        target_item = state['task_background']['target_topic']

        # objective reward
        # check if the target item appear in the generated system response.
        if self.dataset_config.dataset_name == INSPIRED:
            target_item = re.sub(r'\(\d+\)', '', target_item)

        # construct the reward
        # the reward should contain multiple values corresponding to different objectives
        # the reward is in turn level
        # objective and subjective assessment.
        sub_reward = 0.0
        obj_reward = 0.0

        # add some negative reward if the conversation keeps going.
        avg_turn_reward = -0.1

        # objective_done
        o_done = 0

        ## Recommendaion reward
        if 'recommendation' in goal:
            # a small reward for recommendation action
            # encouraging the model to make more recommendations
            obj_reward = 0.2
            # # compute the objective assessment
            # # check if the target item is in the generated system response
            if target_item.replace(" ", "").lower().strip() in system_response.replace(" ", "").lower().strip():
                # give a very right reward if the target item is recommended successfully
                obj_reward = 1.0
                o_done = 1

        # compute the objective assessment
        # check if the predicted topic is the target item
        # then we give a reward to the model (i.e the item freq)
        # if target_item.lower().strip() in topic.lower().strip():
        #     obj_reward = 1.0

        # get the user generated response
        user_response = state['dialogue_context'][-1]['content']
        # compute the user sentiment using a pretrained sentiment analysis model
        user_sentiment_results = get_user_sentiment_for_item_recommendation(user_response)[0]
        sentiment_label, sentiment_score = user_sentiment_results['label'], user_sentiment_results['score']

        # positive sentiment
        # a high reward
        user_reward = 0
        if sentiment_label == "LABEL_2":
            user_reward = sentiment_score
        # neutral
        # zero reward
        elif sentiment_label == "LABEL_1":
            user_reward = 0
        # negative sentiment
        # negative reward
        elif sentiment_label == "LABEL_0":
            user_reward = - sentiment_score

        # check if the game is terminated
        is_terminated = self.is_terminated(goal, state)

        # if the conversation is terminated
        # we evaluate the user's sentiment on the recommended item
        if is_terminated:
            logger.info('--> Terminated conversation !')

            # failed case in default
            done = -1

            # compute the llm-based target-driven assessment
            sub_reward = get_llm_based_assessment_for_recommendation(target_topic=target_item,
                                                                     simulated_conversation=state[
                                                                         'dialogue_context'],
                                                                     demonstration=state['demonstration'],
                                                                     n=self.game_config.n,
                                                                     profile_description=profile_description,
                                                                     model_type=self.model_type
                                                                     )

            # check if the target item appear in the conversation
            # o_done = 1 if the target item appear in the conversation
            for utt in state['dialogue_context']:
                if target_item.lower().strip() in utt['content'].lower().strip():
                    o_done = 1

            # successful case
            if sub_reward >= self.game_config.epsilon:
                done = 1

        else:
            # if the length of the trajectory is greater than the maximal game horizon
            if len(state[
                       'dialogue_context']) == self.game_config.max_horizon or goal == self.game_config.terminated_action:
                logger.info('Maximum number of turns reached !')
                # failed case
                done = -1
            else:
                logger.info('The conversation is on-going !')
                done = 0
                pass


        # vector-valued reward function
        reward = []

        # if we use subjective reward
        if USER_REWARD in self.game_config.objectives:
            reward.append(user_reward)
        # if we use objective reward
        if ITEM_FREQ in self.game_config.objectives:
            reward.append(obj_reward)
        # if we use avg turn.
        if AVG_TURN in self.game_config.objectives:
            reward.append(avg_turn_reward)

        logger.debug(f"reward={reward}")
        return reward, done, o_done


class NegotiationGame(Game):

    def __init__(self, game_config, dataset_config):
        """
        constructor for class negotiation game
        :param game_config: the configuration of the negotiation scenario
        :param dataset_config: the configuration of the dataset.
        """
        super().__init__(game_config, dataset_config)

    def is_terminated(self, action, state):
        """
        method that check if the game is terminated
        :param action: the current action from the system
        :param state: the current state of the game
        :return: True if the game is terminated else False
        """
        if action == self.game_config.terminated_action:
            return True
        if len(state['dialogue_context']) >= self.game_config.max_horizon:
            return True
        return False

    def reset(self, case, simulator):
        """
        method that reset the state of the scenario
        :param case: a particular case, for negotiation, it is a item name
        :param simulator: a simulator used to generate the user's response
        :return:
        """
        if self.dataset_config.dataset_name == CRAIGSLIST_BARGAIN:
            goal = "greet"
        else:
            raise Exception("Invalid dataset")

        # borrowing from PPDPP official implementation
        # in the negotiation dialogue, the system is the buyer
        # the user is the seller
        dialogue_context = [
            {"role": "assistant", "content": "Hi, how much is the %s?" % case['task_background']['item_name']},
            {"role": "user", "content": "Hi, this is a good %s and its price is %s." % (
                case['task_background']['item_name'], case['task_background']['seller_price'])}]

        # construct the initial state
        state = {
            "task_background": {
                "item_name": case['task_background']['item_name'],
                "buyer_price": case['task_background']['buyer_price'],
                "buyer_item_description": case['task_background']['buyer_item_description'],
                "seller_price": case['task_background']['seller_price'],
                "seller_item_description": case['task_background']['seller_item_description']
            },
            "dialogue_context": dialogue_context,
            "goal": goal,  # will not affect anything, only including it for coding convenience
            "response": "",  # will not affect anything, only including it for coding convenience
            "pre_goals": [''],
            "pre_topics": [''],
            # Price-tag protocol flag (CLI: --use_price_tag). Read by the buyer
            # generation prompt and the seller simulator to append a price tag.
            "use_price_tag": getattr(self.game_config, 'use_price_tag', False),
            # Latest declared prices parsed from the price tags (filled in step()).
            "_buyer_declared_price": None,
            "_seller_declared_price": None,
        }
        return state

    def step(self, state, action, generation_model, simulator):
        """
        method that update the current state of the game and return the reward
        :param state: the current state of the game
        :param action: the predicted action by the system
        :param generation_model: the response generation method
        :param simulator: the user simulator
        :return: the new state, reward, and flag indicating if the game is terminated
        """
        goal = action
        logger.info(f"[Goal]: {goal}")

        # prepare state for the response generation
        state['pred_goal'] = goal

        # generate the system response
        system_response = generation_model.generate_response(state)

        # Price-tag protocol: parse the buyer's declared price from the tag and
        # strip the tag before the utterance is stored / shown to the seller.
        # (No-op outside negotiation, where state has no 'use_price_tag' key.)
        # Only credit the price when the action actually COMMITS to one — a
        # 'deny'/'inquire' that echoes the buyer's budget is referential, not a
        # secured price, so it must not overwrite the anchor (anti reward-hack).
        if state.get('use_price_tag'):
            from utils.generation import extract_price_tag, strip_price_tag
            _strategy_now = action[0] if isinstance(action, tuple) else action
            _buyer_price = extract_price_tag(system_response)
            if _buyer_price is not None and _strategy_now in NEG_PRICE_COMMITTING:
                state['_buyer_declared_price'] = _buyer_price
            system_response = strip_price_tag(system_response)

        # update the dialogue context
        state['dialogue_context'].append({"role": "assistant", "content": system_response})

        # generate user response with LLM
        user_response = simulator.respond(state)

        # Price-tag protocol: parse the seller's declared price and strip tag.
        if state.get('use_price_tag'):
            from utils.generation import extract_price_tag, strip_price_tag
            _seller_price = extract_price_tag(user_response)
            if _seller_price is not None:
                state['_seller_declared_price'] = _seller_price
            user_response = strip_price_tag(user_response)

        # construct the new state
        # prepend the system and user reponse to the dialogue context
        # prepend the predicted goal, topic to the previous goals, topics
        state['dialogue_context'].append({'role': 'user', 'content': user_response})
        state['pre_goals'].append(goal)

        logger.info(f"[System]: {system_response}")
        logger.info(f"[USER]: {user_response}")

        # compute the reward
        reward, done, o_done = self.compute_reward(state, action, system_response, simulator.user_profile_description)

        # return the new state, intermediate reward, and termination flag.
        return state, reward, done, o_done

    def compute_reward(self, state, action, system_response, profile_description, eps=1e-1):
        """
        method that compute the reward for each step in a negotiation scenario
        :param state: the current state of the conversation
        :param action: the predicted goal at the current turn
        :param system_response: the generated system response
        :param profile_description: the user profile description
        :param eps:
        :return:
        """
        if isinstance(action, tuple):
            action = action[0]

        goal = action
        # compute the llm-basd assessment
        t = time.time()
        # BUGFIX (#2): temperature 1.1 made the NLI deal-judge stochastic, so
        # neg_sr (the deal_rate reward component AND the terminal trigger) was
        # noisy across identical dialogues. Use temperature=0.0 so the judge is
        # deterministic — stable reward signal and reproducible terminations.
        responses = get_llm_based_assessment_for_negotiation(simulated_conversation=state['dialogue_context'],
                                                             n=self.game_config.n,
                                                             temperature=0.0,
                                                             model_type=self.model_type,
                                                             max_tokens=10
                                                             )

        logger.debug(f"NLI assessment time={time.time() - t:.2f}s")
        logger.debug(f"NLI responses={responses}")

        # deal used to compute the neg_sr
        # indicate whether the system and the user reach a deal
        deals = []
        rewards = []

        # compute fairness score
        # fairness score should be defined as
        for response in responses:
            # compute the neg_sr
            if 'have not' in response.lower():
                # no deal
                deals.append(0)
            elif 'have reached' in response.lower():
                # there is a deal
                deals.append(1)

            # collect the dealed price
            # and now we compute the fairness score
            prices = re.findall(r"[-+]?\d*\.?\d+", response.replace(",", ""))
            if len(prices) > 0:
                deal_price = float(prices[0])
                # compute the sale list ratio
                reward = (deal_price - state['task_background']['seller_price']) / (
                        state['task_background']['buyer_price'] - state['task_background']['seller_price'])
                rewards.append(reward)

        # deal rate
        # Guard against an all-ambiguous judge batch (no response contained
        # 'have not'/'have reached'); previously this raised ZeroDivisionError.
        neg_sr = sum(deals) / len(deals) if len(deals) > 0 else 0.0
        # # computing the negotiation success rate and fairness score
        # if neg_sr < self.game_config.epsilon:
        #     sl_ratio = 0.0
        # else:
        #     if len(rewards) == 0:
        #         sl_ratio = 0.0
        #     else:
        #         sl_ratio = max(set(rewards), key=rewards.count)
        #

        # computing the price gain
        # extracting the price proposed by the system
        system_prices = re.findall(r"[-+]?\d*\.?\d+", system_response.replace(",", ""))

        # Resolve action strategy for non-tuple actions.
        _strategy = action[0] if isinstance(action, tuple) else action

        _seller_p = float(state['task_background']['seller_price'])
        _buyer_p = float(state['task_background']['buyer_price'])

        # BUGFIX (#1): the agent is the BUYER, who wants a LOW price, yet the
        # old code did `max(system_prices)`. When the utterance mentioned both
        # the agent's offer and the seller's ask (e.g. "I'll pay $200, not your
        # $500"), max() picked the SELLER's $500 and zeroed out the agent's
        # gain — a strong wrong-direction reward signal. Worse, stray numbers
        # (model years like 2007, quantities, etc.) leaked in and produced
        # nonsense prices. We now (a) filter to PLAUSIBLE prices inside a band
        # around the buyer/seller range, then (b) pick the LOWEST plausible
        # one — the buyer's actual offer.
        def _pick_buyer_price(price_strs):
            lo = min(_seller_p, _buyer_p) * 0.5
            hi = max(_seller_p, _buyer_p) * 1.5
            plausible = []
            for p in price_strs:
                try:
                    val = float(p)
                except ValueError:
                    continue
                if lo <= val <= hi:
                    plausible.append(val)
            if not plausible:
                return None
            return min(plausible)

        # Only PRICE-COMMITTING actions credit the current utterance's price
        # as the system anchor. For all other actions, the number that
        # appears in the utterance is REFERENTIAL (e.g. agent says
        # "I won't pay $1300" — $1300 is the seller's offer being rejected,
        # NOT a new commitment). Crediting referential numbers destroys the
        # r_gain measurement by ~0.2 on average. For non-committing actions
        # we always fall back to the agent's MOST RECENT real anchor.
        _PRICE_COMMITTING = NEG_PRICE_COMMITTING

        def _scan_prior_anchor():
            for prev_turn in reversed(state['dialogue_context'][:-2]):
                if prev_turn.get('role') == 'assistant':
                    prev_prices = re.findall(
                        r"[-+]?\d*\.?\d+",
                        (prev_turn.get('content') or '').replace(",", "")
                    )
                    picked = _pick_buyer_price(prev_prices)
                    if picked is not None:
                        return picked
            return None

        # Optional LLM-based price extraction (CLI: --use_llm_price_extraction).
        # The agent is the Buyer; when an utterance has several numbers the
        # regex heuristic can pick the wrong one. If enabled, ask the LLM to
        # extract the buyer's committed price for PRICE-COMMITTING actions and
        # use it when it returns a plausible value. Falls back to the heuristic
        # below on None/implausible output so a flaky LLM never breaks training.
        llm_price = None
        if getattr(self.game_config, 'use_llm_price_extraction', False) \
                and _strategy in _PRICE_COMMITTING:
            try:
                llm_price = get_llm_based_price_extraction_for_negotiation(
                    buyer_utterance=system_response,
                    seller_price=_seller_p,
                    buyer_price=_buyer_p,
                    temperature=0.0,
                    model_type=self.model_type,
                )
            except Exception as e:
                logger.warning(f"[LLM price extraction] failed: {e}")
                llm_price = None
            if llm_price is not None:
                lo = min(_seller_p, _buyer_p) * 0.5
                hi = max(_seller_p, _buyer_p) * 1.5
                if not (lo <= llm_price <= hi):
                    llm_price = None   # reject implausible LLM output

        # Price-tag protocol (CLI: --use_price_tag): the buyer declared its
        # current price via a machine tag parsed in step(). _buyer_declared_price
        # holds the MOST RECENT declared buyer price (it persists across turns
        # where the buyer named no price, acting as the prior anchor). This is
        # the most reliable source, so it takes priority.
        tag_price = None
        if state.get('use_price_tag'):
            tag_price = state.get('_buyer_declared_price')
            if tag_price is not None:
                lo = min(_seller_p, _buyer_p) * 0.5
                hi = max(_seller_p, _buyer_p) * 1.5
                if not (lo <= tag_price <= hi):
                    tag_price = None

        committed_price = _pick_buyer_price(system_prices)
        if state.get('use_price_tag'):
            # Tag mode is authoritative and leak-free: _buyer_declared_price
            # only ever holds prices from PRICE-COMMITTING turns (gated in
            # step()). If the buyer has never committed a price yet, no gain is
            # secured → fall straight back to the seller's listing (sl_ratio=0).
            # We deliberately do NOT scan prior utterance text here, because a
            # 'deny' that echoes the buyer's budget would otherwise leak in.
            system_price = tag_price if tag_price is not None else _seller_p
        elif llm_price is not None:
            # Trust the LLM-extracted buyer commitment.
            system_price = llm_price
        elif _strategy in _PRICE_COMMITTING and committed_price is not None:
            # Real commitment: trust the current utterance's (plausible) offer.
            system_price = committed_price
        else:
            # Non-committing action (deny, disagree, inquire, walk_away,
            # inform, affirm, greet, confirm, counter-noprice) OR price-
            # committing action with no extractable plausible price. Use the
            # most recent real anchor.
            system_price = _scan_prior_anchor()
            if system_price is None:
                system_price = _seller_p

        system_price = float(system_price)

        # Guard against degenerate dataset rows where buyer_price >= seller_price
        # (a few CraigslistBargain cases have inverted target/listing — typically
        # an annotation artifact). In those cases the standard sl_ratio formula
        # produces nonsensical values and a few outlier episodes drag down the
        # average r_gain by 0.05-0.10. Treat them as a neutral outcome.
        # (_seller_p / _buyer_p are defined above, near the price extraction.)
        # ALSO clamp the case where the LLM hallucinated a price ABOVE the
        # seller's listing (e.g. counter,2 prompted to say bin 2 price = $220
        # but LLM echoed seller's $580 instead). Without the clamp this gives
        # sl_ratio < -1 (clipped at -1) which destroys both r_gain and r_fair
        # for the offending episode -- exactly the ep19 disaster we kept hitting.
        # Treat overpay as "agent capitulated to seller's full listed price":
        # sl_ratio = 0, fairness = 0.
        if _buyer_p < _seller_p and system_price > _seller_p:
            system_price = _seller_p
        if _buyer_p >= _seller_p or abs(_buyer_p - _seller_p) < 1e-6:
            sl_ratio = 0.0
            fairness = 0.0
            mid_price = (_seller_p + _buyer_p) / 2
        else:
            # encourage the system to gain more benefit
            # this reward is to encourage the model to propose beneficial price for its self.
            sl_ratio = (system_price - _seller_p) / (_buyer_p - _seller_p)

            # clipping the values
            # if the ratio is larger than 1 then it is equivalent to 1
            # otherwise it equals to 0
            if sl_ratio >= 1.0:
                sl_ratio = 1.0
            elif sl_ratio < -1.0:
                sl_ratio = -1.0

            # fairness
            # we compute the fairness score, which will be larger if the proposed price is close to the middle price
            # this should be conflicting to the sl_ratio price.
            mid_price = (_seller_p + _buyer_p) / 2

            # if the system price is equivalent to the mid price
            # we give the system a high fairness reward
            fairness = 0.5 - abs(system_price - mid_price) / (_seller_p - _buyer_p)

            # clipping the values
            # if the fairness is larger than 0.5 then it is equivalent to 0.5
            # otherwise it equals to 0
            if fairness >= 0.5:
                fairness = 0.5
            elif fairness < -0.5:
                fairness = -0.5

        # add some negative reward if the conversation keeps going.
        turn_reward = -0.1

        # a flag to indicate whether the conversation is terminated
        done = 0

        # if the neg_sr is greater than a predefined threshold
        # successful case
        # both parties fail to reach a deal
        # then fairness score is 0
        # if neg_sr < self.game_config.epsilon:
        #     fairness_score = 0.0
        # else:
        #     # if the deal price = seller price, then seller utility = 1, agent utility = -1
        #     # if the deal price = buyer price, then buyer utility = 1, seller utility  = - 1
        #     # fairness = - |buyer_utility - buyer utility|
        #     if sl_ratio == 1.0 or sl_ratio == 0.0:
        #         fairness_score = -1.0
        #     # otherwise, fairness score = 0
        #     else:
        #         fairness_score = 1.0

        # BUGFIX (#4): walk_away is an explicit decision to abandon the deal,
        # but previously it left done=0 (NLI said "have not"), forcing the
        # agent to keep talking after it had quit. Make walk_away a hard
        # FAILURE terminal and zero out price/deal rewards (no commitment was
        # made). This removes wasted post-quit turns and their noisy rewards.
        if _strategy == 'walk_away':
            logger.info('--> Agent walked away (hard terminal).')
            sl_ratio = 0.0
            fairness = 0.0
            neg_sr = 0.0
            done = -1
        # checking if there is a deal
        # that mean the neg_sr should be greater than a predefined threshold
        elif neg_sr >= self.game_config.epsilon:
            logger.info('--> Terminated conversation !')
            done = 1
        # other cases
        else:
            # if the length of the trajectory is greater than the maximal game horizon
            # the conversation should be also terminated here
            if len(state['dialogue_context']) == self.game_config.max_horizon:
                logger.info('Maximum number of turns reached !')
                # failed case
                done = -1
            else:
                # logger.info('The conversation is on-going !')
                pass

        # Fairness range-imbalance correction (CLI: --fairness_train_scale).
        # r_gain and r_deal saturate at 1.0 but the fairness formula caps at
        # 0.5, so under a balanced weight the scalarised return under-weights
        # fairness and the policy drifts to gain/deal. Scaling fairness up
        # during TRAINING (e.g. x2 -> max 1.0) restores parity; eval keeps the
        # default scale 1.0 so the reported r_fair stays comparable to the
        # PADPP paper (max 0.5).
        fairness = fairness * getattr(self.game_config, 'fairness_train_scale', 1.0)

        # PADPP-original reward vector (3 dimensions): [sl_ratio, fairness, neg_sr].
        # No shaping, no avg_turn objective -- matches the paper exactly.
        reward = []
        if SL_RATIO in self.game_config.objectives:
            reward.append(sl_ratio)
        if FAIRNESS in self.game_config.objectives:
            reward.append(fairness)
        if SUCCESS_RATE in self.game_config.objectives or DEAL_RATE in self.game_config.objectives:
            reward.append(neg_sr)
        if AVG_TURN in self.game_config.objectives:
            reward.append(turn_reward)

        # Turn penalty (CLI: --turn_penalty). Without an avg_turn objective the
        # 3-D reward has no pressure to close, so under the MEAN convention the
        # agent harvests per-turn gain/fairness by countering forever. Subtract
        # a constant c from EVERY component: for any simplex preference w,
        # w·[r - c] = w·r - c, i.e. a uniform per-turn penalty c that pushes the
        # policy to close early without changing the in-turn action ranking.
        # Applied only during training (eval keeps c=0 so metrics stay clean).
        c = getattr(self.game_config, 'turn_penalty', 0.0)
        if c:
            reward = [r - c for r in reward]

        logger.debug(f"reward={reward} done={done}")
        return reward, done, done


class EmotionalSupportGame(Game):

    def __init__(self, game_config, dataset_config):
        """
        constructor for class emotional support game
        :param game_config: the configuration of the scenario
        :param dataset_config: the configuration of the dataset
        """
        super().__init__(game_config, dataset_config)

    def is_terminated(self, action, state):
        """
        method that check if the game is terminated
        :param action: the current action from the system
        :param state: the current state of the game
        :return: True if the game is terminated else False
        """
        if action == self.game_config.terminated_action:
            return True
        if len(state['dialogue_context']) >= self.game_config.max_horizon:
            return True
        return False

    def reset(self, case, simulator):
        """
        method that reset the state of the scenario
        :param case: a particular case, for negotiation, it is a item name
        :param simulator: a simulator used to generate the user's response
        :return:
        """
        if self.dataset_config.dataset_name == ES_CONV:
            goal = "Question"
        else:
            raise Exception("Invalid dataset")

        # borrowing from PPDPP official implementation
        # in the negotiation dialogue, the system is the buyer
        # the user is the patient
        # the system is the supporter
        dialogue_context = [
            {"role": "assistant", "content": "Hi ! How do I help you ?"},
            {"role": "user", "content": case['task_background']['situation']}
        ]

        # construct the initial state
        state = {
            "task_background": {
                "problem_type": case['task_background']['problem_type'],
                "emotion_type": case['task_background']['emotion_type'],
                "situation": case['task_background']['situation']
            },
            "dialogue_context": dialogue_context,
            "goal": goal,  # will not affect anything, only including it for coding convenience
            "response": "",  # will not affect anything, only including it for coding convenience
            "pre_goals": [''],
            "pre_topics": ['']
        }
        return state

    def step(self, state, action, generation_model, simulator):
        """
        method that update the current state of the game and return the reward
        :param state: the current state of the game
        :param action: the predicted action by the system
        :param generation_model: the response generation method
        :param simulator: the user simulator
        :return: the new state, reward, and flag indicating if the game is terminated
        """
        goal = action
        logger.info(f"[Goal]: {goal}")

        # prepare state for the response generation
        state['pred_goal'] = goal

        # generate the system response
        system_response = generation_model.generate_response(state)

        # Price-tag protocol: parse the buyer's declared price from the tag and
        # strip the tag before the utterance is stored / shown to the seller.
        # (No-op outside negotiation, where state has no 'use_price_tag' key.)
        # Only credit the price when the action actually COMMITS to one — a
        # 'deny'/'inquire' that echoes the buyer's budget is referential, not a
        # secured price, so it must not overwrite the anchor (anti reward-hack).
        if state.get('use_price_tag'):
            from utils.generation import extract_price_tag, strip_price_tag
            _strategy_now = action[0] if isinstance(action, tuple) else action
            _buyer_price = extract_price_tag(system_response)
            if _buyer_price is not None and _strategy_now in NEG_PRICE_COMMITTING:
                state['_buyer_declared_price'] = _buyer_price
            system_response = strip_price_tag(system_response)

        # update the dialogue context
        state['dialogue_context'].append({"role": "assistant", "content": system_response})

        # generate user response with LLM
        user_response = simulator.respond(state)

        # Price-tag protocol: parse the seller's declared price and strip tag.
        if state.get('use_price_tag'):
            from utils.generation import extract_price_tag, strip_price_tag
            _seller_price = extract_price_tag(user_response)
            if _seller_price is not None:
                state['_seller_declared_price'] = _seller_price
            user_response = strip_price_tag(user_response)

        # construct the new state
        # prepend the system and user reponse to the dialogue context
        # prepend the predicted goal, topic to the previous goals, topics
        state['dialogue_context'].append({'role': 'user', 'content': user_response})
        state['pre_goals'].append(goal)

        logger.info(f"[System]: {system_response}")
        logger.info(f"[USER]: {user_response}")

        # compute the reward
        reward, done, o_done = self.compute_reward(state, action, system_response, simulator.user_profile_description)

        # return the new state, intermediate reward, and termination flag.
        return state, reward, done, o_done

    def compute_reward(self, state, action, system_response, profile_description):
        """
        method that computes the rewards for emotional support conversation scenario
        :param state: the current state of the conversation
        :param action: the actioned predicted by the system
        :param system_response: the system generated response
        :param profile_description: the user profile description
        :return:
        """
        goal = action

        # compute the llm-basd assessment
        responses = get_llm_based_assessment_for_emotional_support(state,
                                                                   simulated_conversation=state['dialogue_context'],
                                                                   n=self.game_config.n,
                                                                   temperature=1.1,
                                                                   model_type=self.model_type,
                                                                   max_tokens=10
                                                                   )

        logger.debug(f"ES NLI responses={responses}")

        # used to compute the es_sr
        # indicate whether the supporter solved the seeker problem.
        rewards = []
        for response in responses:
            for key in self.game_config.reward_dict:
                if key in response.lower():
                    rewards.append(self.game_config.reward_dict[key])
                    break
        # compute the emotional support sr
        if len(rewards) == 0:
            es_sr = 0
        else:
            es_sr = sum(rewards) / len(rewards)

        # a flag indicates if a conversation is terminated
        done = 0
        turn_reward = -0.1

        # for intermediate turn, toxicity is zero.
        toxicity = 0.0

        # checking if the user reward is greater than a predefined threshold
        # that mean it is a successful conversations
        if es_sr >= self.game_config.epsilon:
            done = 1
            logger.info('--> Terminated conversation !')

            # compute toxicity
            # toxicity metric should be computed at the dialogue-level to avoid strong correlation with the avg turn
            # we wish to minimize the toxicity
            # it is equivalent to maximize the negative toxicity

            # constructing the dialogue content
            dialogue_content = ''
            for utt in state['dialogue_context']:
                dialogue_content += utt['content']

            # compute the toxicity
            # the toxicity is at the dialogue-level
            toxicity = -10.0 * get_toxicity_assessment_for_emotional_support(dialogue_content)

        else:
            # if the length of the trajectory is greater than the maximal game horizon
            # this is a failed conversation
            if len(state['dialogue_context']) == self.game_config.max_horizon:
                logger.info('Maximum number of turns reached !')

                # constructing the dialogue content
                dialogue_content = ''
                for utt in state['dialogue_context']:
                    dialogue_content += utt['content']

                # compute the toxicity
                # the toxicity is at the dialogue-level
                toxicity = -10.0 * get_toxicity_assessment_for_emotional_support(dialogue_content)

                # failed case
                done = -1
            else:
                # logger.info('The conversation is on-going !')
                pass

        rewards = []

        # emotional support success rate
        if USER_REWARD in self.game_config.objectives:
            rewards.append(es_sr)
        # toxicity score
        if TOXICITY in self.game_config.objectives:
            rewards.append(toxicity)
        # avg conversation turn
        if AVG_TURN in self.game_config.objectives:
            rewards.append(turn_reward)

        logger.debug(f"rewards={rewards}")
        return rewards, done, done
