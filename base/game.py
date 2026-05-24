import os.path
import re
import math
import time
from abc import ABC, abstractmethod
from loguru import logger

from config.constants import DURECDIAL, INSPIRED, CRAIGSLIST_BARGAIN, ES_CONV
from utils.prompt import get_llm_based_assessment_for_recommendation, get_llm_based_assessment_for_negotiation, \
    get_llm_based_assessment_for_emotional_support, get_toxicity_assessment_for_emotional_support, \
    get_user_sentiment_for_item_recommendation
from config.constants import SUCCESS_RATE, DEAL_RATE, ITEM_FREQ, AVG_TURN, SL_RATIO, FAIRNESS, TOXICITY, USER_REWARD


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

        # update the dialogue context
        state['dialogue_context'].append({"role": "assistant", "content": system_response})

        # generate user response with LLM
        user_response = simulator.respond(state)

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
        responses = get_llm_based_assessment_for_negotiation(simulated_conversation=state['dialogue_context'],
                                                             n=self.game_config.n,
                                                             temperature=1.1,
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
        neg_sr = sum(deals) / len(deals)
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

        # extracting the price proposed by the system
        if len(system_prices) > 0 and action != "inform":
            system_price = max(system_prices)
        else:
            system_price = state['task_background']['seller_price']

        system_price = float(system_price)

        # encourage the system to gain more benefit
        # this reward is to encourage the model to propose beneficial price for its self.
        sl_ratio = (system_price - state['task_background']['seller_price']) / (
                state['task_background']['buyer_price'] - state['task_background']['seller_price'])

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
        mid_price = (state['task_background']['seller_price'] + state['task_background']['buyer_price']) / 2

        # if the system price is equivalent to the mid price
        # we give the system a high fairness reward
        fairness = 0.5 - abs(system_price - mid_price) / (
                state['task_background']['seller_price'] - state['task_background']['buyer_price'])

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

        # checking if there is a deal
        # that mean the neg_sr should be greater than a predefined threshold
        if neg_sr >= self.game_config.epsilon:
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

        # vector-valued reward function
        # sl_ratio / fairness are attenuated except at SUCCESSFUL terminal
        # (done == 1). Both intermediate (done == 0) and timeout-failure
        # (done == -1) get the small signal, so the agent cannot reward-hack
        # by anchoring at a buyer-favorable price without closing a deal.
        # 0.3 chosen as sweet spot: strong enough TD signal for the anchor
        # strategy to dominate capitulation (0.1 made the gap too small and
        # skills collapsed to 1-turn deals), but small enough that "refuse
        # forever" doesn't beat actually closing a deal.
        shaping = 1.0 if done == 1 else 0.3

        reward = []

        # sale-list ratio
        if SL_RATIO in self.game_config.objectives:
            reward.append(shaping * sl_ratio)
        # fairness
        if FAIRNESS in self.game_config.objectives:
            reward.append(shaping * fairness)
        # SR / deal_rate (YAML uses "deal_rate"; constants has both "sr" and "deal_rate")
        if SUCCESS_RATE in self.game_config.objectives or DEAL_RATE in self.game_config.objectives:
            reward.append(neg_sr)
        # avg_turn: constant -0.1 per turn, NOT subject to shaping.
        # Encourages closing deals quickly (fewer turns = smaller cumulative penalty).
        if AVG_TURN in self.game_config.objectives:
            reward.append(turn_reward)   # turn_reward = -0.1 defined above

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

        # update the dialogue context
        state['dialogue_context'].append({"role": "assistant", "content": system_response})

        # generate user response with LLM
        user_response = simulator.respond(state)

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
