import random
import yaml
import copy
import codecs
from collections import namedtuple
from os.path import join as pjoin

import numpy as np

import torch
import torch.nn.functional as F

import memory
from model import QA_DQN
from generic import to_np, to_pt, _words_to_ids, pad_sequences, get_qa_reward, get_answer_strings, get_answer_position
from generic import max_len, max_tensor_len, ez_gather_dim_1, list_of_token_list_to_char_input, list_of_word_id_list_to_char_input
from layers import NegativeLogLoss, compute_mask, masked_softmax


# a snapshot of state to be stored in replay memory for question answering
qa_Transition = namedtuple('qa_Transition', ('observation_list', 'quest_list', 'answer_strings'))


class ObservationPool(object):

    def __init__(self, capacity=1):
        self.capacity = capacity

    def identical_with_history(self, new_stuff, list_of_old_stuff):
        for i in range(len(list_of_old_stuff)):
            if new_stuff == list_of_old_stuff[i]:
                return True
        return False

    def push_batch(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            if not self.identical_with_history(stuff[i], self.memory[i]):
                self.memory[i].append(stuff[i])
            if len(self.memory[i]) > self.capacity:
                self.memory[i] = self.memory[i][-self.capacity:]
                
    def push_one(self, which, stuff):
        assert which < len(self.memory)
        if not self.identical_with_history(stuff, self.memory[which]):
            self.memory[which].append(stuff)
        if len(self.memory[which]) > self.capacity:
            self.memory[which] = self.memory[which][-self.capacity:]

    def get_last(self):
        return [item[-1] for item in self.memory]

    def get(self, which=None):
        if which is not None:
            assert which < len(self.memory)
            output = " ".join(self.memory[which])
            return output

        output = []
        for i in range(len(self.memory)):
            tmp = " ".join(self.memory[i])
            output.append(tmp)
        return output

    def get_sent_list(self):
        return copy.copy(self.memory)

    def reset(self, batch_size):
        self.memory = []
        for _ in range(batch_size):
            self.memory.append([])

    def __len__(self):
        return len(self.memory)


class HistoryScoreCache(object):

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0
        self.alpha_rewards, self.beta_rewards = [], []

    def push(self, is_prior=False, reward=0.0, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = qa_Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
            self.alpha_rewards.append(reward)
            if len(self.alpha_rewards) > self.alpha_capacity:
                self.alpha_rewards = self.alpha_rewards[1:]
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = qa_Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity
            self.beta_rewards.append(reward)
            if len(self.beta_rewards) > self.beta_capacity:
                self.beta_rewards = self.beta_rewards[1:]

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def avg_rewards(self):
        if len(self.alpha_rewards) == 0 and len(self.beta_rewards) == 0 :
            return 0.0
        return np.mean(self.alpha_rewards + self.beta_rewards)

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)


class Agent:
    def __init__(self):
        """
        Arguments:
            word_vocab: List of words supported.
        """
        self.mode = "train"
        with open("config.yaml") as reader:
            self.config = yaml.safe_load(reader)
        print(self.config)
        self.load_config()
        if self.disable_prev_next:
            self.id2action = ["stop", "ctrl+f"]
            self.action2id = {"stop": 0, "ctrl+f": 1}
        else:
            self.id2action = ["previous", "next", "stop", "ctrl+f"]
            self.action2id = {"previous": 0, "next": 1, "stop": 2, "ctrl+f": 3}

        self.online_net = QA_DQN(config=self.config, word_vocab=self.word_vocab, char_vocab=self.char_vocab, action_space_size=len(self.id2action))
        self.target_net = QA_DQN(config=self.config, word_vocab=self.word_vocab, char_vocab=self.char_vocab, action_space_size=len(self.id2action))
        self.online_net.train()
        self.target_net.train()
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False

        if self.use_cuda:
            self.online_net.cuda()
            self.target_net.cuda()

        self.naozi = ObservationPool(capacity=self.naozi_capacity)

        # optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config['training']['optimizer']['learning_rate'])
        self.clip_grad_norm = self.config['training']['optimizer']['clip_grad_norm']

    def load_config(self):
        # word vocab
        self.word_vocab = []
        with codecs.open("./vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        # char vocab
        self.char_vocab = []
        with codecs.open("./char_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.char_vocab.append(line.strip())
        self.char2id = {}
        for i, w in enumerate(self.char_vocab):
            self.char2id[w] = i
        # stopwords
        self.stopwords = []
        with codecs.open("./corenlp_stopwords.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.stopwords.append(line.strip())
        self.data_path = self.config['general']['data_path']
        self.qa_reward_prior_threshold = self.config['general']['qa_reward_prior_threshold']
        self.naozi_capacity = self.config['general']['naozi_capacity']
        self.generate_or_point = self.config['general']['generate_or_point']
        self.disable_prev_next = self.config['general']['disable_prev_next']

        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.max_episode = self.config['training']['max_episode']
        self.target_net_update_frequency = self.config['training']['target_net_update_frequency']
        self.learn_start_from_this_episode = self.config['training']['learn_start_from_this_episode']
        self.run_eval = self.config['evaluate']['run_eval']
        self.eval_batch_size = self.config['evaluate']['batch_size']
        self.eval_max_nb_steps_per_episode = self.config['evaluate']['max_nb_steps_per_episode']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.save_frequency = self.config['checkpoint']['save_frequency']
        self.report_frequency = self.config['checkpoint']['report_frequency']
        self.load_pretrained = self.config['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['checkpoint']['load_from_tag']

        self.discount_gamma = self.config['training']['discount_gamma']
        self.qa_loss_lambda = self.config['training']['qa_loss_lambda']
        self.interaction_loss_lambda = self.config['training']['interaction_loss_lambda']

        # replay buffer and updates
        self.replay_batch_size = self.config['replay']['replay_batch_size']
        self.replay_memory = memory.PrioritizedReplayMemory(self.config['replay']['replay_memory_capacity'],
                                                            priority_fraction=self.config['replay']['replay_memory_priority_fraction'],
                                                            discount_gamma=self.discount_gamma)
        self.qa_replay_memory = PrioritizedReplayMemory(self.config['replay']['replay_memory_capacity'],
                                                        priority_fraction=self.config['replay']['replay_memory_priority_fraction'])
        self.update_per_k_game_steps = self.config['replay']['update_per_k_game_steps']
        self.multi_step = self.config['replay']['multi_step']

        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.noisy_net = self.config['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()
    
    def zero_noise(self):
        if self.noisy_net:
            self.online_net.zero_noise()
            self.target_net.zero_noise()

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                state_dict = torch.load(load_from)
            else:
                state_dict = torch.load(load_from, map_location='cpu')
            self.online_net.load_state_dict(state_dict)
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def init(self, obs, infos):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        # reset agent, get vocabulary masks for verbs / adjectives / nouns
        self.scores = []
        self.dones = []
        self.prev_actions = [["" for _ in range(len(obs))]]
        self.prev_step_is_still_interacting = np.ones((len(obs),), dtype="float32")  # 1s and starts to be 0 when previous action is "stop"
        self.naozi.reset(batch_size=len(obs))
        self.not_finished_yet = None

    def get_agent_inputs(self, string_list):
        sentence_token_list = [item.split() for item in string_list]
        sentence_id_list = [_words_to_ids(tokens, self.word2id) for tokens in sentence_token_list]
        input_sentence_char = list_of_token_list_to_char_input(sentence_token_list, self.char2id)
        input_sentence = pad_sequences(sentence_id_list, maxlen=max_len(sentence_id_list)).astype('int32')
        input_sentence = to_pt(input_sentence, self.use_cuda)
        input_sentence_char = to_pt(input_sentence_char, self.use_cuda)
        return input_sentence, input_sentence_char, sentence_id_list

    def get_game_quest_info(self, infos):
        return [item["q"] for item in infos]
    
    def get_word_mask(self, list_of_query_id_list, list_of_observation_id_list):
        batch_size = len(list_of_query_id_list)
        if self.generate_or_point == "generate":
            sw_ids = set()
            for sw in self.stopwords:
                if sw in self.word2id:
                    sw_ids.add(self.word2id[sw])
            word_mask = np.ones((batch_size, len(self.word_vocab)), dtype="float32")
            for _id in sw_ids:
                word_mask[:, _id] = 0.0
            word_mask = to_pt(word_mask, enable_cuda=self.use_cuda, type="float")
            mask_word_id_list = []
            all_word_ids = set(np.arange(len(self.word_vocab)).tolist())
            m = list(all_word_ids - sw_ids)
            for i in range(batch_size):
                mask_word_id_list.append(m)
            return word_mask, mask_word_id_list

        word_mask_np = np.zeros((batch_size, len(self.word_vocab)), dtype="float32")
        mask_word_id_list = []
        for i in range(batch_size):
            mask_word_id_list.append(set())
            for w_idx in list_of_query_id_list[i]:
                if self.word_vocab[w_idx] in self.stopwords:
                    continue
                word_mask_np[i][w_idx] = 1.0
                mask_word_id_list[i].add(w_idx)
            if self.generate_or_point == "qmpoint":
                for w_idx in list_of_observation_id_list[i]:
                    if self.word_vocab[w_idx] in self.stopwords:
                        continue
                    word_mask_np[i][w_idx] = 1.0
                    mask_word_id_list[i].add(w_idx)
        mask_word_id_list = [list(item) for item in mask_word_id_list]
        for i in range(len(mask_word_id_list)):
            if len(mask_word_id_list[i]) == 0:
                mask_word_id_list[i].append(self.word2id[","])  # just in case this list is empty
                word_mask_np[i][self.word2id[","]] = 1.0
                continue
        word_mask = to_pt(word_mask_np, enable_cuda=self.use_cuda, type="float")
        return word_mask, mask_word_id_list

    def generate_commands(self, action_indices, ctrlf_indices):

        action_indices_np = to_np(action_indices)
        ctrlf_indices_np = to_np(ctrlf_indices)
        res_str = []
        batch_size = action_indices_np.shape[0]
        for i in range(batch_size):
            which = action_indices_np[i][0]
            if which == self.action2id["ctrl+f"]:
                which_word = ctrlf_indices_np[i][0]
                res_str.append("ctrl+f " + self.word_vocab[which_word])
            elif which < len(self.id2action):
                res_str.append(self.id2action[which])
            else:
                raise NotImplementedError
        return res_str

    def choose_random_command(self, action_rank, mask_word_ids=None):
        """
        Generate a command randomly, for epsilon greedy.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(-1)
        if mask_word_ids is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(mask_word_ids[j]))
            indices = np.array(indices)
        action_indices = to_pt(indices, self.use_cuda).unsqueeze(-1)  # batch x 1
        return action_indices

    def choose_maxQ_command(self, action_rank, word_mask=None):
        """
        Generate a command by maximum q values, for epsilon greedy.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if word_mask is not None:
            assert word_mask.size() == action_rank.size(), (word_mask.size().shape, action_rank.size())
            action_rank = action_rank * word_mask
        action_indices = torch.argmax(action_rank, -1, keepdim=True)  # batch x 1
        return action_indices

    def get_match_representations(self, input_description, input_description_char, input_quest, input_quest_char, use_model="online"):
        model = self.online_net if use_model == "online" else self.target_net
        description_representation_sequence, description_mask = model.representation_generator(input_description, input_description_char)
        quest_representation_sequence, quest_mask = model.representation_generator(input_quest, input_quest_char)

        match_representation_sequence = model.get_match_representations(description_representation_sequence,
                                                                        description_mask,
                                                                        quest_representation_sequence,
                                                                        quest_mask)
        match_representation_sequence = match_representation_sequence * description_mask.unsqueeze(-1)
        return match_representation_sequence

    def get_ranks(self, input_description, input_description_char, input_quest, input_quest_char, ctrlf_word_mask, use_model="online"):
        """
        Given input description tensor, and previous hidden and cell states, call model forward, to get Q values of words.
        """
        model = self.online_net if use_model == "online" else self.target_net
        match_representation_sequence = self.get_match_representations(input_description, input_description_char, input_quest, input_quest_char, use_model=use_model)
        action_rank, ctrlf_rank = model.action_scorer(match_representation_sequence, ctrlf_word_mask)
        return action_rank, ctrlf_rank

    def act_greedy(self, obs, infos, input_quest, input_quest_char, quest_id_list):
        with torch.no_grad():
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.copy(obs))
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.copy(obs[i]))

            description_list = self.naozi.get()
            input_description, input_description_char, description_id_list = self.get_agent_inputs(description_list)
            ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, description_id_list)
            action_rank, ctrlf_rank = self.get_ranks(input_description, input_description_char, input_quest, input_quest_char, ctrlf_word_mask, use_model="online")  # list of batch x vocab
            action_indices = self.choose_maxQ_command(action_rank)
            ctrlf_indices = self.choose_maxQ_command(ctrlf_rank, ctrlf_word_mask)
            chosen_strings = self.generate_commands(action_indices, ctrlf_indices)

            for i in range(batch_size):
                if chosen_strings[i] == "stop":
                    self.not_finished_yet[i] = 0.0

            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0
            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [description_list, action_indices, ctrlf_indices, to_pt(self.prev_step_is_still_interacting, self.use_cuda, "float")]

            # cache new info in current game step into caches
            self.prev_actions.append(chosen_strings)
            return chosen_strings, replay_info

    def act_random(self, obs, infos, input_quest, input_quest_char, quest_id_list):
        with torch.no_grad():
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.copy(obs))
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.copy(obs[i]))
            
            description_list = self.naozi.get()
            input_description, input_description_char, description_id_list = self.get_agent_inputs(description_list)
            ctrlf_word_mask, ctrlf_word_ids = self.get_word_mask(quest_id_list, description_id_list)
            # generate commands for one game step, epsilon greedy is applied, i.e.,
            # there is epsilon of chance to generate random commands
            action_rank, ctrlf_rank = self.get_ranks(input_description, input_description_char, input_quest, input_quest_char, ctrlf_word_mask, use_model="online")  # list of batch x vocab
            action_indices = self.choose_random_command(action_rank)
            ctrlf_indices = self.choose_random_command(ctrlf_rank, ctrlf_word_ids)
            chosen_strings = self.generate_commands(action_indices, ctrlf_indices)

            for i in range(batch_size):
                if chosen_strings[i] == "stop":
                    self.not_finished_yet[i] = 0.0

            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0
            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [description_list, action_indices, ctrlf_indices, to_pt(self.prev_step_is_still_interacting, self.use_cuda, "float")]

            # cache new info in current game step into caches
            self.prev_actions.append(chosen_strings)
            return chosen_strings, replay_info

    def act(self, obs, infos, input_quest, input_quest_char, quest_id_list, random=False):

        with torch.no_grad():
            if self.mode == "eval":
                return self.act_greedy(obs, infos, input_quest, input_quest_char, quest_id_list)
            if random:
                return self.act_random(obs, infos, input_quest, input_quest_char, quest_id_list)
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.copy(obs))
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.copy(obs[i]))
            
            description_list = self.naozi.get()
            input_description, input_description_char, description_id_list = self.get_agent_inputs(description_list)
            ctrlf_word_mask, ctrlf_word_ids = self.get_word_mask(quest_id_list, description_id_list)
            # generate commands for one game step, epsilon greedy is applied, i.e.,
            # there is epsilon of chance to generate random commands
            action_rank, ctrlf_rank = self.get_ranks(input_description, input_description_char, input_quest, input_quest_char, ctrlf_word_mask, use_model="online")  # list of batch x vocab
            action_indices_maxq = self.choose_maxQ_command(action_rank)
            action_indices_random = self.choose_random_command(action_rank)
            ctrlf_indices_maxq = self.choose_maxQ_command(ctrlf_rank, ctrlf_word_mask)
            ctrlf_indices_random = self.choose_random_command(ctrlf_rank, ctrlf_word_ids)
            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(input_description.size(0), 1))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon
            less_than_epsilon = to_pt(less_than_epsilon, self.use_cuda, type='long')
            greater_than_epsilon = to_pt(greater_than_epsilon, self.use_cuda, type='long')

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_ctrlf_indices = less_than_epsilon * ctrlf_indices_random + greater_than_epsilon * ctrlf_indices_maxq
            chosen_strings = self.generate_commands(chosen_indices, chosen_ctrlf_indices)

            for i in range(batch_size):
                if chosen_strings[i] == "stop":
                    self.not_finished_yet[i] = 0.0

            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0
            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [description_list, chosen_indices, chosen_ctrlf_indices, to_pt(self.prev_step_is_still_interacting, self.use_cuda, "float")]

            # cache new info in current game step into caches
            self.prev_actions.append(chosen_strings)
            return chosen_strings, replay_info

    def point_random_position(self, point_distribution, mask):
        """
        Generate a command by random, for epsilon greedy.

        Arguments:
            point_distribution: Q values for each position batch x time x 2.
            mask: position masks.
        """
        batch_size = point_distribution.size(0)
        mask_np = to_np(mask)  # batch x time
        indices = []
        for i in range(batch_size):
            msk = mask_np[i]  # time
            indices.append(np.random.choice(len(msk), 2, p=msk / np.sum(msk, -1)))
        indices = to_pt(np.stack(indices, 0), self.use_cuda)   # batch x 2
        return indices

    def point_maxq_position(self, point_distribution, mask):
        """
        Generate a command by maximum q values, for epsilon greedy.

        Arguments:
            point_distribution: Q values for each position batch x time x 2.
            mask: position masks.
        """
        point_distribution_np = to_np(point_distribution)  # batch x time
        mask_np = to_np(mask)  # batch x time
        point_distribution_np = point_distribution_np - np.min(point_distribution_np) + 1e-2  # minus the min value, so that all values are non-negative
        point_distribution_np = point_distribution_np * np.expand_dims(mask_np, -1)  # batch x time x 2
        indices = np.argmax(point_distribution_np, 1)  # batch x 2
        indices = to_pt(np.array(indices), self.use_cuda)   # batch x 2
        return indices

    def answer_question_act(self, observation_list, quest_list):
        with torch.no_grad():

            point_rank, mask = self.answer_question(observation_list, quest_list, use_model="online")  # batch x time x 2
            positions_maxq = self.point_maxq_position(point_rank, mask)  # batch x 2
            chosen_position = positions_maxq
            return chosen_position  # batch x 2

    def get_dqn_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None, None

        data = self.replay_memory.get_batch(self.replay_batch_size, self.multi_step)
        if data is None:
            return None, None
        obs_list, quest_list, action_indices, ctrlf_indices, rewards, next_obs_list, actual_ns = data

        input_observation, input_observation_char, observation_id_list = self.get_agent_inputs(obs_list)
        input_quest, input_quest_char, quest_id_list = self.get_agent_inputs(quest_list)
        next_input_observation, next_input_observation_char, next_observation_id_list = self.get_agent_inputs(next_obs_list)

        ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, observation_id_list)
        next_ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, next_observation_id_list)

        action_rank, ctrlf_rank = self.get_ranks(input_observation, input_observation_char, input_quest, input_quest_char, ctrlf_word_mask, use_model="online")  # batch x vocab
        # ps_a
        q_value_action = ez_gather_dim_1(action_rank, action_indices).squeeze(1)  # batch
        q_value_ctrlf = ez_gather_dim_1(ctrlf_rank, ctrlf_indices).squeeze(1)  # batch
        is_ctrlf = torch.eq(action_indices, float(self.action2id["ctrl+f"])).float()  # when the action is ctrl+f, batch
        q_value = (q_value_action + q_value_ctrlf * is_ctrlf) / (is_ctrlf + 1)  # masked average
        # q_value = torch.mean(torch.stack([q_value_action, q_value_ctrlf], -1), -1)

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            
            # pns Probabilities p(s_t+n, ·; θonline)
            next_action_rank, next_ctrlf_rank = self.get_ranks(next_input_observation, next_input_observation_char, input_quest, input_quest_char, next_ctrlf_word_mask, use_model="online")  # batch x vocab
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_command(next_action_rank)  # batch x 1
            next_ctrlf_indices = self.choose_maxQ_command(next_ctrlf_rank, next_ctrlf_word_mask)  # batch x 1
            # pns # Probabilities p(s_t+n, ·; θtarget)
            next_action_rank, next_ctrlf_rank = self.get_ranks(next_input_observation, next_input_observation_char, input_quest, input_quest_char, next_ctrlf_word_mask, use_model="target")  # batch x vocab
            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value_action = ez_gather_dim_1(next_action_rank, next_action_indices).squeeze(1)  # batch
            next_q_value_ctrlf = ez_gather_dim_1(next_ctrlf_rank, next_ctrlf_indices).squeeze(1)  # batch
            next_is_ctrlf = torch.eq(next_action_indices, float(self.action2id["ctrl+f"])).float()  # when the action is ctrl+f, batch
            next_q_value = (next_q_value_action + next_q_value_ctrlf * next_is_ctrlf) / (next_is_ctrlf + 1)  # masked average

            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def update_interaction(self):
        # update neural model by replaying snapshots in replay memory
        interaction_loss, q_value = self.get_dqn_loss()
        if interaction_loss is None:
            return None, None
        loss = interaction_loss * self.interaction_loss_lambda
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(interaction_loss)), to_np(torch.mean(q_value))

    def get_qa_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.qa_replay_memory) < self.replay_batch_size:
            return None
        transitions = self.qa_replay_memory.sample(self.replay_batch_size)
        batch = qa_Transition(*zip(*transitions))

        answer_distribution, obs_mask = self.answer_question(batch.observation_list, batch.quest_list, use_model="online")  # answer_distribution is batch x time x 2
        answer_distribution = masked_softmax(answer_distribution, obs_mask.unsqueeze(-1), axis=1)

        answer_strings = [item[0] for item in batch.answer_strings]
        groundtruth_answer_positions = get_answer_position(batch.observation_list, answer_strings)  # list: batch x 2
        groundtruth = pad_sequences(groundtruth_answer_positions).astype('int32')
        groundtruth = to_pt(groundtruth, self.use_cuda)  # batch x 2
        batch_loss = NegativeLogLoss(answer_distribution * obs_mask.unsqueeze(-1), groundtruth)

        return torch.mean(batch_loss)

    def update_qa(self):
        # update neural model by replaying snapshots in replay memory
        qa_loss = self.get_qa_loss()
        if qa_loss is None:
            return None
        loss = qa_loss * self.qa_loss_lambda
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(qa_loss))

    def answer_question(self, observation_list, quest_list, use_model="online"):
        # first pad matching_representation_sequence, and get the mask
        model = self.online_net if use_model == "online" else self.target_net

        input_observation, input_observation_char, _ = self.get_agent_inputs(observation_list)
        input_quest, input_quest_char, _ = self.get_agent_inputs(quest_list)
        matching_representation_sequence = self.get_match_representations(input_observation, input_observation_char, input_quest, input_quest_char, use_model=use_model)
        # get mask
        mask = compute_mask(input_observation)
        # returns batch x time x 2
        point_rank = model.answer_question(matching_representation_sequence, mask)

        return point_rank, mask

    def finish_of_episode(self, episode_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()

        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)
            self.epsilon = max(self.epsilon, 0.0)
