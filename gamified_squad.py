import os
import re
import codecs
import operator
from tqdm import tqdm

import numpy as np
import gym


class GamifiedSquad(gym.Env):

    def __init__(self, config):
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        # load dataset from file
        self.dataset = dict(
            train=dict(
                story=[], story_sentence_list=[], question=[], answers=[], answer_positions=[]),
            valid=dict(
                story=[], story_sentence_list=[], question=[], answers=[], answer_positions=[]),
            test=dict(
                story=[], story_sentence_list=[], question=[], answers=[], answer_positions=[]),
        )
        self.tmp_vocab = {}
        for split in ["train", "valid", "test"]:
            self.load_dataset(split)
        print("loaded dataset from %s ..." % self.data_path)
        self.train_size = len(self.dataset["train"]["story_sentence_list"])
        self.valid_size = len(self.dataset["valid"]["story_sentence_list"])
        self.test_size = len(self.dataset["test"]["story_sentence_list"])
        self.batch_pointer, self.step_counter = None, None
        self.train_batch_pointer = 0
        self.data_size, self.batch_size, self.data, self.infos, self.max_nb_steps_per_episode = None, None, None, None, None
        self.current_story = None
        self.current_story_sentence_list, self.current_question, self.current_answers, self.current_answer_positions, self.last_actions = None, None, None, None, None
        self.split = "train"

        if self.max_vocab_size > 5:
            sorted_vocab = sorted(self.tmp_vocab.items(), key=operator.itemgetter(1))[::-1]  # descending
            vocab = [item[0] for item in sorted_vocab[: self.max_vocab_size - 5]]
        else:
            vocab = [item for item in self.tmp_vocab]
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"] + vocab
        char_vocab = self.get_char_vocab(vocab)
        char_vocab = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"] + char_vocab
        with codecs.open("vocab.txt", "w", "utf-8") as text_file:
            text_file.write("\n".join(vocab))
        with codecs.open("char_vocab.txt", "w", "utf-8") as text_file:
            text_file.write("\n".join(char_vocab))

    def get_char_vocab(self, word_list):
        chars = set()
        for word in word_list:
            for ch in word:
                chars.add(ch)
        chars = list(chars)
        chars.sort()
        return chars

    def update_vocab(self, inp):
        word_list = inp.split()
        for w in word_list:
            if w not in self.tmp_vocab:
                self.tmp_vocab[w] = 0
            self.tmp_vocab[w] += 1

    def load_dataset(self, split):
        file_path = self.data_path + "/" + (split if split != "test" else "dev")
        story_file = file_path + "-v1.1-story.txt"
        question_file = file_path + "-v1.1-question.txt"
        answer_range_file = file_path + "-v1.1-answer-range.txt"
        with codecs.open(story_file, mode='r', encoding='utf-8', errors='ignore') as story_reader,\
                codecs.open(question_file, mode='r', encoding='utf-8', errors='ignore') as question_reader,\
                codecs.open(answer_range_file, mode='r', encoding='utf-8', errors='ignore') as answer_range_reader:
            for _story, _question, _a_ranges in tqdm(zip(story_reader, question_reader, answer_range_reader)):
                _story, _question, _a_ranges = _story.strip(), _question.strip(), _a_ranges.strip()
                if len(_a_ranges) <= 0:
                    continue
                _story, _question = _story.lower(), _question.lower()
                self.update_vocab(_story)
                self.update_vocab(_question)
                story_word_list = _story.split()

                _a_ranges = _a_ranges.split(" ||| ")
                if len(_a_ranges) <= 0:
                    continue
                answers = []
                h, t = _a_ranges[0].split(':', 1)
                h, t = int(h), int(t)
                if h >= len(story_word_list) or t - 1 >= len(story_word_list):
                    continue
                self.dataset[split]["answer_positions"].append([h, t - 1])

                for arange in _a_ranges:
                    head, tail = arange.split(':', 1)
                    answers.append(" ".join(story_word_list[int(head): int(tail)]))
                story_sent_list = re.split(r" \. | \? | \! ", _story)
                story_sent_list = [item.strip() for item in story_sent_list]
                story_sent_list = [item for item in story_sent_list if len(item) > 0]
                self.dataset[split]["story"].append(_story)
                self.dataset[split]["story_sentence_list"].append(story_sent_list)
                self.dataset[split]["question"].append(_question)
                self.dataset[split]["answers"].append(answers)
    
    def read_config(self):
        self.data_path = self.config["general"]["data_path"]
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]
        self.start_from_beginning = self.config["general"]["start_from_beginning"]
        self.insert_random_distractors = self.config["general"]["insert_random_distractors"]

        self.training_batch_size = self.config["training"]["batch_size"]
        self.training_max_nb_steps_per_episode = self.config["training"]["max_nb_steps_per_episode"]
        self.evaluate_batch_size = self.config["evaluate"]["batch_size"]
        self.evaluate_max_nb_steps_per_episode = self.config["evaluate"]["max_nb_steps_per_episode"]

        self.max_vocab_size = self.config["model"]["max_vocab_size"]

    def split_reset(self, split):
        if split == "train":
            self.data_size = self.train_size
            self.batch_size = self.training_batch_size
            self.max_nb_steps_per_episode = self.training_max_nb_steps_per_episode
        elif split == "valid":
            self.data_size = self.valid_size
            self.batch_size = self.evaluate_batch_size
            self.max_nb_steps_per_episode = self.evaluate_max_nb_steps_per_episode
        else:
            self.data_size = self.test_size
            self.batch_size = self.evaluate_batch_size
            self.max_nb_steps_per_episode = self.evaluate_max_nb_steps_per_episode
        
        if split == "train" and self.use_this_many_data > 0:
            self.data = {"story": self.dataset[split]["story"][: self.use_this_many_data],
                         "story_sentence_list": self.dataset[split]["story_sentence_list"][: self.use_this_many_data],
                         "question": self.dataset[split]["question"][: self.use_this_many_data],
                         "answers": self.dataset[split]["answers"][: self.use_this_many_data],
                         "answer_positions": self.dataset[split]["answer_positions"][: self.use_this_many_data]}
            self.data_size = self.use_this_many_data
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0
        self.current_story_sentence_list, self.current_question, self.current_answers = None, None, None
        self.infos = None

    def reset(self, random=True):
        if random is True:
            # randomly sample a batch of d-q-a tuple
            indices = np.random.choice(self.data_size, self.batch_size).tolist()
        else:
            # just take next batch
            if self.split == "train":
                self.batch_pointer = self.train_batch_pointer
            indices = np.arange(self.batch_pointer, self.batch_pointer + self.batch_size).tolist()
            self.batch_pointer += self.batch_size
            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0
            if self.split == "train":
                self.train_batch_pointer = self.batch_pointer
        self.current_story_sentence_list, self.current_question, self.current_answers = [], [], []
        for idx in indices:
            if idx >= len(self.data["story_sentence_list"]):
                break
            story_sentence_list = self.data["story_sentence_list"][idx]
            if self.insert_random_distractors > 0 and self.split not in ["valid", "test"]:
                # randomly sample k paragraphs from dataset
                distractor_indices = np.random.choice(self.data_size, self.insert_random_distractors).tolist()
                distractor_sentences = []
                for distractor_idx in distractor_indices:
                    distractor_sentences += self.data["story_sentence_list"][distractor_idx]
                # randomly insert these distractor sentences into story sentence list
                for item in distractor_sentences:
                    where = np.random.randint(len(story_sentence_list) + 1)
                    story_sentence_list = story_sentence_list[:where] + [item] + story_sentence_list[where:]

            self.current_story_sentence_list.append(story_sentence_list)
            self.current_question.append(self.data["question"][idx])
            self.current_answers.append(self.data["answers"][idx])
        
        # for each story_sentence_list, randomly sample a sentence as init observation
        obs = []
        which_sentence = []
        for item in self.current_story_sentence_list:
            init_idx = 0 if self.start_from_beginning is True else np.random.choice(len(item), 1)[0]
            obs.append(item[init_idx])
            which_sentence.append(init_idx)
        infos = [{"q": q, "a": a, "which": which, "stopped": False} for q, a, which in zip(self.current_question, self.current_answers, which_sentence)]
        self.infos = infos
        self.last_actions = None
        self.step_counter = 0
        return obs, infos

    def step(self, actions):
        if self.step_counter > self.max_nb_steps_per_episode:
            return None, None
        # given action, return new obs sentence, and update infos
        obs, infos = [], []
        for i in range(len(actions)):
            stopped = False
            if actions[i] == "next":
                new_which = self.infos[i]["which"] + 1
                if new_which >= len(self.current_story_sentence_list[i]):
                    new_which = 0
            elif actions[i] == "previous":
                new_which = self.infos[i]["which"] - 1
                if new_which < 0:
                    new_which = len(self.current_story_sentence_list[i]) - 1
            elif actions[i] == "stop":
                new_which = self.infos[i]["which"]
                stopped = True
            elif actions[i].startswith("ctrl+f"):
                # for now just exact match
                query = actions[i][6:].strip()
                curr_which = self.infos[i]["which"]
                for j in range(1, len(self.current_story_sentence_list[i]) + 1):
                    new_which = (curr_which + j) % len(self.current_story_sentence_list[i])
                    sent = self.current_story_sentence_list[i][new_which]
                    if query in sent.split():
                        break
            else:
                raise NotImplementedError
            obs.append(self.current_story_sentence_list[i][new_which])
            infos.append({"q": self.infos[i]["q"], "a": self.infos[i]["a"], "which": new_which, "stopped": stopped})

        self.last_actions = actions
        self.infos = infos
        self.step_counter += 1
        return obs, infos

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        np.random.seed(seed)