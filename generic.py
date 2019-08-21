import numpy as np
import torch
import random
import uuid
import os
import re
import time
import string
from collections import Counter
from os.path import join as pjoin


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix.copy()).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix.copy()).type(torch.FloatTensor))


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        try:
            ids.append(word2id[word])
        except KeyError:
            ids.append(1)
    return ids


def max_len(list_of_list):
    return max(map(len, list_of_list))


def max_tensor_len(list_of_tensor, dim):
    tmp = []
    for t in list_of_tensor:
        tmp.append(t.size(dim))
    return max(tmp)


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def is_sub_list(long_list, short_list):
    # return True if the short list is a sublist of the long one
    if long_list is None or short_list is None:
        return False
    if len(long_list) == 0 or len(short_list) == 0:
        return False
    key = short_list[0]
    for i in range(len(long_list)):
        if long_list[i] == key:
            if long_list[i: i + len(short_list)] == short_list:
                return True
    return False


def get_sufficient_info_reward(observation_string_list, ground_truth_answers):
    sufficient_info_reward = []

    for i in range(len(observation_string_list)):
        observation = observation_string_list[i]
        observation = normalize_string(observation)
        gt_answers = [normalize_string(item) for item in ground_truth_answers[i]]
        has_answer = 0.0
        for j in range(len(gt_answers)):
            if gt_answers[j] in observation:
                has_answer = 1.0
                break
        sufficient_info_reward.append(has_answer)
    return np.array(sufficient_info_reward)


def get_qa_reward(pred_string, ground_truth_string, mode="f1"):
    qa_reward = []
    for i in range(len(pred_string)):
        if mode == "f1":
            qa_reward.append(f1_score_over_ground_truths(pred_string[i], ground_truth_string[i]))
        else:
            pred = normalize_string(pred_string[i])
            gt = [normalize_string(item) for item in ground_truth_string[i]]
            qa_reward.append(float(pred in gt))
    return np.array(qa_reward)


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i][0]])
    return torch.stack(res, 0)


def get_answer_strings(sentence_words, head_and_tails):
    res = []
    for sent, ht in zip(sentence_words, head_and_tails):
        sent = sent.split()
        h, t = ht[0], ht[1]
        if h >= len(sent):
            h = len(sent) - 1
        if t >= len(sent):
            t = len(sent) - 1
        if h < t:
            words = sent[h: t + 1]
        else:
            words = sent[t: h + 1]
        res.append(" ".join(words))
    return res


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|<bos>|<eos>|<sep>|<pad>|<unk>)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1.0
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = f1_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def list_of_token_list_to_char_input(list_of_token_list, char2id):
    batch_size = len(list_of_token_list)
    max_token_number = max_len(list_of_token_list)
    max_char_number = max([max_len(item) for item in list_of_token_list])
    if max_char_number < 6:
        max_char_number = 6
    res = np.zeros((batch_size, max_token_number, max_char_number), dtype='int32')
    for i in range(batch_size):
        for j in range(len(list_of_token_list[i])):
            for k in range(len(list_of_token_list[i][j])):
                res[i][j][k] = char2id[list_of_token_list[i][j][k]]
    return res


def list_of_word_id_list_to_char_input(list_of_word_id_list, id2word, char2id):
    res = []
    for i in range(len(list_of_word_id_list)):
        res.append([id2word[item] for item in list_of_word_id_list[i]])
    return list_of_token_list_to_char_input(res, char2id)


def get_answer_position(list_of_strings, list_of_answers):
    res = []
    for i in range(len(list_of_strings)):
        if list_of_answers[i] not in list_of_strings[i]:
            res.append([0, 0])
        else:
            story = list_of_strings[i].split()
            answer = list_of_answers[i].split()
            tmp = [0, 0]
            for j in range(len(story)):
                if story[j] != answer[0]:
                    continue
                if story[j: j + len(answer)] == answer:
                    tmp = [j, j + len(answer) - 1]
                    break
            res.append(tmp)
    return res
