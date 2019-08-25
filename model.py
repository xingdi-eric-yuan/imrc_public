import os
import copy
import logging
import numpy as np

import torch
import torch.nn.functional as F
from layers import Embedding, FastBiLSTM, LSTMCell, BiMatchLSTM, BoundaryDecoder, masked_softmax, NoisyLinear, ActionScorerAttention, ActionScorerAttentionAdvantage
from layers import EncoderBlock, CQAttention, AnswerPointer, MergeEmbeddings, StackedGraphConvolution, AggregationBlock
from generic import to_pt

logger = logging.getLogger(__name__)


class QA_DQN(torch.nn.Module):
    model_name = 'qa_dqn'

    def __init__(self, config, word_vocab, char_vocab, action_space_size):
        super(QA_DQN, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(word_vocab)
        self.char_vocab = char_vocab
        self.char_vocab_size = len(char_vocab)
        self.action_space_size = action_space_size  # previous, next, stop, ctrl+f <query>
        self.read_config()
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        model_config = self.config['model']
        # word
        self.use_pretrained_embedding = model_config['use_pretrained_embedding']
        self.word_embedding_size = model_config['word_embedding_size']
        self.word_embedding_trainable = model_config['word_embedding_trainable']
        # char
        self.char_embedding_size = model_config['char_embedding_size']
        self.char_embedding_trainable = model_config['char_embedding_trainable']
        self.embedding_dropout = model_config['embedding_dropout']

        self.encoder_layers = model_config['encoder_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.aggregation_layers = model_config['aggregation_layers']
        self.aggregation_conv_num = model_config['aggregation_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.attention_dropout = model_config['attention_dropout']
        self.action_scorer_hidden_dim = model_config['action_scorer_hidden_dim']
        self.action_scorer_softmax = model_config['action_scorer_softmax']
        self.question_answerer_hidden_dim = model_config['question_answerer_hidden_dim']

        self.pretrained_embedding_path = "crawl-300d-2M.vec.h5"
        self.noisy_net = self.config['epsilon_greedy']['noisy_net']

    def _def_layers(self):

        # word embeddings
        if self.use_pretrained_embedding:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            id2word=self.word_vocab,
                                            dropout_rate=self.embedding_dropout,
                                            load_pretrained=True,
                                            trainable=self.word_embedding_trainable,
                                            embedding_oov_init="zero",
                                            pretrained_embedding_path=self.pretrained_embedding_path)
        else:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            trainable=self.word_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        # char embeddings
        self.char_embedding = Embedding(embedding_size=self.char_embedding_size,
                                        vocab_size=self.char_vocab_size,
                                        trainable=self.char_embedding_trainable,
                                        dropout_rate=self.embedding_dropout)

        self.emb_enc = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])
        self.merge_embeddings = MergeEmbeddings(block_hidden_dim=self.block_hidden_dim, word_emb_dim=self.word_embedding_size, char_emb_dim=self.char_embedding_size, dropout=self.embedding_dropout)
        self.context_question_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.context_question_attention_resizer = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)
        self.model_enc_blks = torch.nn.ModuleList([AggregationBlock(conv_num=self.aggregation_conv_num, ch_num=self.block_hidden_dim,
                                                                    k=5, block_hidden_dim=self.block_hidden_dim,
                                                                    n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.aggregation_layers)])
        self.answer_pointer = AnswerPointer(block_hidden_dim=self.block_hidden_dim, noisy_net=self.noisy_net)
        encoder_output_dim = self.block_hidden_dim

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear

        action_scorer_output_size = 1
        action_scorer_advantage_output_size = self.action_space_size
        action_scorer_ctrlf_output_size = 1
        action_scorer_ctrlf_advantage_output_size = self.word_vocab_size

        if self.action_scorer_hidden_dim > 0:
            self.action_scorer_shared_linear = linear_function(encoder_output_dim, self.action_scorer_hidden_dim)
            action_scorer_input_size = self.action_scorer_hidden_dim
        else:
            action_scorer_input_size = encoder_output_dim

        self.action_scorer_ctrlf = linear_function(action_scorer_input_size, action_scorer_ctrlf_output_size)
        self.action_scorer_ctrlf_advantage = linear_function(action_scorer_input_size, action_scorer_ctrlf_advantage_output_size)

        self.action_scorer_linear = linear_function(action_scorer_input_size, action_scorer_output_size)
        self.action_scorer_linear_advantage = linear_function(action_scorer_input_size, action_scorer_advantage_output_size)

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask):
        X = self.context_question_attention(doc_encodings, q_encodings, doc_mask, q_mask)
        M0 = self.context_question_attention_resizer(X)
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)
        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        for i in range(self.aggregation_layers):
             M0 = self.model_enc_blks[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        return M0

    def representation_generator(self, _input_words, _input_chars):
        embeddings, mask = self.word_embedding(_input_words)  # batch x time x emb
        char_embeddings, _ = self.char_embedding(_input_chars)  # batch x time x nchar x emb
        merged_embeddings = self.merge_embeddings(embeddings, char_embeddings, mask)  # batch x time x emb
        square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x time x time
        for i in range(self.encoder_layers):
            encoding_sequence = self.emb_enc[i](merged_embeddings, square_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc

        return encoding_sequence, mask

    def action_scorer(self, state_representation_sequence, ctrlf_word_mask):
        # state_representation: batch x time x enc_dim
        # ctrlf_word_mask: batch x vocab
        state_representation, _ = torch.max(state_representation_sequence, 1)
        if self.action_scorer_hidden_dim > 0:
            state_representation = self.action_scorer_shared_linear(state_representation)  # action scorer hidden dim
            state_representation = torch.relu(state_representation)

        a_rank = self.action_scorer_linear(state_representation)  #  batch x 1
        a_rank_advantage = self.action_scorer_linear_advantage(state_representation)  # advantage stream  batch x n_vocab
        a_rank = a_rank + a_rank_advantage - a_rank_advantage.mean(1, keepdim=True)  # combine streams
        if self.action_scorer_softmax:
            a_rank = masked_softmax(a_rank, axis=-1)  # batch x n_vocab

        ctrlf_rank = self.action_scorer_ctrlf(state_representation)  #  batch x 1
        ctrlf_rank_advantage = self.action_scorer_ctrlf_advantage(state_representation)  # advantage stream  batch x n_vocab
        ctrlf_rank_advantage = ctrlf_rank_advantage * ctrlf_word_mask
        ctrlf_rank = ctrlf_rank + ctrlf_rank_advantage - ctrlf_rank_advantage.mean(1, keepdim=True)  # combine streams
        ctrlf_rank = ctrlf_rank * ctrlf_word_mask
        if self.action_scorer_softmax:
            ctrlf_rank = masked_softmax(ctrlf_rank, ctrlf_word_mask, axis=-1)  # batch x n_vocab
        return a_rank, ctrlf_rank

    def answer_question(self, matching_representation_sequence, doc_mask):
        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        M0 = matching_representation_sequence
        M1 = M0
        for i in range(self.aggregation_layers):
             M0 = self.model_enc_blks[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        M2 = M0
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)
        for i in range(self.aggregation_layers):
             M0 = self.model_enc_blks[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        M3 = M0
        pred = self.answer_pointer(M1, M2, M3, doc_mask)
        return pred

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_ctrlf.reset_noise()
            self.action_scorer_ctrlf_advantage.reset_noise()
            self.action_scorer_linear.reset_noise()
            self.action_scorer_linear_advantage.reset_noise()
            self.answer_pointer.zero_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.action_scorer_ctrlf.zero_noise()
            self.action_scorer_ctrlf_advantage.zero_noise()
            self.action_scorer_linear.zero_noise()
            self.action_scorer_linear_advantage.zero_noise()
            self.answer_pointer.zero_noise()
