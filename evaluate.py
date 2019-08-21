import numpy as np
import generic
import os
import json
from generic import _words_to_ids, to_np


def evaluate(env, agent, valid_test="valid"):

    env.seed(42)
    env.split_reset(valid_test)
    agent.eval()
    print_qa_acc, print_correct_state_acc, print_steps = [], [], []

    while(True):
        obs, infos = env.reset(random=False)

        agent.init(obs, infos)
        quest_list = agent.get_game_quest_info(infos)
        input_quest, input_quest_char, quest_id_list = agent.get_agent_inputs(quest_list)

        tmp_replay_buffer = []

        for step_no in range(agent.eval_max_nb_steps_per_episode):
            commands, replay_info = agent.act_greedy(obs, infos, input_quest, input_quest_char, quest_id_list)

            tmp_replay_buffer.append(replay_info)
            obs, infos = env.step(commands)

            still_running = generic.to_np(replay_info[-1])
            if np.sum(still_running) == 0:
                break

        # The agent has exhausted all steps, now answer question.
        chosen_head_tails = agent.answer_question_act(agent.naozi.get(), quest_list)  # batch
        chosen_head_tails_np = generic.to_np(chosen_head_tails)
        chosen_answer_strings = generic.get_answer_strings(agent.naozi.get(), chosen_head_tails_np)
        answer_strings = [item["a"] for item in infos]
        masks_np = [generic.to_np(item[-1]) for item in tmp_replay_buffer]

        qa_reward_np = generic.get_qa_reward(chosen_answer_strings, answer_strings)
        correct_state_reward_np = generic.get_sufficient_info_reward(agent.naozi.get(), answer_strings)
        step_masks_np = np.sum(np.array(masks_np), 0)
        for i in range(len(qa_reward_np)):
            # if the answer is totally wrong, we assume it used all steps
            if qa_reward_np[i] == 0.0:
                step_masks_np[i] = agent.eval_max_nb_steps_per_episode
        print_qa_acc += qa_reward_np.tolist()
        print_correct_state_acc += correct_state_reward_np.tolist()
        print_steps += step_masks_np.tolist()
        if env.batch_pointer == 0:
            break

    print("===== Eval =====: qa acc: {:2.3f} | correct state: {:2.3f} | used steps: {:2.3f}".format(np.mean(np.array(print_qa_acc)), np.mean(np.array(print_correct_state_acc)), np.mean(np.array(print_steps))))
    return np.mean(np.array(print_qa_acc)), np.mean(np.array(print_correct_state_acc)), np.mean(np.array(print_steps))
