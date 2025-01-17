import os
import time
import yaml
import numpy as np
from os.path import join as pjoin
from gamified_squad import GamifiedSquad
from gamified_newsqa import GamifiedNewsQA


with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

config["evaluate"]["batch_size"] = 1
if config['general']['dataset'] == "squad":
    env = GamifiedSquad(config)
else:
    env = GamifiedNewsQA(config)
env.split_reset("valid")
print("There are {:d} validation games...".format(env.valid_size))


try:
    while(True):
        print("------------------------------------------------------ NEW GAME, ctlf+c to exit")
        obs, infos = env.reset(random=False)
        print("=" * 50)
        print("question: " + infos[0]["q"])
        step_idx = 0
        while True:
            if step_idx > 100:
                break
            print("-" * 50 +  ", step ", step_idx)
            print(obs[0])
            command = input("type your command (previous, next, stop, ctrl+f <token>): ")
            if command == "p":
                command = "previous"
            elif command == "n":
                command = "next"
            elif command == "s":
                command = "stop"
            elif command.startswith("cf "):
                command = command.replace("cf ", "ctrl+f ")

            obs, infos = env.step([command])
            step_idx += 1
            if obs is None or command == "stop":
                break
        print("=" * 50)
        print("answer: ")
        for a in infos[0]["a"]:
            print(a)
# At any point you can hit Ctrl+C to break.
except KeyboardInterrupt:
    print("Bye!")