import argparse
import os
import pickle
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import custom_gym
from model import Model
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MassPointTraj-v1")
    parser.add_argument("--expert-name", type=str, default="MassPointTraj-v1")
    parser.add_argument("--save-name", type=str, default="MassPointTraj-v1-test")
    parser.add_argument("--random-seed", type=str, default="00")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--type", type=str, default="traj")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--human_demo_size", type=int)
    args = parser.parse_args()

    # batch_size
    batch_size = 64
    human_demo_size_set = [args.human_demo_size]

    # sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # expert data
    expert_data_path = "Asset/expert/%s/00.pickle" % args.expert_name
    expert_data = load_data(expert_data_path)
    np.random.seed(int(time.time()))
    np.random.shuffle(expert_data)

    # env & model & other
    env = gym.make(args.env_id)
    model = Model(len(expert_data[0][0]), len(expert_data[0][1]))

    logdir_path = "./Asset/logdir/%s" % args.save_name
    save_variable = [v for v in tf.global_variables() if "model" in v.name]

    writer = tf.summary.FileWriter(logdir_path)
    saver = tf.train.Saver(max_to_keep=1)

    # make dir
    for human_demo_size in human_demo_size_set:
        save_model_dir_path = "./Asset/model/%s/%s/%s" % (
            args.save_name, human_demo_size, args.random_seed)
        save_result_dir_path = "./Asset/result/%s/%s" % (
            args.save_name, human_demo_size)
        os.makedirs(save_model_dir_path, exist_ok=True)
        os.makedirs(save_result_dir_path, exist_ok=True)

    # define
    def train(human_demo_size, step):
        data_train = expert_data[:human_demo_size]
        print(len(data_train))
        save_model_path = "./Asset/model/%s/%s/%s/model" % (
            args.save_name, human_demo_size, args.random_seed)
        # batch
        states, actions = get_batch(data_train, batch_size, step)
        loss, summary = model.train(states, actions)
        # tensorboard & model
        writer.add_summary(summary, step)
        saver.save(sess, save_model_path, global_step = step)
        print(save_model_path)

    def evaluation():
        evals = []
        for _ in range(args.num_eval):
            obs = env.reset()
            done = False
            while not done:
                action = model.predict([obs])
                obs, _, done, info = env.step(action[0][0])
            if args.type == "goal":
                evals.append((info["dist"]))
            if args.type == "traj":
                evals.append((info["min_dist_cp"] + info['min_dist_ft']))
        return evals

    # run
    for human_demo_size in human_demo_size_set:
        # random init / restore
        if args.model_path != "":
            model_path = "%s/%s/%s" % (args.model_path, human_demo_size, args.random_seed)
            print("restore: %s" % tf.train.latest_checkpoint(model_path))
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            sess.run(tf.global_variables_initializer())

        # output data
        output_data = {"data_size": [], "dist": []}

        # iteration
        for step in range(300):
            # train & eval
            _ = train(human_demo_size, step)
            evals = evaluation()
            # collect
            v1, v2 = step, np.mean(evals)
            print("data: %d, dist: %f" % (v1, v2))
            output_data["data_size"].append(v1)
            output_data["dist"].append(v2)

        # save result
        save_result_file_path = "Asset/result/%s/%s/%s.pickle" % (
            args.save_name, human_demo_size, args.random_seed)
        with open(save_result_file_path, "wb") as fp:
            pickle.dump(output_data, fp)
        print("save: %s" % save_result_file_path)
