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
    parser.add_argument("--root", type=str, default="save_0")
    args = parser.parse_args()

    def print_log(*s):
        if args.random_seed == "00":
            print(*s)
    
    # batch_size
    root = args.root
    num_train = 300
    batch_size = 64
    human_demo_size = args.human_demo_size

    # sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # path
    expert_data_path = "Asset/expert/%s/00.pickle" % (
        args.expert_name)
    logdir_path = "./Asset/%s/logdir/%s" % (
        root, args.save_name)    
    
    save_model_dir_path = "./Asset/%s/model/%s/%s/%s" % (
        root, args.save_name, human_demo_size, args.random_seed)
    save_result_dir_path = "./Asset/%s/result/%s/%s" % (
        root, args.save_name, human_demo_size)

    save_model_path = "./Asset/%s/model/%s/%s/%s/model" % (
        root, args.save_name, human_demo_size, args.random_seed)
    load_model_path = "./Asset/%s/model/%s/%s/%s" % (
        root, args.model_path, human_demo_size, args.random_seed)

    save_result_file_path = "Asset/%s/result/%s/%s/%s.pickle" % (
        root, args.save_name, human_demo_size, args.random_seed)

    # expert data
    expert_data = load_data(expert_data_path)
    np.random.seed(int(time.time()))
    np.random.shuffle(expert_data)

    # env & model & other
    env = gym.make(args.env_id)
    model = Model(len(expert_data[0][0]), len(expert_data[0][1]))
    
    save_variable = [v for v in tf.global_variables() if "model" in v.name]
    writer = tf.summary.FileWriter(logdir_path)
    saver = tf.train.Saver(max_to_keep=1)

    # make dir
    os.makedirs(save_model_dir_path, exist_ok=True)
    os.makedirs(save_result_dir_path, exist_ok=True)

    # define
    print_log(args.env_id, ":", len(expert_data[:human_demo_size]))

    def train(step):
        data_train = expert_data[:human_demo_size]
        # batch
        states, actions = get_batch(data_train, batch_size, step)
        loss, summary = model.train(states, actions)
        # tensorboard & model
        writer.add_summary(summary, step)
        saver.save(sess, save_model_path, global_step = step)
        print_log("save model:", save_model_path)

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

    # random init / restore
    if args.model_path != "":
        print_log("restore model: %s" % (load_model_path))
        saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
    else:
        sess.run(tf.global_variables_initializer())

    # output data
    output_data = {"data_size": [], "dist": []}

    # iteration
    for step in range(num_train):
        # train & eval
        _ = train(step)
        evals = evaluation()
        # collect
        v1, v2 = step, np.mean(evals)
        print_log("training: data: %d, dist: %f" % (v1, v2))
        output_data["data_size"].append(v1)
        output_data["dist"].append(v2)

    # save result
    with open(save_result_file_path, "wb") as fp:
        pickle.dump(output_data, fp)
    print_log("save result: %s" % save_result_file_path)
    print_log("----")
