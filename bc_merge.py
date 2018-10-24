import argparse
import os
import pickle
import time

import gym
import numpy as np
import tensorflow as tf

import custom_gym
from model import Model
from utils import *

class BCScript:
    def __init__(self, sess, args):
        # args
        self.sess = sess
        self.args = vars(args)


        # load dataset
        self.data = load_data("Asset/expert/%s/00.pickle" % self.args["name"])
        np.random.seed(int(time.time()))
        np.random.shuffle(self.data)
        print(len(self.data))

        # make dir
        os.makedirs("./Asset/model/%s/%s/model" %
                    (self.args["save_name"], self.args["id"]), exist_ok=True)
        os.makedirs("./Asset/result/%s" %
                    (self.args["save_name"]), exist_ok=True)

        # hyper
        self.batch_size = 16


        self.env = gym.make(self.args["env"])
        self.model = Model(len(self.data[0][0]), len(self.data[0][1]))

        self.writer = tf.summary.FileWriter(
            "./Asset/logdir/%s" % (self.args["save_name"]))
        self.saver = tf.train.Saver(
            [v for v in tf.global_variables() if "model" in v.name], max_to_keep=10)

        # sets
        self.record = {"timesteps": [], "total_evals": []}

    def train(self, data_size, step):
        data_train = self.data[:data_size]
        # batch
        states, actions = get_batch(data_train, self.batch_size, step)
        loss, summary = self.model.train(states, actions)
        # tensorboard
        self.writer.add_summary(summary, step)
        self.saver.save(self.sess, "Asset/model/%s/%s/model" %
                        (self.args["save_name"], self.args["id"]), global_step=step)
        return loss

    def evaluation(self):
        evals = []
        for _ in range(self.args["num_eval"]):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.model.predict([obs])
                obs, _, done, info = self.env.step(action[0][0])
                if done:
                    if self.args["type"] == "goal":
                        evals.append((info["dist"]))
                    if self.args["type"] == "traj":
                        evals.append(
                            (info["min_dist_cp"] + info['min_dist_ft']))
        return evals

    def collect(self, step, evals):
        v1, v2 = step, np.mean(evals)
        print("data: %d, dist: %f" % (v1, v2))
        self.record["timesteps"].append(v1)
        self.record["total_evals"].append(v2)

    def save_result(self):
        save_path = "Asset/result/%s/%s.pickle" % (
            self.args["save_name"], self.args["id"])
        with open(save_path, "wb") as fp:
            result = {
                "data_size": self.record["timesteps"], "dist": self.record["total_evals"]}
            pickle.dump(result, fp)
        print("save_path: %s" % save_path)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="environment id")
    parser.add_argument("--name", type=str, help="name")
    parser.add_argument("--save-name", type=str, help="save name")
    parser.add_argument("--id", type=str, help="id")
    parser.add_argument("--num_train", type=int)
    parser.add_argument("--num_eval", type=int)
    parser.add_argument("--type", type=str)
    parser.add_argument("--model-path", type=str, defualt="")
    args = parser.parse_args()

    # session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # script
    script_1 = BCScript(sess, args)

    # train_and_evaluation
    for data_size in [16, 32, 64, 128, 256]:
        # load / reset model
        if len(script_1.data) < data_size:
            break
        if script_1.args["model_path"] != "":
            script_1.saver.restore(
                script_1.sess, tf.train.latest_checkpoint(model_path))
        else:
            script_1.sess.run(tf.global_variables_initializer())

        # train
        for step in range(10):
            script_1.train(data_train, data_size, step, sess)
        # eval
        script_1.collect(data_size, script_1.evaluation())

    # save result
    script_1.save_result()
