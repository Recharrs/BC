import os
import time
import numpy as np
import tensorflow as tf

import argparse
import pickle
import matplotlib.pyplot as plt
 
import gym
import custom_gym

from model import Model
from utils import *

class bc_script:
    def __init__(self, args):
        self.args = {}
        self.args["id"] = args.id
        self.args["name"] = args.name
        self.args["env"] = args.env
        self.args["save_name"] = args.save_name
        self.args["num_eval"] = args.num_eval
        self.args["type"] = args.type
        self.args["model_path"] = args.model_path

        self.data = self.load_data()
        self.batch_size = 16

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
    def load_data(self):
        data = load_data("Asset/expert/%s/00.pickle" % self.args["name"])
        np.random.seed(int(time.time()))
        np.random.shuffle(data)
        print(len(data))
        return data

    def train_and_eval(self):
        data_size_set = [16, 32, 64, 128, 256]
        self.make_dir(data_size)
        with tf.Session(config=self.config) as sess:
            self.setting()
            
            for data_size in data_size_set:
                # train
                self.timesteps = []
                self.total_evals = []
                
                # load / reset model
                if len(self.data) < data_size: 
                    break   
                if self.args["model_path"] != "":
                    saver.restore(sess, tf.train.latest_checkpoint(model_path))
                else:
                    sess.run(tf.global_variables_initializer())
                
                # train
                data_train = self.data[:data_size]
                print(len(data_train))
                for step in range(10):
                    self.train(data_train, data_size, step, sess)
                    
                # eval
                evals = self.eval()
                self.collect(data_size, evals)
            
            # save
            self.save_result(data_size)

    def setting(self):
        self.env = gym.make(self.args["env"])
        self.model = Model(len(self.data[0][0]), len(self.data[0][1]))

        self.writer = tf.summary.FileWriter("./Asset/logdir/%s" % (self.args["save_name"]))
        self.saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name], max_to_keep=10)

    def make_dir(self, data_size):
        os.makedirs("./Asset/model/%s/%s/model" % (self.args["save_name"], self.args["id"]), exist_ok=True)
        os.makedirs("./Asset/result/%s" % (self.args["save_name"]), exist_ok=True)

    def train(self, data_train, data_size, step, sess):
        # batch
        states, actions = get_batch(data_train, self.batch_size, step) 
        loss, summary = self.model.train(states, actions)
        # tensorboard
        self.writer.add_summary(summary, step) 
        save_path = self.saver.save(sess, "Asset/model/%s/%s/model" % (self.args["save_name"], self.args["id"]), global_step=step)

    def eval(self):
        evals = []
        for _ in range(self.args["num_eval"]):
            obs = self.env.reset()
            done = False
            while not done:               
                action = self.model.predict([obs])
                obs, r, done, info = self.env.step(action[0][0])
                if done: 
                    if self.args["type"] == "goal": evals.append( (info["dist"]) )
                    if self.args["type"] == "traj": evals.append( (info["min_dist_cp"] + info['min_dist_ft']) )
        return evals
    
    def collect(self, step, evals):
        v1, v2 = step, np.mean(evals)
        print("data: %d, dist: %f" % (v1, v2))
        self.timesteps.append(v1)
        self.total_evals.append(v2)

    def save_result(self, data_size):
        with open("Asset/result/%s/%s.pickle" % (self.args["save_name"], self.args["id"]), "wb") as fp:
            result = {"data_size": self.timesteps, "dist": self.total_evals}
            pickle.dump(result, fp)
        print("save: Asset/result/%s/%s.pickle" % (self.args["save_name"], self.args["id"]))

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

    # run
    script_1 = bc_script(args)
    script_1.train_and_eval()
