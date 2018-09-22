import numpy as np
import tensorflow as tf

import argparse
import pickle
import matplotlib.pyplot as plt
 
import gym
import custom_gym

from model import Model
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="environment id")
parser.add_argument("--name", type=str, help="name")
args = parser.parse_args()

env_id = args.env
name = args.name

env = gym.make(env_id)
data = load_data("Asset/expert/%s.pickle" % name)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model = Model(len(data[0][0]), len(data[0][1]))
    saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name], max_to_keep=10)
    saver.restore(sess, tf.train.latest_checkpoint("Asset/model/%s" % name))

    # evaluation
    while True:
        obs = env.reset()
        done = False
        while not done:
            env.render()       
            action = model.predict([obs])
            obs, r, done, info = env.step(action[0][0])
                