import os
import numpy as np
import tensorflow as tf

import argparse
import pickle
import matplotlib.pyplot as plt
 
import gym
import custom_gym

from model import Model
from utils import *

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="environment id")
parser.add_argument("--name", type=str, help="name")
parser.add_argument("--id", type=str, help="id")
parser.add_argument("--num_train", type=int)
parser.add_argument("--num_eval", type=int)
args = parser.parse_args()

# make directory
os.makedirs("./Asset/logdir/%s" % args.name, exist_ok=True)
os.makedirs("./Asset/model/%s/%s/model" % (args.name, args.id), exist_ok=True)
os.makedirs("./Asset/result/%s" % args.name, exist_ok=True)
os.makedirs("./Asset/picture/%s" % args.name, exist_ok=True)

# session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # load data
    data = load_data("Asset/expert/%s/00.pickle" % args.name)
    print(len(data))
    batch_size = 1000

    data_size = []
    total_evals = []

    # set env & model
    env = gym.make(args.env)
    model = Model(len(data[0][0]), len(data[0][1]))
    sess.run(tf.global_variables_initializer())

    # set tensorboard
    writer = tf.summary.FileWriter("./Asset/logdir/%s" % args.name)
    saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name], max_to_keep=10)

    # train & test
    for train_data_size in range(1,1000):
        sess.run(tf.global_variables_initializer())    

        if len(data) < train_data_size * 1000: break
        np.random.shuffle(data)
        data_train = data[:train_data_size*1000]

        # train
        for i in range(len(data_train) // batch_size):
            # batch
            states, actions = get_batch(data_train, batch_size, i)
            loss, summary = model.train(states, actions)
            
            # tensorboard
            writer.add_summary(summary, i)
            save_path = saver.save(sess, "Asset/model/%s/%s/model" % (args.name, args.id), global_step=i)

        # test
        evals = []
        for _ in range(args.num_eval):
            obs = env.reset()
            done = False
            while not done:               
                action = model.predict([obs])
                obs, r, done, info = env.step(action[0][0])
                if done: evals.append( (info["dist"] ) )

        v1, v2 = train_data_size * 1000, np.mean(evals)
        print("data: %d, dist: %f" % (v1, v2))
            
        data_size.append(v1)
        total_evals.append(v2)

    # save result
    with open("Asset/result/%s/%s.pickle" % (args.name, args.id), "wb") as file_out:
        data_out = {"data_size": data_size, "dist": total_evals}
        pickle.dump(data_out, file_out)
    plt.plot(data_size, total_evals)
    plt.savefig("Asset/picture/%s/%s.png" % (args.name, args.id))

