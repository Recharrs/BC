import numpy as np
import tensorflow as tf
import pickle

import gym
import custom_gym

import matplotlib.pyplot as plt

class Model:
    def __init__(self, state_dim, action_dim, restore_path=None):
        self.sess = tf.get_default_session()
        self.state_dim = state_dim
        self.action_dim = action_dim

        ''' Build Model '''
        with tf.variable_scope("model") as scope:
            self.s = tf.placeholder(tf.float32, [None, self.state_dim])
            self.a = tf.placeholder(tf.float32, [None, self.action_dim])
            
            h1 = tf.layers.dense(self.s, 64, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)

            mu = tf.layers.dense(h2, self.action_dim)
            sigma = tf.layers.dense(h2, self.action_dim)

            pd = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
            self.a_pred = pd.sample()

        ''' Loss & Opt '''
        self.loss = tf.losses.mean_squared_error(labels=self.a, predictions=self.a_pred)
        tf.summary.scalar("data_size", tf.shape(self.s)[0])
        tf.summary.scalar("loss", self.loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss)

        ''' Summary '''
        self.summary = tf.summary.merge_all()

    def train(self, batch_s, batch_a):
        ''' action should be normalized to [-1, 1]'''
        loss, summary, _ = self.sess.run(
            [self.loss, self.summary, self.train_op], feed_dict={
            self.s: batch_s,
            self.a: batch_a,
        })
        return loss, summary

    def predict(self, s):
        action = self.sess.run([self.a_pred], feed_dict={
            self.s: s
        })
        return action

def get_batch(data, batch_size, step):
    num_batch = len(data) // batch_size
    batch_idx = step % num_batch 
    
    start = (batch_idx) * batch_size
    end = (batch_idx + 1) * batch_size

    return zip(*data[start:end])

if __name__ == "__main__":
    batch_size = 1024

    with open("logger_2.txt", "w+") as log:
        with open("traj/state_action_pair.pickle", "rb") as file:
            data = pickle.load(file)
        
        env = gym.make("ReacherFiveTarget-v1")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            model = Model(11, 2)
            sess.run(tf.global_variables_initializer())
            
            writer = tf.summary.FileWriter("./logdir/reacher")
            saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name], max_to_keep=10)

            for i in range(1000):
                # train a batch
                states, actions = get_batch(data, batch_size, i)
                loss, summary = model.train(states, actions)
  
                writer.add_summary(summary, i)
                save_path = saver.save(sess, "./model/reacher/model", global_step=i)
        
                # evaluation
                dists = []
                for _ in range(20):
                    obs = env.reset()
                    done = False
                    while not done:               
                        action = model.predict([obs])
                        obs, r, done, info = env.step(action[0][0])
                        if done:
                            dists.append(info["dist"])

                print("data: %d , dist: %f" % (batch_size * (i + 1), np.mean(dists)))
                print("data: %d , dist: %f" % (batch_size * (i + 1), np.mean(dists)), file=log)

    with open("logger_2.txt", "r") as file:
        lines = file.read().splitlines()

        data_size = []
        dists = []

        for line in lines:
            data = line.split(" ")
            data_size.append(int(data[1]))
            dists.append(float(data[4]))
        
        plt.xlabel("# of data")
        plt.ylabel("mean distance")

        plt.plot(data_size, dists)
        plt.savefig("reacher-learning_curve.png")
        plt.show()

        with open("logger_2_cooked.pickle", "wb") as file_out:
            data_out = {
                "data_size": data_size,
                "dists": dists,
            }
            pickle.dump(data_out, file_out)