import numpy as np
import tensorflow as tf

import gym
import custom_gym

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
        tf.summary.scalar("loss", self.loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self.loss)

        ''' Summary '''
        self.summary = tf.summary.merge_all()

        if restore_path is None:
            self.sess.run(tf.global_variables_initializer())

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

if __name__ == "__main__":
    env = gym.make("FiveTarget-v1")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Setup
        model = Model(9, 2)
        
        writer = tf.summary.FileWriter("./logdir")
        saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name])
        
        saver.restore(sess, "./model_random/model.ckpt")

        # Evaluation
        
        for i in range(1000):
            
            obs = env.reset()
            done = False

            while True:
                env.render()
                
                action = model.predict([obs])
                obs, r, done, _ = env.step(action[0][0])

                if done:break
