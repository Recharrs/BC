import numpy as np
import tensorflow as tf
 
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

    def train(self, batch_s, batch_a):
        ''' action should be normalized to [-1, 1]'''
        loss, summary, _ = self.sess.run(
            [self.loss, self.summary, self.train_op], feed_dict={
            self.s: batch_s,
            self.a: batch_a,
        })
        return loss, summary

    def predict(self, s, a):
        action = self.sess.run([self.a_pred], feed_dict={
            self.s: s
        })
        return action

if __name__ == "__main__":
    import pickle
    with open("expert/python_env_expert.pickle", "rb") as file:
        experts = pickle.load(file)
    
    states = np.array([np.array(expert["states"]) for expert in experts])
    actions = np.array([np.array(expert["actions"]) for expert in experts])
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print(states.shape)
        model = Model(states.shape[-1], actions.shape[-1])
        
        writer = tf.summary.FileWriter("./logdir/almost_random/")
        saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name])

        for i in range(10000):
            loss, summary = model.train(states, actions)
            
            writer.add_summary(summary, i)
            save_path = saver.save(sess, "./model/model.ckpt")

            print(loss)

        
