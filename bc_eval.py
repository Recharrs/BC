import numpy as np
import tensorflow as tf

import gym
import custom_gym

from bc import Model

if __name__ == "__main__":
    with open("logger.txt", "w+") as log:

        env = gym.make("FiveTarget-v1")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # Setup
            model = Model(9, 2)
            
            writer = tf.summary.FileWriter("./logdir")
            saver = tf.train.Saver([v for v in tf.global_variables() if "model" in v.name])
            
            for i in range(1000):
                
                saver.restore(sess, "./model/model-%d" % i)
                dists = []

                # Evaluation                
                for _ in range(3):
                    print(i * 10 + _)
                    
                    obs = env.reset()
                    done = False

                    while not done:
                        env.render()
                        
                        action = model.predict([obs])
                        obs, r, done, info = env.step(action[0][0])
                        if done:
                            dists.append(info["dist"])
                           
                print("data: %d , dist: %f" % (66 * (i + 1), np.mean(dists)))
                print("data: %d , dist: %f" % (66 * (i + 1), np.mean(dists)), file=log)
