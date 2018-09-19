import gym
#import custom_gym
import numpy as np
from custom_gym.classic_control import FiveTargetEnv_v1

# Create Environment
#env = gym.make('FiveTarget-v1')
env = FiveTargetEnv_v1()

# Print action & observation space
print(env.action_space)
print(env.observation_space)

# Target Position
targets = range(18, 180, 36)
targets = [np.deg2rad(x) for x in targets]
targets = np.array([(np.cos(x), np.sin(x)) for x in targets])
print(targets)

# parameter
rotate_scale = 0.3
threshold = rotate_scale * 0.01

# Test Environment
experts = []
for i_episode in range(1000):
  expert = {
    "states": [],
    "actions": [],
  }

  # Reset Environment
  obs = env.reset(i_episode % 5)
  t = 0

  agent = obs[:2]
  instr = obs[-5:]
  target = targets[np.where(instr == 1)[0]][0]
  face = np.array([0, 1])
  
  # Run Episode
  while True:
    # Render Environment
    env.render()
    
    # Interact with Environment
    action = [1, 0]
    # target direction & delta angle
    target_dir = target - agent
    cos_theta = np.sum(target_dir * face) / (np.linalg.norm(target_dir)*np.linalg.norm(face))
    cos_theta = np.clip(cos_theta, -1, 1)
    delta_theta = np.arccos(cos_theta)
    
    if delta_theta > threshold:
      # right
      dir_sign = 1
      right_dir = np.array([target_dir[1], -target_dir[0]])
      if np.sum(right_dir * face) < 0:
        dir_sign = -1
        
      delta_theta = np.clip(delta_theta, -1, 1)
      action[1] = dir_sign * delta_theta / rotate_scale

    else:
      action[0] = 1

    expert_action = np.array(action)

    # Random action
    #if t < 10:
    action = env.action_space.sample()
    action = action*1.0 + expert_action*0.0

    # collect trajectory
    expert["states"].append(obs)
    expert["actions"].append(expert_action)

    obs, reward, done, info = env.step(action)

    agent = obs[:2]
    face = obs[2:4]
    
    t = t+1

    # Check Done
    if done:
      print("Episode finished after {} timesteps".format(t+1))
      break
  experts.append(expert)

import pickle
with open("expert/python_env_expert_random.pickle", "wb") as file:
  pickle.dump(experts, file)

# Close Environment
env.close()
