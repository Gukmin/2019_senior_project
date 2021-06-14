from env_supermario import SuperMarioEnv
from nes_py.wrappers import JoypadSpace

env = SuperMarioEnv()
env = JoypadSpace(env,actions.RIGHT_ONLY)

for i_episode in range(20):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation,reward,done,info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1));
            break