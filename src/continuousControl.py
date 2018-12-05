from lib         import memory, utils
from collections import deque
from unityagents import UnityEnvironment

from tests import memTests

import numpy as np


def main():

    memTests.testMemoryBuffers()



    # env = UnityEnvironment(file_name='../p2_continuous-control/Reacher.app')

    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]

    # memBuffer = memory.ReplayBuffer(100)
    # for i in range(10):
    #     memBuffer = utils.updateReplayBuffer(memBuffer, env, brain_name, policy, maxBuffer = 200, tMax = 200, gamma=1)

    # print('Total size of the memorey: {}'.format(len(memBuffer.memory)))
    # for i, m in enumerate(memBuffer.memory):
    #     rewards = [m[2]  for m in m.memory]
        
    #     print('Total size of memory {} = {}, Total rewards: {}'.format( i, len(m.memory), sum(rewards) ))

    # # print(memBuffer)

    # for i, m in enumerate(memBuffer.sample(5)):
    #     print(f'--------------[{i}]-------------')
    #     # print(m)
    #     # samples = m.sample(10)
    #     print('State: {}\nAction: {}\nReward: {}\nNext State: {}\nDone: {}'.format(*m))
        


    # # reset the environment
    # env_info = env.reset(train_mode=True)[brain_name]

    # # number of agents
    # num_agents = len(env_info.agents)
    # print('Number of agents:', num_agents)

    # # size of each action
    # action_size = brain.vector_action_space_size
    # print('Size of each action:', action_size)

    # # examine the state space 
    # states = env_info.vector_observations
    # state_size = states.shape[1]
    # print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    # print('The state for the first agent looks like:', states[0])

    # env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
    # states = env_info.vector_observations                  # get the current state (for each agent)
    # scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    # while True:
    #     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    #     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    #     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    #     next_states = env_info.vector_observations         # get next state (for each agent)
    #     rewards = env_info.rewards                         # get reward (for each agent)
    #     dones = env_info.local_done                        # see if episode finished
    #     scores += env_info.rewards                         # update the score (for each agent)
    #     states = next_states                               # roll over states to next time step
    #     if np.any(dones):                                  # exit loop if episode finished
    #         break
    # print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


    return


if __name__ == '__main__':
    main()

