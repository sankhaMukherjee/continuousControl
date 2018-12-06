from lib         import memory, utils
from tqdm        import tqdm
from collections import deque
from unityagents import UnityEnvironment

import numpy as np

def policy(state):
    return np.random.rand(4)

def testMemoryBuffers():

    env = UnityEnvironment(file_name='../p2_continuous-control/Reacher.app')

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    memBuffer = memory.ReplayBuffer(100)
    for i in tqdm(range(50)):
        memBuffer = utils.updateReplayBuffer(memBuffer, env, brain_name, policy, tMax = 200, gamma=1)

    print('Total size of the memorey: {}'.format(len(memBuffer.memory)))
    for i, m in enumerate(memBuffer.memory):
        rewards = [m[2]  for m in m.memory]
        
        print('Total size of memory {} = {}, Total rewards: {}'.format( i, len(m.memory), sum(rewards) ))
        # print(rewards)

    # print(memBuffer)

    for i, m in enumerate(memBuffer.sample(20, 100)):
        print('--------------[{}]-------------'.format(i))
        print('State: {}\nAction: {}\nReward: {}\nNext State: {}\nDone: {}'.format(*m))
    

    return

