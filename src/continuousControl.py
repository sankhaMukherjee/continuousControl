from lib         import NN
from collections import deque
from unityagents import UnityEnvironment

import numpy as np

def main():

    env = UnityEnvironment(file_name='../p2_continuous-control/Reacher.app')
    brainName = env.brain_names[0]

    agent = NN.Agent()
    # agent.updateBuffer(env, brainName)
    ep = agent.playEpisode(env, brainName)
    for e in ep:
        print('-'*50)
        vals = np.array([m[2] for m in e.memory])
        print(vals.sum())

    
    return


if __name__ == '__main__':
    main()

