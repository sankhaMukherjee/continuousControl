from lib         import NN
from collections import deque
from unityagents import UnityEnvironment

import numpy as np

def main():

    env = UnityEnvironment(file_name='../p2_continuous-control/Reacher.app')
    brainName = env.brain_names[0]

    agent = NN.Agent()
    agent.updateBuffer(env, brainName)
    agent.learn(10, 100, 5)

    # This is for checking the current episode
    result = agent.playEpisode(env, brainName)
    print(result)
    

    
    return


if __name__ == '__main__':
    main()

