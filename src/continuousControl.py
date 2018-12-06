from lib         import NN
from collections import deque
from unityagents import UnityEnvironment

import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np

from tqdm import tqdm

import json

def main():

    config = json.load(open('config.json'))

    env = UnityEnvironment(file_name='../p2_continuous-control/Reacher.app')
    brainName = env.brain_names[0]

    agent = NN.Agent()


    allResults = []

    for i in tqdm(range(config['mainLearn']['totalN'])):

        if (i % config['mainLearn']['updateEvery']) == 0:
            tqdm.write('+--------------------------------------+')
            tqdm.write('| Updating Buffer ...                  |')
            tqdm.write('+--------------------------------------+')
            agent.updateBuffer(env, brainName)


        # print('---------[Learning]-------------------')
        agent.learn( **config['learnParams'] )


        # This is for checking the current episode
        result = agent.playEpisode(env, brainName)
        tqdm.write(str(result))
        allResults.append(result)
        
    env.close()
    allResults = np.array(allResults)

    plt.plot(allResults[:, 0])
    plt.plot(allResults[:, 1])
    now = dt.now().strftime('%Y-%m-%d--%H-%M-%s')
    plt.savefig(f'../results/img/{now}.png')
    plt.close()


    np.save(f'../results/data-{now}.npy', allResults)

    with open(f'../results/config-{now}.json', 'w') as f:
        f.write(json.dumps(config))
    
    return


if __name__ == '__main__':
    main()

