from lib import memory
import numpy as np 

def collectEpisodes(env, brainName, policy, tMax=200, gamma=1, train_mode=True):
    '''collect one set of 20 episodes form the set of learners
    
    [description]
    
    Parameters
    ----------
    env : {[type]}
        [description]
    brainName : {[type]}
        [description]
    policy : {[type]}
        [description]
    tMax : {number}, optional
        [description] (the default is 200, which [default_description])
    gamma : {number}, optional
        [description] (the default is 1, which [default_description])
    
    Returns
    -------
    [type]
        [description]
    '''

    env_info = env.reset(train_mode=train_mode)[brainName] 
    nAgents = len(env_info.agents)

    episodes = []

    for a in range(nAgents):
        episodes.append(
            memory.Episode(tMax, 4, gamma=gamma))

    states = env_info.vector_observations
    scores = np.zeros(nAgents)

    for t in range(tMax):

        actions    = [policy(states[j]) for j in range(nAgents)]

        envInfo    = env.step(actions)[brainName]
        nextStates = envInfo.vector_observations   # get next state (for each agent)
        rewards    = envInfo.rewards               # get reward (for each agent)
        dones      = envInfo.local_done

        for k, e in enumerate(episodes):
            e.append( 
                states[k], actions[k], rewards[k], 
                nextStates[k], dones[k])

        if any(dones):
            break

        states = nextStates

    return episodes

def updateReplayBuffer(buffer, env, brainName, policy, tMax = 200, gamma=1, numDelete=10):

    # We are expecting a ReplayBuffer over here ...

    buffer.deleteNEpisodes(numDelete)
    episodes = collectEpisodes(env, brainName, policy, tMax=tMax, gamma=gamma)
    for e in episodes:
        buffer.append(e)

    return buffer



