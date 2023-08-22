import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from numba import jit
from cmath import exp
from sys import flags
from numpy import linalg
from numpy import random

verbose=1
discount=1.0

#-----------------------------------------------------------

def prepFrozen(map_name=None):
    np.random.seed(5467)
    env = gym.make('FrozenLake-v1',desc=None,map_name=map_name)
    nA=env.nA
    nS=env.nS
    env.seed(48304)
    return(env,nS,nA)
    

def identity(x):
    return x

def prepFrozen4():
    dname="Frozen4"
    env,nS,nA=prepFrozen(map_name='4x4')
    mymaxsteps=250
    env._max_episode_steps = mymaxsteps
    return(env,nS,nA,identity,dname)

def prepFrozen8():
    dname="Frozen8"
    env,nS,nA=prepFrozen()
    mymaxsteps=1000
    env._max_episode_steps = mymaxsteps
    return(env,nS,nA,identity,dname)

# Cart Pole
# Observation: 
#        Type: Box(4)
#        Num	Observation                 Min         Max
#        0	Cart Position             -4.8            4.8
#        1	Cart Velocity             -Inf            Inf
#        2	Pole Angle                 -24 deg        24 deg
#        3	Pole Velocity At Tip      -Inf            Inf
        
#    Actions:
#        Type: Discrete(2)
#        Num	Action
#        0	Push cart to the left
#        1	Push cart to the right

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def discCP(state):
    x, x_dot, theta, theta_dot = state
    d=np.zeros(4,dtype=int)
    d[0]=np.floor((x+4.799)/0.96)
    d[1]=np.floor(sigmoid(x_dot)*9.99)
    d[2]=np.floor((theta+23.99)/4.8)
    d[3]=np.floor(sigmoid(theta_dot)*9.99)
    return d[0]+10*d[1]+100*d[2]+1000*d[3]
    
def prepCartPole(map_name=None):
    np.random.seed(5467)
    env = gym.make('CartPole-v0')
    dname="CartPole"
    env.seed(48304)
    nA=2
    nS=10000
    mymaxsteps=500
    env._max_episode_steps = mymaxsteps
    return(env,nS,nA,discCP,dname)

#-----------------------------------------------------------

# stub for your code
# replace demo with sarsa, model-based etc

def demo(env,nS,nA,phi,dname,render=False):
    print("Demo environment ",dname)
    s = phi(env.reset())
    for t in range(env._max_episode_steps):
        if render:
            env.render()
            time.sleep(0.3)
        action = env.action_space.sample()
        s, reward, done, info = env.step(action)
        s2=phi(s)
        #print("Now in discretized state ",s,s2)
        print("Now in discretized state ",s2)
        if done:
            break

def demoenv(prepFunction,render=False):
    env,nS,nA,phi,dname=prepFunction()
    demo(env,nS,nA,phi,dname,render=render)

# SARSA implementation 
#random argmax: break ties randomly
def rargmax(vec):
    max=np.amax(vec)
    maxargs=np.array([])
    for i, v in enumerate(vec):
        if v>=max:
            maxargs=np.append(maxargs, i)
    return int(np.random.choice(maxargs))

def eGreedy(eps, Q, s, env):
    coin=np.random.uniform(0, 1)<eps
    if coin:
        #randomly choose an action
        return env.action_space.sample() 
    else:
        return rargmax(Q[s, :])

def GetAvgReward(env, s):
    total=0
    for a in range(env.nA):
        for p, ns, r, done in env.P[s][a]:
            total+=p*r
    return total/env.nA

def SARSA(max_eps, env, max_steps, trained_Q=np.ones((1, 1)), trained_policy=np.ones(1), AlreadyTrained=False):
    #initialize Q, fitting in randomized values
    Q=np.ones((env.nS, env.nA))
    #initialize the states, using optmistic initialization
    s=env.reset()
    avg=GetAvgReward(env, s)
    Q=avg*Q
    #if we already trained, then set to the prviously trained value
    #good for incrementally training
    if AlreadyTrained:
        Q=trained_Q
        print("Already trained!", Q)
        policy=trained_policy
    else:
        policy=np.zeros(env.nS, dtype=int)
    #print("initialization", Q)
    #initiaize the policy...
    policy_prev=policy
    Q_prev=Q
    #start the loop
    for t in range(max_eps):
        policy_prev=policy
        Q_prev=Q
        # For every episode, train the Q function
        if t>0:
            s=env.reset()
        #Get action according to epsilon greedy
        a=policy[s]
        #check online
        #a=eGreedy(0.01, Q, s, env)
        for _ in range(max_steps):
            #into next step
            nextS, r, done, info=env.step(a) 
            delta=0
            if done:
                delta=r-Q[s, a]
                Q[s, a]=Q[s, a]+0.01*delta  
            else:
                #decide what is epsilon, in the present situation, we will fix on 0.6
                #nextA=eGreedy(0.06, Q, nextS, env)
                nextA=policy[nextS]
                delta=r+Q[nextS, nextA]-Q[s, a]
                Q[s, a]=Q[s, a]+0.01*delta  
                a=nextA
                s=nextS
            #terminate for sinking states...
            if done:
                break
        #Update the policy according to Q by eGreedy
        for s in range(env.nS):
            policy[s]=eGreedy(0.04, Q, s, env)
        #saving time, if we reach an optimal point, break
        # Some issues: might be saddle point
        if (np.array_equal(policy_prev, policy) and t>=max_eps) and np.array_equal(Q_prev, Q):
            break
    #returns the rewards for this episode and its policy
    return policy, Q

#use probablistic greedy policy according to Q
def GreedyEval(env, max_steps, max_episodes, trained_Q=np.ones((1, 1)), trained_policy=np.ones(1), AlreadyTrained=False):
    Q=np.ones((env.nS, env.nA))
    policy=np.ones(env.nS, dtype=int)
    if AlreadyTrained:
        policy, Q=SARSA(max_episodes, env, max_steps, trained_Q, trained_policy, AlreadyTrained)
    else:
        print("NOT")
        policy, Q=SARSA(max_episodes, env, max_steps)
   # policy=np.zeros(env.nS)
    #for s in range(env.nS):
    #   policy[s]=np.argmax(Q[s])\
    #print(Q)
    for i in range(env.nS):
        policy[i]= np.argmax(Q[i, :])
    return policy, Q

def QEvaluation(env, policy):
    currObs=env.reset()
    total_reward=0
    action=policy[currObs]
    for t in range(2501000500):
        nextObs, reward, done, info = env.step(action)
        action=int(policy[nextObs])
        total_reward+=reward
        if done:
            break
    return total_reward

def Plot(env):
    reward_mean=np.zeros(10)
    reward_std=np.zeros(10)
    Alpha=range(10000, 110000, 10000)
    itr=0
    nRuns=Alpha[0]
    Q=np.ones((env.nS, env.nA))
    policy=np.zeros(env.nS, dtype=int)
    for alpha in Alpha:
        rounds=0
        MAXROUNDS=500
        reward=np.zeros(MAXROUNDS)
        if alpha-nRuns>0:
            policy, Q=GreedyEval(env, 2501000500, alpha-nRuns, Q,policy,True)
        else: # when
            policy, Q=GreedyEval(env, 2501000500, alpha) 
        nRuns=alpha
        print("The policy is ", policy)
        while(rounds<MAXROUNDS):
            reward[rounds]=QEvaluation(env, policy)
            rounds+=1
        reward_mean[itr]=np.mean(reward)
        reward_std[itr]=np.std(reward)
        print("The mean and std for ", alpha, " is ",reward_mean[itr], reward_std[itr])
        itr+=1
    plt.title("SARSA planning")
    plt.xlabel("alpha")
    plt.ylabel("mean of Reward")
    plt.errorbar(Alpha, reward_mean, yerr=reward_std, fmt='-o')
    plt.show()


# run demo

env,dim,nA,frozenPhi4,dname = prepFrozen8()
Plot(env)