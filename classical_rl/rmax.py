from email import policy
from statistics import quantiles
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
    env = gym.make('FrozenLake-v0',desc=None,map_name=map_name)
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

def rmax(data,frequency, nStates, nActions):
    LData=data
    for s in range(nStates+1):
        for a in range(nActions):
            total_occur=0
            total_reward=0
            for next in range(nStates):
                info = data[s][a][next]
                total_occur+=info[0]
                total_reward+=info[1]
            for next in range(nStates):
                info = data[s][a][next]
                if(total_reward>0):
                    avg_reward=info[1]/total_reward
                else:
                    avg_reward=0
                LData[s][a][next]=[info[0]/total_occur, avg_reward]
            LData[s][a][nStates]=[0, 0]
            if frequency[s, a]<10:
                # clear out all the counts to other states
                # only learn the ones we've got plenty data
                for next in range(nStates):
                    data[s][a][next]=[0, 0]
                LData[s][a][nStates]=[1, 1]
            if s==nStates:
                LData[s][a][nStates]=[1,0]
    return LData

def BellmanBackup(env, s, V, LData):
    actionValue=np.zeros(env.nA)
    for a in range(env.nA):
        for next in range(env.nS+1):
            curr=LData[s][a][next]
            transp=curr[0]
            reward=curr[1]
            actionValue[a]+=transp*(reward+V[next])
    #Check if we are done
    return actionValue


def ValueIteration(env, LData):
    Q=np.zeros((env.nS+1, env.nA))
    V=np.zeros(env.nS+1)
    policy=np.zeros(env.nS+1)
    theta=0.001
    steps_MAX=7000
    nruns=1
    diff=float("inf") #difference of value iterations V_n-V_n+1
    while(steps_MAX>0 and diff>theta):
        steps_MAX-=1
        nruns+=1
        for state in range(env.nS):
            ActionValue=BellmanBackup(env, state, V, LData)
            bestActionValue=np.max(ActionValue)
            Q[state, :]=ActionValue
            diff=max(diff, np.abs(bestActionValue-V[state])) #supnorm
            V[state]=bestActionValue
            policy[state]=np.argmax(ActionValue)
    return Q, V, policy

def epsilon(t, total):
    return 0.05+(1-t/total)*0.95

def mrl(max_eps, env, max_steps):
    #initialize Q, fitting in randomized values
    Q=np.ones((env.nS+1, env.nA))
    #initialize the states, using optmistic initialization
    s=env.reset()
    avg=GetAvgReward(env, s)
    Q=avg*Q
    #quality
    itr=np.arange(0, max_eps+1, 20)
    quality=np.zeros(max_eps//20+1)
    #build the raw data... plus the special state, env.nS
    frequency=np.zeros((env.nS+1, env.nA), dtype=int)
    data={s: {a: {next: [] for next in range(env.nS+1)} for a in range(env.nA)} for s in range(env.nS+1)}
    for s in range(env.nS+1):
        for a in range(env.nA):
            for next in range(env.nS+1):
                data[s][a][next]=[0, 0] #visit times, rewards
    rmax_data=data
    for t in range(max_eps):
        if t%20==0 and t>0:
            rmax_data=rmax(data,frequency, env.nS, env.nA)
            if t%100==0:
                print("Q is ", Q)
                check(env, rmax_data)
            Q, V, policy=ValueIteration(env, rmax_data)
            quality[t//20]=Evaluation(env, policy)
            print("The result for ", t//20, " run is ", quality[t//20])
        # For every episode, train the Q function
        if t>0:
            s=env.reset()
        #Get action according to epsilon greedy
        #check online
        a=eGreedy(epsilon(t, max_eps), Q, s, env)
        for _ in range(max_steps):
            frequency[s][a]+=1
            #into next step
            nextS, r, done, _=env.step(a) 
            #Update the count
            curr=data[s][a][nextS]
            data[s][a][nextS]=[curr[0]+1, curr[1]+r] 
            s=nextS #update the current step
            a=eGreedy(epsilon(t, max_eps), Q, s, env)
            #encounter sink state
            if done:
                #update every actions, with pseudocounts 10, to the same sink state, however with same reward as before.
                # The code here is hedious
                for a in range(env.nA):
                    curr=data[s][a][s]
                    data[s][a][s]=[curr[0]+1, curr[1]]
                for s in range(env.nS+1):
                    for a in range(env.nA):
                        isMLE=False
                        for next in range(env.nS+1):
                            if data[s][a][next]==[0,0]:
                                data[s][a][next]=[1, 0]
                                isMLE=True
                        if isMLE:
                            for s in range(env.nS+1):
                                for a in range(env.nA):
                                    for next in range(env.nS+1):
                                        info=data[s][a][next]
                                        data[s][a][next]=[info[0]+1, info[1]] 
                break
    reward_mean=np.zeros(max_eps//20+1)
    reward_std=np.zeros(max_eps//20+1)
    for t in range(max_eps//20+1):
        reward_mean[t]=np.mean(quality[:t+1])
        reward_std[t]=np.std(quality[:t+1])/2
    plt.title("rmax planning")
    plt.xlabel("alpha")
    plt.ylabel("mean of Reward")
    plt.errorbar(itr, reward_mean, yerr=reward_std, fmt='-o')
    plt.show()

#testing the learning process...
def check(env, data):
    for s in range(env.nS):
        print("State: ", s)
        for a in range(env.nA):
            for next in range(env.nS+1):
                if data[s][a][next][0]>0.1:
                    print("Action: ", a, "Next: ", next, "pr:", data[s][a][next][0])
                    

def QEvaluation(env, Q):
    policy=np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        policy[s]=np.argmax(Q[s, :])
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

def Evaluation(env, policy):
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
# run demo

env,dim,nA,frozenPhi4,dname = prepFrozen4()
mrl(1500, env, 2501000500)
#print(env.P)