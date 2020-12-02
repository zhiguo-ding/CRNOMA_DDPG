# -*- coding: utf-8 -*-
"""
Modified version based on the code by mengxiaomao for the paper
https://arxiv.org/abs/1901.07159
"""
import scipy
from scipy import special
import numpy as np
from scipy.special import lambertw
import math

dtype = np.float32

class Env_cellular():
    def __init__(self,    MAX_EP_STEPS, s_dim, location_vector, location_GF,K, Pn,fading_n, fading_0):
        self.emax = 0.1  # battery capacity
        #self.P0 = P0
        self.K = K
        self.T =1
        self.eta =0.7
        self.Pn = Pn # grant-based user's power
        self.Pmax = 0.1


        BW = 10**6 # 10MHz
        sigma2_dbm = -170 + 10 * np.log10(BW) #  Thermal noise in dBm
        #sigma2_dbm = -94
        self.noise = 10 ** ((sigma2_dbm - 30) / 10)
        self.Pn = self.Pn

        #self.P0n = 10 #grant free user's power
        self.s_dim = s_dim

        self.MAX_EP_STEPS = MAX_EP_STEPS


        distance_GF = np.sqrt(np.sum((location_vector-location_GF)**2, axis=1))
        distance_GB = np.sqrt(np.sum((location_vector)**2, axis=1))

        distance = np.matrix.transpose(np.array([distance_GF, distance_GB]))
        distance = np.maximum(distance, np.ones((self.K,2)))
        PL_alpha = 3
        PL = 1/distance**PL_alpha/(10**3.17)
        self.hn = np.multiply( PL, fading_n)
        distance_GF0 = np.sqrt(np.sum( location_GF ** 2, axis=1))
        distance0 = np.maximum(distance_GF0, 1)
        PL0 = 1/(distance0 ** PL_alpha)/(10**3.17)
        self.h0 = fading_0*PL0
        #print(f"{self.hn} and {self.h0}")

        self.channel_sequence = np.zeros(( self.MAX_EP_STEPS,2))
        for i in range(self.MAX_EP_STEPS):
            id_index = i % self.K
            self.channel_sequence[i,:] = self.hn[id_index,:]

    def step(self, action, state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise #state[0,2]/self.noise
        En = state[0,-1]


        #print(uchannel)

        En_bar = action*min(self.emax-En, self.T*self.eta*self.Pn*hn0) - (1-action)*min(En,self.T*self.Pmax)

        #print(f"energy is {self.T*self.eta*self.Pn*hn0}")
        #print(f"en_bar is {En_bar}, and action is {action}")
        mu1 = self.eta*self.Pn*hn0*h0/(1+self.Pn*hn)
        mu2 = En_bar*h0/self.T/(1+self.Pn*hn)
        wx0 = np.real(lambertw(math.exp(-1)*(mu1-1), k=0))
        #print(f"{np.exp(wx0+1) } ")
        alphaxx = (mu1-mu2)/(math.exp(wx0+1) - 1 + mu1)
        alpha01 = 1 - (En+En_bar)/(self.T*self.eta*self.Pn*hn0)
        alpha02 = (self.T*self.eta*self.Pn*hn0-En_bar)\
                  /(self.T*self.eta*self.Pn*hn0 + self.T*self.Pmax)
        alphax2 = max(alpha01, alpha02)
        alphan = min(1, max(alphaxx,alphax2))
        #print(alphan)
        if En_bar == self.T * self.eta * self.Pn * hn0:
            P0n=0
            reward = 0  # remark in the paper
        elif alphan ==0:#<= 0.00000001:
            P0n=0
            reward = 0
        else:
            P0n = (1-alphan)*self.eta*self.Pn*hn0/alphan - En_bar/alphan/self.T
            #print(f"power is {P0n}, en is {En}, enbar is {En_bar}, alpha is {alphan}")
            #print(f"power {P0n} divided by noise {self.noise} as {P0n/self.noise}")
            reward = alphan*np.log(1 + P0n*h0/(1 +self.Pn*hn))

        #print(f"power is {P0n}, en is {En}, enbar is {En_bar}, alpha is {alphan}")
        #print(f"signal power is {P0n*h0} and inter is {self.Pn*hn}")
        #print(f"noise is {self.noise}")
        #print(f"signal strentch={P0n} times {h0},   interference is {self.Pn * hn}")
        #print(f"Pn is {self.Pn} and hn is {hn}")
        #print(f"rate is {np.log(1 + P0n * h0 / (1 + self.Pn * hn))}")
        #print(f"signal strentch={P0n} times {h0}, noise is {self.noise}, interference is {self.Pn*hn}")
        #print(f"{j} iteration - reward is {reward}")
        #print(f"get {(1-alphan)*self.T * self.eta * self.Pn * hn0} and consumed {alphan*self.T*P0n}")
        #print(f"get {(1-alphan)}* {self.T} * {self.eta} * {self.Pn} * {hn0} ")

        #print(f"the first part of power{(1 - alphan)}*{self.eta * self.Pn * hn0 / alphan}")
        #print(f"the second part of power{- En_bar/alphan/self.T}")

        if math.isnan(reward):
            print(f"start---------------------------------------")
            print(f"action is{action} ")
            print(f"finallly alphan is {alphan}")
            print(f"end-----------------------------------")
            reward = 0
        #if reward <0:
        #    print(f"alpha is {alphan} and power is {P0n}, reward is {reward}, "
        #          f"bound is {1 - (En_bar)/(self.T*self.eta*self.Pn*hn0)}, enbar is {En_bar}, exist battery is {En}")


        batter_new = min(self.emax, En + En_bar)

        #print(f"{j} - Energy:  {En} minus {(alphan)*self.T*P0n} plus {(1-alphan)*self.T*self.eta*self.Pn*hn0}, remain {batter_new}, enbar is {En_bar} ")
        #print(f"Parameters are - alpha {alphan}, power is {P0n},   GB-GF channel is {hn0},    reward is {reward}")

        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        #state_next.append(self.h0)
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))

        done=False

        return reward, state_next, done

    def step_greedy(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise
        En = state[0,-1]
        #print(uchannel)

        alphan = min(1, En/self.T/self.Pmax)
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*h0/(1 +self.Pn*hn))
            #print(f"random alpha is {alphan}- signal strentch={self.Pmax} times {h0}, noise is {self.noise}, interference is {self.Pn*hn}")
            #print(f"{j}-iteration - energy is {En}-random reward is {reward}")


        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        batter_new = min(self.emax, En -alphan*self.T*self.Pmax +(1-alphan)*self.T*self.eta*self.Pn*hn0)

        #print(f"{j} random - Energy:  {En} minus {(alphan)*self.T*self.Pmax} plus {(1-alphan)*self.T*self.eta*self.Pn*hn0}, remain {batter_new} ")
        #print(f"Parameters are - alpha {alphan}, power is {self.Pmax},   GB-GF channel is {hn0},    reward is {reward}")

        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        #state_next.append(self.h0)
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))

        done=False

        return reward, state_next, done

    def step_random(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise
        En = state[0,-1]
        #print(uchannel)

        alphan = np.random.uniform(0, min(1, En/self.T/self.Pmax))
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*h0/(1 +self.Pn*hn))
            #print(f"random alpha is {alphan}- signal strentch={self.Pmax} times {h0}, noise is {self.noise}, interference is {self.Pn*hn}")
            #print(f"{j}-iteration - energy is {En}-random reward is {reward}")


        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        batter_new = min(self.emax, En -alphan*self.T*self.Pmax +(1-alphan)*self.T*self.eta*self.Pn*hn0)

        #print(f"{j} random - Energy:  {En} minus {(alphan)*self.T*self.Pmax} plus {(1-alphan)*self.T*self.eta*self.Pn*hn0}, remain {batter_new} ")
        #print(f"Parameters are - alpha {alphan}, power is {self.Pmax},   GB-GF channel is {hn0},    reward is {reward}")

        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        #state_next.append(self.h0)
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))

        done=False

        return reward, state_next, done

    def reset(self):
        batter_ini = self.emax
        return batter_ini

