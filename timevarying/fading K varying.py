"""
Note: This is based on Mofan's codes from: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
This code is used to generate the figure for the random fading but also try to average them.
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from enviroment import Env_cellular as env
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################


ct=1000
MAX_EPISODES =20
MAX_EP_STEPS = 100
LR_A = 0.002    # learning rate for actor
LR_C = 0.004    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()


        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s })[0]

    def learn(self):
        indices = np.random.choice(min(MEMORY_CAPACITY,self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        r = np.reshape(r,(1,1))
        a = np.reshape(a,(1,1))
        #print(f"state is {s}, action is {a}, reward is {r}, next state is {s_}")
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            a2 = tf.layers.dense(net, 64, activation=tf.nn.tanh, name='l2', trainable=trainable)
            #a3 = tf.layers.dense(a2, 30, activation=tf.nn.tanh, name='l3', trainable=trainable)

            a = tf.layers.dense(a2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net, 64, activation=tf.nn.relu, name='lx2', trainable=trainable)
            #net3 = tf.layers.dense(net2, 30, activation=tf.nn.relu, name='lx3', trainable=trainable)

            #not sure about this part
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################



s_dim = 4# dimsion of states
a_dim = 1# dimension of action
a_bound = 1 #bound of action
state_am = 10000


#GB_power_dBm = np.array([0,  10,  20,   30])
K_range = np.array([2,4,6,8, 10])
rate_DDPG = []
rate_greedy = []
rate_random = []
for k in range(len(K_range)):
    K = K_range[k]
    ratect_DDPG = 0
    ratect_greedy = 0
    ratect_random = 0
    for mct in range(ct):
        var = 1  # control exploration
        ddpg = DDPG(a_dim, s_dim, a_bound)

        locationspace = np.linspace(1, 1000, num=K)
        location_vector = np.zeros((K, 2))
        location_vector[:, 1] = locationspace
        location_GF = np.array([[1,1]])# np.ones((1, 2))

        Pn = 10**((30-30)/10)

        ratek_DDPG = 0
        ratek_greedy = 0
        ratek_random = 0
        for i in range(MAX_EPISODES):
            ##### fading for GB user
            hnx1 = np.random.randn(K, 2)
            hnx2 = np.random.randn(K, 2)
            fading_n = hnx1 ** 2 + hnx2 ** 2
            #### fading for GF user
            h0x1 = np.random.randn(1, 1)
            h0x2 = np.random.randn(1, 1)
            fading_0 = h0x1[0, 0] ** 2 + h0x2[0, 0] ** 2

            myenv = env(MAX_EP_STEPS, s_dim, location_vector, location_GF, K, Pn, fading_n, fading_0)
            batter_ini = myenv.reset()
            s = myenv.channel_sequence[i % myenv.K, :].tolist()  # the current GB user, 2 element [GB-GF, GB-BS]
            s.append(myenv.h0)
            s.append(batter_ini)
            s = np.reshape(s, (1, s_dim))
            s = s * state_am  # amplify the state
            s_greedy = s
            s_random = s
            reward_dpgg_vector =[]
            reward_greedy_vector = []
            reward_random_vector = []
            #print(s[0,0:2])
            for j in range(MAX_EP_STEPS):

                # Add exploration noise
                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a, var), 0, 1)    # add randomness to action selection for exploration
                #print(myenv.location)
                r, s_, done = myenv.step(a, s / state_am, j)
                s_ = s_ * state_am
                #print(f"{j} state is {s}, action is {a}, reward is {r}, next state is {s_}")
                ddpg.store_transition(s, a, r, s_)
                if var>0.05:
                    var *= .9998  # decay the action randomness
                ddpg.learn()
                s = s_
                reward_dpgg_vector.append(r)

                ##### greedy
                r_greedy, s_next_greedy, done = myenv.step_greedy(s_greedy/state_am, j)
                s_greedy = s_next_greedy*state_am
                reward_greedy_vector.append(r_greedy)

                ##### random
                r_random, s_next_random, done = myenv.step_random(s_random/state_am, j)
                s_random = s_next_random*state_am
                reward_random_vector.append(r_random)



                if j == MAX_EP_STEPS-1:
                    #print(f"Iteration., {k}--{mct},  Episode:, {i},  Reward: {sum(reward_dpgg_vector)/MAX_EP_STEPS},")
                    #      f" Reward Greedy:   {sum(reward_greedy_vector)/MAX_EP_STEPS},"
                    #      f" Reward random:   { sum(reward_random_vector)/MAX_EP_STEPS}, Explore: {var} ")
                    break
            ratek_DDPG = sum(reward_dpgg_vector)/MAX_EP_STEPS
            ratek_greedy = sum(reward_greedy_vector)/MAX_EP_STEPS
            ratek_random = sum(reward_random_vector)/MAX_EP_STEPS

        tf.keras.backend.clear_session()
        ratect_DDPG +=ratek_DDPG
        ratect_greedy +=ratek_greedy
        ratect_random += ratek_random
        print(ratek_DDPG)
    print(f"Iteration., {k}--{mct},  Episode:, {i},  Reward: {ratect_DDPG/ct },")
    rate_DDPG.append(ratect_DDPG/ct )
    rate_greedy.append(ratect_greedy /ct )
    rate_random.append(ratect_random  /ct)

print(f"rate_greedy is {rate_greedy} and rate_random is {rate_random}")
#GB_power_label = ['0',  '10',  '20', '30']
GB_power_label = ['0', '5', '10', '15', '20', '25', '30']
K_rangex =  ['2', '4', '6', '8', '10']
plt.plot(K_rangex,rate_DDPG, "^-", label='DDPG: rewards')
plt.plot(K_rangex,rate_greedy, "+:", label='Greedy: rewards')
plt.plot(K_rangex,rate_random, "o--", label='Random: rewards')
plt.xlabel("The Number of Primary Users (K)")
plt.ylabel("  Average  Data Rate (NPCU)")
plt.legend()
plt.show()


''' Save final results'''
np.savez_compressed('data/dataK', rate_DDPG=rate_DDPG, rate_greedy=rate_greedy, rate_random=rate_random)