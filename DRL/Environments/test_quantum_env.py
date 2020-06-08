# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:38:28 2019

@author: seb
"""

from quan_T4_2d import Quantum_T4_2D
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf


from datetime import datetime
#from packaging import version

# Load the TensorBoard notebook extension.

tf.reset_default_graph()


# -----------------------------------------------------------------------------------
FILE_NAME="T4_scan_data_res_350_win_350"
seb=Quantum_T4_2D(FILE_NAME) # change filename to access different plot
seb.starting_pos=[10,10]
seb.current_pos=[10,10]
f, axarr  = plt.subplots(seb.dim[0],seb.dim[1],figsize=(10,10))
# -----------------------------------------------------------------------------------
# Testing environment

for ii in range(seb.dim[0]):
    for jj in range(seb.dim[1]):
        axarr[ii,jj].imshow(seb.image_smallpatch_data[ii][jj][4])
        axarr[ii,jj].axis('off')
f.show()


print(seb.dim)

# .K action vector
# .D is the input vector (9 windows described by mean and standard deviation of current)
print(seb.K,seb.D)

# Initial location
print("starting positions:",seb.starting_loc)

# Current location
print("current positions:",seb.current_pos)


print("r:",seb.get_reward(seb.current_pos))
#seb.plot_current_state()

# plot the current state
plt.clf()
fig=plt.figure()
myrow,mycol=seb.current_pos
plt.imshow(seb.image_smallpatch_data[myrow][mycol][4])
plt.show()



#0 up 1 down 2 left 3 right 4 top right 5 bot left
seb.step(2)

#seb.plot_current_state()

print("current positions:",seb.current_pos)
print("r:",seb.get_reward(seb.current_pos))

print("starting positions:",seb.starting_loc)

seb.step(0)
print("current positions:",seb.current_pos)
#seb.plot_current_state()


seb.step(1)
print("current positions:",seb.current_pos)
#seb.plot_current_state()

print(seb.possible_actions_from_location())

print("-----------------------------------------")
print(" I am resetting")
print(seb.reset())
print("-----------------------------------------")

print("current positions:",seb.current_pos)
#seb.plot_current_state()
'''
# -----------------------------------------------------------------------------------

#Building NN

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        tf.summary.histogram("loss", loss)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})
        return

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        #self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state,location = self._env.reset_at_rand_loc()
        tot_reward = 0
        max_x = -100
        position_list_x, position_list_y = [], []
        while True:
            #if self._render:
            #    self._env.render()
            x, y = env.current_pos
            position_list_x.append(x)
            position_list_y.append(y)

            action = self._choose_action(state)
            print("---------------------------")
            plt.clf()
            #fig = plt.figure()
            #plt.imshow(env.image_smallpatch_data[x][y][4])
            #plt.show()
            print("Location",location)
            print("Action is being taken.", action)
            next_state, reward, done, location = self._env.step(action)
            
            # is the game complete? If so, set the next state to
            # None for storage sake
            '''if done:
                reward = 1000
                next_state = None
            elif location[0] in position_list_x and location[1] in position_list_y:
                reward = -50
            else:
                reward = -10
            '''
            reward = env.get_reward(env.current_pos)
            print("Reward",reward)
            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                                      * np.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)

                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))
        return position_list_x, position_list_y

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model._num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        GAMMA = 1.0
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)
if __name__ == "__main__":
    FILE_NAME = "T4_scan_data_res_350_win_350"
    seb = Quantum_T4_2D(FILE_NAME)  # change filename to access different plot
    seb.starting_pos = [20, 20]
    seb.current_pos = [20, 20]
    BATCH_SIZE = 100
    MAX_EPSILON = 0.6
    MIN_EPSILON = 0.3
    LAMBDA = 1.0e-5

    env = seb

    num_states = env.D
    num_actions = env.K

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)
        sess.run(model._var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        LAMBDA)
        num_episodes = 1
        cnt = 0

        while cnt < num_episodes:
            if cnt % 2 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                #print("current positions:", env.current_pos)
            merge = tf.summary.merge_all()
            position_list_x, position_list_y = [], []
            position_list_x, position_list_y = gr.run()

            cnt += 1
    plt.plot(gr._reward_store)
    plt.title("Reward store")
    plt.show()
    plt.clf()
    #plt.plot(gr._max_x_store)
    #plt.show()

    print("-----------------------")
    print("-----------------------")
    #print("x, position", position_list_x)
    #print("y, position", position_list_y)


    f, axarr = plt.subplots(env.dim[0], env.dim[1], figsize=(10, 10))
    # -----------------------------------------------------------------------------------
    # Testing environment
    for ii in range(seb.dim[0]):
        for jj in range(seb.dim[1]):
            axarr[ii, jj].imshow(np.zeros_like(seb.image_smallpatch_data[ii][jj][4]))
            axarr[ii, jj].axis('off')

    for ii in range(len(position_list_x)):
        for jj in range(len(position_list_y)):
            if (ii == jj):
                axarr[position_list_x[ii], position_list_y[jj]].imshow(seb.image_smallpatch_data[position_list_x[ii]][position_list_y[jj]][4])
                axarr[position_list_x[ii], position_list_y[jj]].axis('off')

    f.show()

    print("Done")
