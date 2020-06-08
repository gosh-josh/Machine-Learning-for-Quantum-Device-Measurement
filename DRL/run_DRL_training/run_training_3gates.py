# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:35:17 2019

@author: Vu
"""
import sys

sys.path.append('../')
sys.path.append('../../')

sys.path.append('../environments')
sys.path.append('../utilities')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime


#from scipy.misc import imresize
#from gym import wrappers
import random
from tqdm import tqdm
from prioritized_experience_replay import Memory
from vdrl_utilities import HiddenLayer

#from train_cnn import CNNQuantum


#from quan_env_rand_loc_T4_3d import Quantum_T4_3D
from quan_T4_3d import Quantum_T4_3D
from sklearn.model_selection import train_test_split
from time import sleep
import pickle
import gc      

from print_trajectories_policies import print_trajectory_from_location,final_policy_on_test
from play_episodes import play_train_episode, play_test_episode_from_location,burn_in_experience

IM_SIZE = 2 #80
N_CHANEL=27 # this is the representation of a block by 9 blocks
K = 8 #env.action_space.n

import warnings
warnings.filterwarnings("ignore")



class Dueling_DQN_3D:
    def __init__(self, D,K, hidden_layer_sizes, gamma,scope="DDQN",
              batch_sz=16,memory_size=50000,min_y=0,max_y=1,max_experiences=50000, min_experiences=2000):
                 
        with tf.variable_scope(scope):

            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            #self.ISWeights_=tf.placeholder(tf.float32, shape=(None,1), name='ISW')
            self.X = tf.placeholder(tf.float32, shape=(None, N_CHANEL*IM_SIZE), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.neighborMap = tf.placeholder(tf.float32, shape=(None,K), name='neighborMap')
            self.memory_size=memory_size
            self.memory= Memory(memory_size)
    
            self.min_y=min_y
            self.max_y=max_y
            
            self.K=K
            self.D=D
            
            self.learning_rate=2.65e-6
            Z=self.X
            tf.summary.histogram("X", self.X)

    
            for M2 in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z, M2)
            
            tf.summary.histogram("fully_connected_Z", Z)

            #M3=hidden_layer_sizes[1]
            
            print(Z.get_shape())
            tf.summary.histogram("neighborMap", self.neighborMap)

            # neighborMap is 4-dim [top-right-down-left] or 8-dim []
            Z=tf.concat([Z, self.neighborMap], 1)
            tf.summary.histogram("concat_Z", Z)

            print(Z.get_shape())
    
            #Z = tf.contrib.layers.fully_connected(Z, M2)
            
            #tf.summary.histogram("fully_connected_Z2", Z)

      
                
            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs = Z,
                units = 64,  activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value_fc")
    
            tf.summary.histogram("value_fc",self.value_fc)

            self.value = tf.layers.dense(inputs = self.value_fc,
                units = 1, activation = None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value")
            tf.summary.histogram("value",self.value)

            
            # add the visited map here
            
            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs = Z,
                units = 64,  activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantage_fc")
            
            tf.summary.histogram("advantage_fc",self.advantage_fc)

            #self.advantage_fc=tf.concat([self.advantage_fc, self.neighborMap], 1)
            
            #tf.summary.histogram("advantage_fc_concat",self.advantage_fc)

    
            self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                units = K, activation = None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantages")
                
            tf.summary.histogram("advantage",self.advantage)

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage, 
                               tf.reduce_mean(self.advantage, axis=1, keepdims=True))
    
            tf.summary.histogram("output",self.output)

            # Q is our predicted Q value.
            #self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
    
            self.predict_op = self.output
    
            selected_action_values = tf.reduce_sum(
                self.output * tf.one_hot(self.actions, K),
                reduction_indices=[1])
    
            self.loss = tf.reduce_sum(tf.square(self.G - selected_action_values))
            #self.loss = tf.reduce_sum(self.ISWeights_*tf.square(self.G - selected_action_values))
            tf.summary.histogram("G", self.G)
            tf.summary.scalar("loss", self.loss)

    
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            #self.loss = tf.reduce_mean(tf.square(self.G - self.Q))
    
            # The loss is modified because of PER 
            self.absolute_errors = tf.abs(selected_action_values - self.G)# for updating Sumtree
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            #self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)
    
            # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
            # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
            # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
    
            # create replay memory
            #self.experience={'s':[],'a':[],'r':[],'s2':[],'done':[]}
            #self.max_experiences=max_experiences
            #self.min_experiences=min_experiences
            self.batch_sz=batch_sz
            self.gamma=gamma
            self.count_exp=0
            
            self.merge = tf.summary.merge_all()
            
            # create replay memory
            self.experience={'s':[],'a':[],'r':[],'s2':[],'s2_loc':[]
            ,'neighborMaps':[],'neighborMaps_next':[],'env_id':[],'done':[]}
            self.max_experiences=max_experiences
            self.min_experiences=min_experiences
            self.batch_sz=batch_sz
            self.gamma=gamma
    
            
    def set_session(self, session):
        self.session = session
    
    def set_learning_rate(self,lr):
        self.learning_rate=lr
    def predict(self, X, neighborMap):
        X = np.atleast_2d(X)
        
        if len(X.shape)==3:
            X=np.squeeze(X)

        
        neighborMap=np.atleast_2d(neighborMap)
        return self.session.run(self.predict_op, feed_dict={self.X: X, self.neighborMap:neighborMap})
            
    def fit(self,states,targets,actions):
        # call optimizer
        actions=np.atleast_1d(actions)
        targets=np.atleast_1d(targets)

        states=np.atleast_2d(states)

        self.session.run(self.train_op,feed_dict={self.X: states,self.G: targets,self.actions: actions}         )
        
   
    def fit_exp_replay(self,env_list):
        if self.count_exp < self.memory_size:
        # don't do anything if we don't have enough experience
            return None,None

        # Obtain random mini-batch from memory
        #tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_sz)
        
        # experience replay
        idx=np.random.choice(len(self.experience['s']), 
                     size=self.batch_sz, replace=False)
            
        states=[self.experience['s'][i] for i in idx]
        #states_loc=[self.experience['s_loc'][i] for i in idx]
        actions=[self.experience['a'][i] for i in idx]
        rewards=[self.experience['r'][i] for i in idx]
        next_states=[self.experience['s2'][i] for i in idx]
        next_states_loc=[self.experience['s2_loc'][i] for i in idx]
        neighborMaps=[self.experience['neighborMaps'][i] for i in idx]
        neighborMaps_next=[self.experience['neighborMaps_next'][i] for i in idx]

        env_id=[self.experience['env_id'][i] for i in idx]
        dones=[self.experience['done'][i] for i in idx]


        #neighborMaps=[env.get_neighborMap(loc) for loc in states_loc]
        neighborMaps_next=[env_list[myid].get_neighborMap(loc) for loc,myid in zip(next_states_loc,env_id)]
        
        next_actions = np.argmax(self.predict(next_states,neighborMaps_next), axis=1)
        targetQ=self.predict(next_states,neighborMaps_next)
        next_Q=[q[a] for a,q in zip(next_actions,targetQ)]

        targets = [r + self.gamma*next_q 
        if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        #scale target
        #targets=[ (val-self.min_y)/(self.max_y-self.min_y) for val in targets]

        #print(targets)
        summary,loss,_,absolute_errors=self.session.run( [self.merge,self.loss,self.train_op,self.absolute_errors], 
                     feed_dict={self.X:states, self.G:targets, self.actions: actions,
                            self.neighborMap:neighborMaps })

        # Update priority
        #print(absolute_errors)

        #self.memory.batch_update(tree_idx, absolute_errors)
        
        return summary,loss
    

    def add_experience(self, s,loc,a,r,s2,loc_s2,done,count_neighbor,neighborMaps_next,env_id):
        """
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
            self.experience['s2_loc'].pop(0)
            self.experience['neighborMaps'].pop(0)
            self.experience['neighborMaps_next'].pop(0)
            self.experience['env_id'].pop(0)
            
        # Add experience to memory
        if type(r) is list: # adding a list of them
            self.experience['s']+=s
            self.experience['a']+=a
            self.experience['r']+=r
            self.experience['s2']+=s2
            self.experience['done']+=done
            self.experience['s2_loc']+=loc_s2
            self.experience['neighborMaps']+=count_neighbor
            self.experience['neighborMaps_next']+=neighborMaps_next
            self.experience['env_id']+=env_id
        else:
            self.experience['s'].append(s)
            self.experience['a'].append(a)
            self.experience['r'].append(r)
            self.experience['s2'].append(s2)
            self.experience['done'].append(done)
            self.experience['s2_loc'].append(loc_s2)
            self.experience['neighborMaps'].append(count_neighbor)
            self.experience['neighborMaps_next'].append(neighborMaps_next)
            self.experience['env_id'].append(env_id)
        """
    
        if type(r) is list:
            for ii in range(len(s)):
                experience = s[ii], loc[ii], a[ii], r[ii], s2[ii], loc_s2[ii], done[ii],\
                    count_neighbor[ii], env_id[ii]
                self.memory.store(experience)

    def sample_action(self, env, x, loc_x,eps,isNoOverlapping=False,is2Action=False):
        
        # check all posible actions
        #possible_actions=env.possible_actions()
        possible_actions=env.possible_actions_from_location(loc_x)
        
        if isNoOverlapping==True and possible_actions == []: # we got stuck, let jump out
            # find the empty place in the visit_map
            # then find the nearest location
            [idxRow,idxCol]=np.where(env.visit_map==0)
            #idxRow=idxRow.astype(float)
            #idxCol=idxCol.astype(float)
            coordinate=np.array(list(zip(idxRow,idxCol)))
            coordinate=np.reshape(coordinate,(-1,2))
            loc_x=np.asarray(loc_x).reshape((1,2))
            
            dist=[np.linalg.norm(loc_x[0]-coordinate[ii]) for ii in range(coordinate.shape[0])]
            
            idxNearest=np.argmin(dist)
            return coordinate[idxNearest] # we return a new coordinate of the new action
                        
        if is2Action==True:
            neighborMap=env.get_neighborMap(loc_x)
            
            val=self.predict(x,neighborMap)

            val_selected=val[0][possible_actions]
            #print("here",val,val_selected)
            #return np.argmax(self.predict([x])[0])
            
            idx=np.argmax(val_selected)
            
            idx2Best=np.argsort(val_selected)[-2]
            
            if len(possible_actions)==1:
                return possible_actions[idx],val_selected[idx],\
                    possible_actions[idx],val_selected[idx]

            else:
                return possible_actions[idx],val_selected[idx],possible_actions[idx2Best],\
                        val_selected[idx2Best]
        
        
        #print("possible actions",possible_actions)
        if np.random.random() < eps:
            idx=np.random.choice( len(possible_actions))
            return possible_actions[idx] 
        else:
            #X = np.atleast_2d(x)
            neighborMap=env.get_neighborMap(loc_x)
            
            val=self.predict(x,neighborMap)

            val_selected=val[0][possible_actions]
            #print("here",val,val_selected)
            #return np.argmax(self.predict([x])[0])
            
            idx=np.argmax(val_selected)
            return possible_actions[idx]
        
        
    def fit_prioritized_exp_replay(self,env_list):
        if self.count_exp < self.memory_size:
        # don't do anything if we don't have enough experience
            return None,None

        # Obtain random mini-batch from memory
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_sz)
        
        # experience replay
        #idx=np.random.choice(len(self.experience['s']), 
                     #size=self.batch_sz, replace=False)
            
        """
        states=[self.experience['s'][i] for i in idx]
        #states_loc=[self.experience['s_loc'][i] for i in idx]
        actions=[self.experience['a'][i] for i in idx]
        rewards=[self.experience['r'][i] for i in idx]
        next_states=[self.experience['s2'][i] for i in idx]
        next_states_loc=[self.experience['s2_loc'][i] for i in idx]
        neighborMaps=[self.experience['neighborMaps'][i] for i in idx]
        #neighborMaps_next=[self.experience['neighborMaps_next'][i]/20 for i in idx]
        env_id=[self.experience['env_id'][i] for i in idx]
        dones=[self.experience['done'][i] for i in idx]
        """

        states=[each[0][0] for each in batch]
        states_loc=[each[0][1] for each in batch]
        actions=[each[0][2] for each in batch]
        rewards=[each[0][3] for each in batch]
        next_states=[each[0][4] for each in batch]
        next_states_loc=[each[0][5] for each in batch]
        dones=[each[0][6] for each in batch]
        neighborMaps=[each[0][7] for each in batch]
        #neighborMaps_next=[each[0][8] for each in batch]
        env_id=[each[0][8] for each in batch]
        

        #neighborMaps=[env.get_neighborMap(loc) for loc in states_loc]
        neighborMaps_next=[env_list[myid].get_neighborMap(loc) for loc,myid in zip(next_states_loc,env_id)]
        
        next_actions = np.argmax(self.predict(next_states,neighborMaps_next), axis=1)
        targetQ=self.predict(next_states,neighborMaps_next)
        next_Q=[q[a] for a,q in zip(next_actions,targetQ)]

        targets = [r + self.gamma*next_q 
        if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        #scale target
        #targets=[ (val-self.min_y)/(self.max_y-self.min_y) for val in targets]

        #print(targets)
        summary,loss,_,absolute_errors=self.session.run( [self.merge,self.loss,self.train_op,self.absolute_errors], 
                     feed_dict={self.X:states, self.G:targets, self.actions: actions,
                            self.neighborMap:neighborMaps })

        # Update priority
        #print(absolute_errors)

        self.memory.batch_update(tree_idx, absolute_errors)
        
        return summary,loss
    
      

    
np.random.seed(1)
random.seed(1)        
tf.set_random_seed(1)

tf.reset_default_graph()   
   
#env = Quantum_G5G9()

# create multiple environment
starting_pixel_loc_list=[[10,20,30],[10,140,20],[70,20,10],[70,140,140],[70,130,110]
                    ,[70,40,130],[10,140,140],[70,140,10],[20,140,20],[30,30,40],
                    [50,90,135],[70,100,120],[30,40,100],[70,35,120],[30,60,90],
                    [60,30,125],[70,110,30],[60,10,10],[50,90,30],[50,80,140]]


n_env=len(starting_pixel_loc_list)

env_list=[0]*n_env
for ii in range(n_env):
    env_list[ii]= Quantum_T4_3D(starting_pixel_loc=starting_pixel_loc_list[ii])
    env_list[ii].id=ii

D = env_list[0].D
K = env_list[0].K
# (32,8,4): #filter, kernelsize, strides
#conv_layer_sizes = [(32, 8, 4), (64, 4, 2),(32,4,2)]
  

hidden_layer_sizes = [64,64,32,32]
#hidden_layer_sizes=[512]
gamma = 0.5
#batch_sz = 32
num_episodes =30100
total_t = 0
experience_replay_buffer = []
episode_rewards = np.zeros(num_episodes)
myloss = np.zeros(num_episodes)

last_100_avg=np.zeros(num_episodes)
last_100_avg_step=np.zeros(num_episodes)

num_steps=np.zeros(num_episodes)
episode_rewards_Test=[]
num_steps_Test=[]


# epsilon
# decays linearly until 0.1
eps = 1.0
eps_min = 0.1
#epsilon_change = (epsilon - epsilon_min) / 500000
eps_change = (eps - eps_min) / (3*num_episodes)

# Create environment
#cnnmodel = CNNQuantum(K=2,batch_sz=16,max_iter=20,dropout=.4)

# number of random test
batch_sz=32
count=0


model = Dueling_DQN_3D(D=D,K=K,batch_sz=batch_sz,hidden_layer_sizes=hidden_layer_sizes,    
                   gamma=gamma,    scope="DDQN")
  
init = tf.global_variables_initializer()
#sess = tf.InteractiveSession()

gpu_options = tf.GPUOptions(visible_device_list=str(1)) # set a GPU ID for multiple GPU cards
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(init)

model.set_session(sess)

starting_loc_test=starting_pixel_loc_list
nTest=len(starting_loc_test)
# Create models
   
#cnnmodel=train_cnnmodel(cnnmodel)

#debug(cnnmodel)
# Merge all summaries into a single operator
#merged_summary_op = tf.merge_all_summaries()

# Set the logs writer to the folder /tmp/tensorflow_logs
summary_writer = tf.summary.FileWriter('../log/3d', graph=sess.graph)


print("Populating experience replay buffer...")
#state = env_list.reset()
# assert(state.shape == (4, 80, 80))


count_found_target=0
for i in range(5000): # burn in
    c=burn_in_experience(  env_list,  experience_replay_buffer,  model,MaxStep=80)
    count_found_target+=c
    
print("Found Target {:d}/5000".format(count_found_target))
    


# Play a number of episodes and learn!
for i in range(num_episodes):
    
    
    total_t, episode_rewards[i], duration, num_steps[i], time_per_step, eps,myloss[i], summary_writer = play_train_episode(env_list, 
                    total_t,i,experience_replay_buffer,model,gamma,batch_sz, eps,eps_change,eps_min,summary_writer,MaxStep=100)
    

    last_100_avg[i] = episode_rewards[max(0, i - 100):i + 1].mean()  
    last_100_avg_step[i] = num_steps[max(0, i - 100):i + 1].mean()  
    
    if i==7000:
        model.set_learning_rate(lr=2.59e-6)
        
    if i==12000:
        model.set_learning_rate(lr=2.56e-6)
        
    if i==16000:
        model.set_learning_rate(lr=2.54e-6)
        
    if i%500==0:
        print("Epi:", i,"Duration:", duration,"#steps:", num_steps[i],"Reward:", episode_rewards[i],\
        "Train time/step:", "%.3f" % time_per_step,"Avg Reward (Last 100):", "%.3f" % last_100_avg[i], "Eps:", "%.3f" % eps     )
        
        #print(env.isquantum)
        # create another test screnario
        # where we will start at other location (not the middle)
        temp_reward=[0]*nTest
        temp_step=[0]*nTest
        location_state_list_multiple=[0]*nTest
        for jj in range(nTest):
            
            newenv= Quantum_T4_3D(starting_pixel_loc=starting_loc_test[jj])
            #newenv=np.copy(env_list[jj])

            temp_reward[jj], temp_step[jj], visit_map,location_state_list_multiple[jj],newenv = \
            play_test_episode_from_location(newenv,model, eps,MaxStep=200)
        
            #print(newenv.isquantum)
            #print(visit_map)
            #print(newenv.isquantum)

            if i==10000:
                #print_policy_map_flexible_location(newenv,location_state_list_multiple[jj], idx=jj)
                
                #export pickle                            
                strTest="../plot/3d/location_state_list_multiple_{}.pickle".format(jj)
                pickle_out = open(strTest,"wb")
                pickle.dump(location_state_list_multiple, pickle_out)
                pickle_out.close()
            
            #optimal_policy=final_policy_on_test(cnnmodel, model,starting_loc)
            #print(optimal_policy)

        #print("0:up \t 1:right \t 2:down \t 3:left")
        
        episode_rewards_Test.append(temp_reward)
        num_steps_Test.append(temp_step)
        
        print("reward Test:",episode_rewards_Test[-1]," #step Test:",num_steps_Test[-1])
              
              
        count_found_target=0
        for i in range(5000): # burn in
            c=burn_in_experience(  env_list,  experience_replay_buffer,  model,MaxStep=50)
            count_found_target+=c
            
        print("Found Target {:d}/5000".format(count_found_target))


saver = tf.train.Saver()
save_path = saver.save(sess,  "../log/3d/save_models/2d_mean_std")


fig=plt.figure()
plt.plot(np.log(myloss))
plt.title('Log of Training Loss')
plt.xlabel('Episode')
plt.ylabel('Log of Loss')
#fig.savefig("../fig/TrainingLoss_3d.pdf",box_inches="tight")



logloss=np.log(myloss)
ave_logloss=[np.mean(logloss[max(0,i-100):i+1]) for i in range(len(logloss))]
fig=plt.figure()
plt.plot(ave_logloss)
plt.title('Average Training Loss')
plt.xlabel('Episode')
plt.ylabel('Log of Loss')


fig=plt.figure()
plt.plot(episode_rewards)
plt.title('Training Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
#fig.savefig("../fig/TrainingReward_3d.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(last_100_avg)
plt.title('Training Average Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
#fig.savefig("fig/TrainingReward_Ave_3d.pdf",box_inches="tight")


fig=plt.figure()
plt.plot(last_100_avg[1000:])
plt.title('Training Average Reward from 1000...')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
#fig.savefig("fig/TrainingReward_Ave1000_3d.pdf",box_inches="tight")


fig=plt.figure()
plt.plot(num_steps)
plt.title('Number of Training Steps')
plt.xlabel('Episode')
plt.ylabel('Step')
#fig.savefig("fig/TrainingStep_3d.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(last_100_avg_step)
plt.title('Average Training Steps')
plt.xlabel('Episode')
plt.ylabel('Average Step')
#fig.savefig("fig/TrainingAveStep_3d.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(episode_rewards_Test)
plt.title('Average Reward Test')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
#fig.savefig("fig/TestAveReward_3d.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(num_steps_Test)
plt.title('Average Test Steps')
plt.xlabel('Episode')
plt.ylabel('Average Step')
#fig.savefig("fig/TestAveStep_3d.pdf",box_inches="tight")


ave_step=np.asarray(num_steps_Test)
ave_step=np.mean(ave_step,axis=1)
fig=plt.figure()
plt.plot(ave_step)
plt.title('Average Test Steps')
plt.xlabel('Episode')
plt.ylabel('Average Step')


output=[myloss,episode_rewards,last_100_avg,num_steps,last_100_avg_step
    ,num_steps_Test,episode_rewards_Test ]
pickle.dump( output, open( "results/result_T4_3d.p", "wb" ) )