# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:36:00 2019

@author: Vu
"""


import sys

sys.path.append('../')
sys.path.append('E:/Dropbox/03.Research/09.QuantumProjects/Pygor')
import Pygor

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from tf_cnn_classification_quantum_images import CNNQuantum
import math
import itertools
from tqdm import tqdm
import gc      

reward_table=[]


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
    
class OnDevice_2D_Norepeat: # 2D
    # in the test environment, we start at a particular location
    # then, we start from there, we can not pre-define the blocks.
    
    def possible_actions(self):
                        
        irow,icol=self.current_pos
        
        possible_actions=[]
        
        if irow>0 and self.visit_map[irow-1,icol]==0: #up
            possible_actions.append(0)
        
        if irow<self.dim[0]-1 and self.visit_map[irow+1,icol]==0: #down
            possible_actions.append(1)
        if icol>0 and self.visit_map[irow,icol-1]==0: #left
            possible_actions.append(2)
        if icol<self.dim[1]-1 and self.visit_map[irow,icol+1]==0: # right
            possible_actions.append(3)
            
        return possible_actions
    
    def possible_actions_from_location(self,location):
        
        irow,icol=location
        possible_actions=[]

        if irow>0 and self.visit_map[irow-1,icol]==0: #up
            possible_actions.append(0)
        
        if irow<self.dim[0]-1 and self.visit_map[irow+1,icol]==0: #down
            possible_actions.append(1)
        if icol>0 and self.visit_map[irow,icol-1]==0: #left
            possible_actions.append(2)
        if icol<self.dim[1]-1 and self.visit_map[irow,icol+1]==0: # right
            possible_actions.append(3)
            
        return possible_actions
    
   
        
        
    def __init__(self, name="T4",bh=28,bw=28):
        # img_row_idx and img_col_idx are the pixel indices, not the block indices
        
        defaults_gate=[147.79,-1458.17628565,  -283.97105228,  -622.40207719, -1871.07883712,\
       -276.39614676,  -679.4902697 , -1113.81260623]
        
        #-962.40868202,  -822.30385845,  -802.15568511,  -142.18028428, -1747.07073912,  -742.81312161, -1729.35970301
        #  -861.71015786,  -192.0956257 ,  -430.84084329, -1282.81330064, -812.66953594, -1801.0843591 ,  -738.3556092 ,  -306.16635879
       
        #initial_gate_c8_c12=[ -718.55425722, -679.4902697]
        #initial_gate_c8_c12=[-652.40207719,-699.4902697]
    
        
        self.bw=bw #block width
        self.bh=bh #block height
        
        if name=="T4":
            defaults_gate=[30, -1150.9539658 ,  -808.79439154,  -620., -1815.3845539 , 
                           -1127.79332466, -1406.53212482,  -943.5,  -682.98769082]
            
            self.pygor=Pygor.Experiment(mode='none', xmlip="http://129.67.86.107:8000/RPC2")
            gates=['c1','c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']

            self.pygor.setvals(gates,defaults_gate)
            initial_gate_cu_cv=[ defaults_gate[3], defaults_gate[7]]

        elif name=="B2":
            defaults_gate=[147.79,-1103.16825894, -1377.40951456,  -718.55425722, \
                       -950.07426596, -1003.78814508,  -651.385735  , -1934.87693926]
            
            self.pygor=Pygor.Experiment(mode='none', xmlip= "http://129.67.85.235:8000/RPC2")
            
            gates=['c1','c3', 'c4', 'c8', 'c10', 'c11', 'c12', 'c16']
            #gates = ["c3","c4","c5","c6","c7","c8","c9","c10"]
            self.pygor.setvals(gates,defaults_gate)
            
            # c8 vs c12
            initial_gate_cu_cv=[ defaults_gate[3], defaults_gate[6]]

        
        else:
            print("please specify the device correctly")
        #self.device1 = Pygor.Device(pygor)
        
        #self.pygor.setval('call',0)
        
        #gates=['c1','c3', 'c4', 'c8', 'c10', 'c11', 'c12', 'c16']
        #gates = ["c3","c4","c5","c6","c7","c8","c9","c10"]
        #self.pygor.setvals(gates,defaults_gate)

        # scan big region
        
        if 0:
            temp = self.pygor.do2d("c8",initial_gate_cu_cv[0]-100,initial_gate_cu_cv[0]+100,20,
                           "c12",initial_gate_cu_cv[1]-100,initial_gate_cu_cv[1]+100,20)
            
            self.bigregion=temp.data[0]
            return
        
        
        
        #gates = ["c8","c12"]
        #self.pygor.setvals(gates,initial_gate_c8_c12)

        
        # this needs to be checked
        #self.dd_dist=[0.02, 0.05] # mean and variance of golden observation
        
        self.barrier_dist=[0.01,0.005]
        self.sd_dist=[0.02,0.2]
        
        self.dd_dist=[0.008, 0.1]
            
        #idxNegative=np.where(self.image<0)
        #self.image[idxNegative]=self.image[idxNegative]*(-1)
        
        # for normalizing purpose           
        self.minMu,self.maxMu,self.minSig,self.maxSig=0,0.9,0,0.39
        
        self.minCurrent=1.0637832448105646e-13
        self.maxCurrent=1.0428198307779363e-09

        #self.offset=-1.8862508749766253e-12
        
        # action space
        self.K=6
 
        # create an empty visit_map of [50x50]
        self.visit_map=np.zeros((50,50))
        self.isquantum=np.zeros((50,50))
        
        # start at the middle
        self.current_pos=np.copy([25,25])
        self.starting_pos=[25,25]
        self.initial_voltage=initial_gate_cu_cv # initial gate voltage of C8 and C12

        # base on this location, construct the data
        self.data=[0]*50
        self.data_ori=[0]*50

        self.data_img=[0]*50
        self.data_img_ori=[0]*50

        for ii in range(50):
            self.data[ii]=[0]*50
            self.data_ori[ii]=[0]*50
            self.data_img[ii]=[0]*50
            self.data_img_ori[ii]=[0]*50
            
        #self.data=np.copy(self.bigpatch_data)
        
        data,img=self.get_data(self.current_pos,pre_pos=None)
        self.data[25][25]=data
        self.data_img_ori[25][25]=np.copy(img)
        self.data_img[25][25]=(img-self.minCurrent)/(self.maxCurrent-self.minCurrent)
        
        #D is the dimension of each patch
        # self.dim is the dimension of blocks in the images
        self.D=len(self.data[25][25])
        
    
    def get_data(self,positions,pre_pos=None):
        # positions and previous positions
        
        res_w=int(self.bw*1)
        res_h=int(self.bh*1)
        
        if pre_pos is not None:
            pre_id1,pre_id2=pre_pos
            
        id1,id2=positions  
        
        if id1<=self.starting_pos[0]: # decrease C8
            c8_val=self.initial_voltage[0]-self.bh*(self.starting_pos[0]-id1)
        else:
            c8_val=self.initial_voltage[0]+self.bh*(id1-self.starting_pos[0])
        
        if id2<=self.starting_pos[1]: # decrease C12
            c12_val=self.initial_voltage[1]-self.bw*(self.starting_pos[1]-id2)
        else:
            c12_val=self.initial_voltage[1]+self.bw*(id2-self.starting_pos[1])
        
        if pre_pos is None: # full scan
            minc8=int(c8_val-self.bw)
            maxc8=int(c8_val+self.bw)
            minc12=int(c12_val-self.bh)
            maxc12=int(c12_val+self.bh)
            
            temp = self.pygor.do2d("c8",minc8,maxc8,2*res_w,
                                    "c12",minc12,maxc12,2*res_h)
            data_2d=temp.data[0]
        else:
            
            if id1<=self.starting_pos[0]: # decrease C8
                old_data=self.data_img_ori[pre_id1][pre_id2][:,:self.bh]
                
                # measure half
                new_data = self.pygor.do2d("c8",c8_val-self.bh,c8_val,1*res_h,
                                        "c12",c12_val-self.bw,c12_val+self.bw,2*res_w)
                
                # combine
                new_data=new_data.data[0]
                data_2d=np.hstack((new_data,old_data))
                
            else: # increase C8
                old_data=self.data_img_ori[pre_id1][pre_id2][:,self.bh:]
                new_data = self.pygor.do2d("c8",c8_val,c8_val+self.bw,1*res_w,
                                        "c12",c12_val-self.bh,c12_val+self.bh,2*res_h)
                
                new_data=new_data.data[0]

                data_2d=np.hstack((old_data,new_data))
                
                
            if id2<=self.starting_pos[1]: # decrease C12
                old_data=self.data_img_ori[pre_id1][pre_id2][:self.bw,:]
                
                # measure half
                new_data = self.pygor.do2d("c8",c8_val-self.bh,c8_val+self.bh,2*res_h,
                                        "c12",c12_val-self.bw,c12_val,1*res_w)
                
                # combine
                new_data=new_data.data[0]

                data_2d=np.vstack((new_data,old_data))
            else:
                old_data=self.data_img_ori[pre_id1][pre_id2][self.bw:,:]
                
                # measure half
                new_data = self.pygor.do2d("c8",c8_val-self.bh,c8_val+self.bh,2*self.bh,
                                        "c12",c12_val,c12_val+self.bw,1*self.bw)
                
                # combine
                new_data=new_data.data[0]

                data_2d=np.vstack((old_data,new_data))

            
        #data_2d = self.pygor.do2d("c8",c8_val-self.bw,c8_val+self.bw,2*self.bw,
                                    #"c12",c12_val-self.bh,c12_val+self.bh,2*self.bh)
        
        # clip to zero
        #data_2d=data_2d+self.offset
        idx = np.where(data_2d<0)
        data_2d[idx]=0
        
        # normalize data_2d: 0->1
        
        if np.min(data_2d)<self.minCurrent:
            print("np.min(data_2d)<self.minCurrent",np.min(data_2d))
            self.minCurrent=np.min(data_2d)
            
        if np.max(data_2d)>self.maxCurrent:
            print("np.max(data_2d)>self.maxCurrent",np.max(data_2d))
            self.maxCurrent=np.max(data_2d)
            
        #data_2d=(data_2d-self.minCurrent)/(self.maxCurrent-self.minCurrent)
        
        # constructing the block data to get 9 block
        size_patch=[2*self.bh,2*self.bw]
        
        # find p1,p2,p3 equally between l1,l2,l3
        mypp=[0]*3
        pp_block=[0]*2 # 2D
        for dd in range(2): # number of dimension
            mypp[dd]=[0]*5
            temp=np.linspace(0, size_patch[dd], num=5)
            temp=temp.astype(int)
            mypp[dd]=temp.tolist()
            
            pp_block[dd]=[[mypp[dd][0],mypp[dd][2]],[mypp[dd][1],mypp[dd][3]],
                     [mypp[dd][2],mypp[dd][4]]]
            
        # based on p1,p2,p3, we split into 3^3 blocks
        pp_blocks=list(itertools.product(*pp_block))
        # 27 elements
        
        data=[0]*len(pp_blocks)
        data_ori=[0]*len(pp_blocks)
        #temp_patch=[]
        #temp_patch_ori=[]
        for kk,mypp in enumerate(pp_blocks): # 27 items
            #print(kk)
            temp=data_2d[mypp[0][0]:mypp[0][1],
                       mypp[1][0]:mypp[1][1]]

            mymean=np.mean(temp)/(self.maxMu-self.minMu)
            mystd=np.std(temp)/(self.maxSig-self.minSig)
            mymean_ori=np.mean(temp)
            mystd_ori=np.std(temp)
            
            #temp_patch+=[mymean,mystd]  
            #temp_patch_ori+=[mymean_ori,mystd_ori]  

            data[kk]=np.copy([mymean,mystd])
            data_ori[kk]=np.copy([mymean_ori,mystd_ori]  )
            
        self.data[id1][id2]=np.copy(data)
        self.data_ori[id1][id2]=np.copy(data_ori)
        
        img=np.asarray(data_2d)
        img=np.reshape(img,(res_h*2,res_w*2))
        self.data_img[id1][id2]=np.copy(img)
        return data,img
        
        
    def get_state_and_location(self):
        
        
        #self.current_pos=np.copy(self.starting_loc)
		
        id1,id2=self.current_pos
        #self.visit_map=np.zeros_like(self.reward_table)

        return np.reshape(self.data[id1][id2],(-1,2*9)),np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)
    
    def get_state(self,positions):
        id1,id2=positions  
        #return  self.add_another_dim(self.data[irow][icol])  
        return  np.reshape(self.data[id1][id2],(-1,2*9))

        
    def current_state(self):
        id1,id2=self.current_pos      
        #return  self.add_another_dim(self.data[irow][icol])
        return  np.reshape(self.data[id1][id2],(-1,2*9))

		
    def add_another_dim(self,myinput):
        return np.expand_dims(myinput,axis=0)

    def get_neightborMapIndividual(self,location):
    
        id1,id2=location
    
        output=[]
        # return a 6 dimensional vector
        if id1==0:
            output.append(0)
        else:
            output.append(self.visit_map[id1-1,id2])
            
        if id1==self.dim[0]-1:
            output.append(0)
        else:
            output.append(self.visit_map[id1+1,id2])
            
        if id2==0:
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2-1])
            
        if id2==self.dim[1]-1:
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2+1])

        # replace zero by -1
        output2=[-1 if o==0 else o*1 for o in output ]
        return output2
    
    def get_neighborMap(self,locations):
        locations=np.asarray(locations)
        if len(locations.shape)==1: # 1 data point
            output=self.get_neightborMapIndividual(locations)
        else:
            output=np.apply_along_axis( self.get_neightborMapIndividual,1,locations)
                    
        return output
    
    def set_session(self, session):
        self.session = session
        
    def step(self,action):
        # perform an action to move to the next state
        
        #0: Decrease dim 1
        #1: Increase dim 1
        #2: Decrease dim 2
        #3: Increase dim 2

        pres_pos=np.copy(self.current_pos)
        if action==0:

            self.current_pos[0]=self.current_pos[0]-1
        elif action==1:

            self.current_pos[0]=self.current_pos[0]+1
        elif action==2:
           
            self.current_pos[1]=self.current_pos[1]-1
        elif action==3:
          
            self.current_pos[1]=self.current_pos[1]+1
        else:
            print("action is 0-4")
   
       
        id1,id2=self.current_pos
        
        
        self.visit_map[id1,id2]+=1

        done=False
  
        if self.data[id1][id2]==0: # not measured yet
            data,img=self.get_data(self.current_pos,pres_pos)
            
            self.data[id1][id2]=data
            self.data_img_ori[id1][id2]=np.copy(img)
            self.data_img[id1][id2]=(img-self.minCurrent)/(self.maxCurrent-self.minCurrent)
            
        obs=self.data[id1][id2]
        
        done=self.is_state_quantum(obs)     
        if done:
            self.isquantum[id1,id2]=1
        
        loc_x=np.copy(self.current_pos)
    
        return obs, done, loc_x
    
    def Sym_KL(self, a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)
    
        return 0.5*np.sum(np.where(a != 0, a * np.log(a / b), 0)) + 0.5*np.sum(np.where(b != 0, b * np.log(b /a), 0))

    def predict_scoreQuantum(self,input_block_feat): # input_block_feat includes [mean, std]
        
        # convert to original scale
        input_block_feat=input_block_feat*([self.maxMu-self.minMu,\
                                            self.maxSig-self.minSig])
        
        # distance to golden
        
        distQ=self.Sym_KL(input_block_feat,self.dd_dist)
        distS=self.Sym_KL(input_block_feat,self.sd_dist)
        distB=self.Sym_KL(input_block_feat,self.barrier_dist)
        
        idxMin=np.argmin([distQ,distS,distB])
        # distance to Barrier
        #distB=self.Sym_KL(input_block_feat,self.dd_dist)
        #if dist<0.004:
        if idxMin==0 and distQ<0.001:
        #if dist<0.004:
            return True,distQ
        else:
            return False,distQ
        
    def predict_scoreBarrier(self,input_block_feat): # input_block_feat includes [mean, std]
        # convert to original scale
        input_block_feat=input_block_feat*([self.maxMu-self.minMu,\
                                            self.maxSig-self.minSig])
        # distance to golden
        #distG=self.Sym_KL(input_block_feat,self.dd_dist)
        # distance to Barrier
        distQ=self.Sym_KL(input_block_feat,self.dd_dist)
        distS=self.Sym_KL(input_block_feat,self.sd_dist)
        distB=self.Sym_KL(input_block_feat,self.barrier_dist)
        
        idxMin=np.argmin([distQ,distS,distB])        #if dist<0.004:
        if idxMin==2:
        #if distB<0.1: # I185
            return True,distB
        else:
            return False,distB

    def predict_scoreShortCircut(self,input_block_feat): # input_block_feat includes [mean, std]
        # convert to original scale
        input_block_feat=input_block_feat*([self.maxMu-self.minMu,\
                                            self.maxSig-self.minSig])
        # distance to golden
        #distG=self.Sym_KL(input_block_feat,self.dd_dist)
        # distance to Barrier
        distQ=self.Sym_KL(input_block_feat,self.dd_dist)
        distS=self.Sym_KL(input_block_feat,self.sd_dist)
        distB=self.Sym_KL(input_block_feat,self.barrier_dist)
        
        idxMin=np.argmin([distQ,distS,distB])        #if dist<0.004:
        if idxMin==1:
        #if distB<0.1: # I185
            return True,distS
        else:
            return False,distS
        
    def is_state_quantum(self,state):
        
        countQ,countB=0,0
        for uu in range(9): # 3**len(self.dim) = 9
            #predicted_value=self.cnnmodel.predict(obs[uu])
            isQuantum,scoreQ=self.predict_scoreQuantum(state[uu])             
            isBarrier,scoreB=self.predict_scoreBarrier(state[uu])             
            
            if isQuantum is True:
                #sumscore+=score
                countQ+=1
            if isBarrier is True:
                countB+=1
                
        if (countQ==1 or countQ==2) and countB>=7:
            return True
        
        return False
            
  
    def reset_at_rand_loc(self):
        self.current_pos=[np.random.randint(0,self.dim[0]),np.random.randint(0,self.dim[1])]
		
        id1,id2=self.current_pos
        self.visit_map=np.zeros_like(self.reward_table)
        
        return self.data[id1][id2],np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)

    def reset(self):
        self.current_pos=np.copy(self.starting_loc)
		
        id1,id2=self.current_pos
        self.visit_map=np.zeros_like(self.reward_table)

        return self.data[id1][id2],np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)

