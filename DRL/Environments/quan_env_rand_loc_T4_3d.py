# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:36:00 2019

@author: Vu
"""

#reset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from tf_cnn_classification_quantum_images import CNNQuantum
import math
import itertools
from tqdm import tqdm
import gc      

import visvis as vv 
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    
    
class Quantum_T4_3D: # 3D
    # in the test environment, we start at a particular location
    # then, we start from there, we can not pre-define the blocks.
    
    def possible_actions(self):
                        
        id1,id2,id3=self.current_pos
        
        possible_actions=[]
        
        if id1>0: #decrease d1
            possible_actions.append(0)
        if id1<self.dim[0]-1: #increase d1
            possible_actions.append(1)
        if id2>0: #decrease d2
            possible_actions.append(2)
        if id2<self.dim[1]-1: #increase d2
            possible_actions.append(3)
        if id3>0: #decrease d3
            possible_actions.append(4)
        if id3<self.dim[2]-1: #increase d3
            possible_actions.append(5)
        return possible_actions
    
    def possible_actions_from_location(self,location):
        
        id1,id2,id3=location
        
        possible_actions=[]
        
        if id1>0: #decrease d1
            possible_actions.append(0)
        if id1<self.dim[0]-1: #increase d1
            possible_actions.append(1)
        if id2>0: #decrease d2
            possible_actions.append(2)
        if id2<self.dim[1]-1: #increase d2
            possible_actions.append(3)
        if id3>0: #decrease d3
            possible_actions.append(4)
        if id3<self.dim[2]-1: #increase d3
            possible_actions.append(5)
        return possible_actions
        
    
    def construct_block_given_starting_locations(self, starting_pixel_locs=[0,0,0], w1=8, w2=8,w3=4):
                
        [idx1,idx2,idx3]=starting_pixel_locs
        
        img=self.image
        
        #img=self.image
        extended_img=self.extended_image
        
        self.max_d1,self.max_d2,self.max_d3=img.shape
        #print('Shape for patching',self.imgheight,self.imgwidth)
        
        self.current_pos=np.copy([np.int(idx1/w1),np.int(idx2/w2),np.int(idx3/w3)])
        self.starting_loc=np.copy(self.current_pos)
        
        # scale the data to 0-1
        range_data=[np.min(extended_img),np.max(extended_img)]
        
        nDim1=math.ceil(self.max_d1/w1)
        nDim2=math.ceil( self.max_d2/w2)
        nDim3=math.ceil( self.max_d3/w3)

        count=0

        patch_data=[0]*nDim1
        for ii in range(nDim1):
            patch_data[ii]=[0]*nDim2
            for jj in range(nDim2):
                patch_data[ii][jj]=[0]*nDim3
                for kk in range(nDim3):

                    # first, collect a large patch
                    patch=extended_img[ii*w1:ii*w1+w1,jj*w2:jj*w2+w2,kk*w3:kk*w3+w3]
                    
                    """
                    print(ii*w1,ii*w1+w1,jj*w2,jj*w2+w2,kk*w3,kk*w3+w3)
                    strPath="debug/img_{}".format(count)
                    f, axarr = plt.subplots(2,2,figsize=(12,12))
                    axarr[0,0].imshow(patch[:,:,0],vmin=0,vmax=1)
                    axarr[0,1].imshow(patch[:,:,0],vmin=0,vmax=1)
                    axarr[1,0].imshow(patch[:,:,0],vmin=0,vmax=1)
                    axarr[1,1].imshow(patch[:,:,0],vmin=0,vmax=1)
                    f.savefig(strPath,bbox_inches = 'tight')
        
                    # release RAM
                    f.clf()
                    plt.close()
                    gc.collect()
                    """
                    
                    count+=1
                    
                    size_patch=patch.shape
                    
                    # find p1,p2,p3 equally between l1,l2,l3
                    mypp=[0]*3
                    pp_block=[0]*3
                    for dd in range(3):
                        mypp[dd]=[0]*5
                        temp=np.linspace(0, size_patch[dd], num=5)
                        temp=temp.astype(int)
                        mypp[dd]=temp.tolist()
                        
                        pp_block[dd]=[[mypp[dd][0],mypp[dd][2]],[mypp[dd][1],mypp[dd][3]],
                                 [mypp[dd][2],mypp[dd][4]]]
                        
                    # based on p1,p2,p3, we split into 3^3 blocks
                    pp_blocks=list(itertools.product(*pp_block))
                    # 27 elements
                    
                    #patch_data[ii][jj][kk]=[]
                    temp_patch=[]
                    for mypp in pp_blocks: # 27 items
                        temp=patch[mypp[0][0]:mypp[0][1],
                                   mypp[1][0]:mypp[1][1],
                                    mypp[2][0]:mypp[2][1]]

                        #patch_data[ii][jj][kk]+=[np.mean(temp),np.std(temp)]  
                        temp_patch+=[np.mean(temp),np.std(temp)]  
                        
                    patch_data[ii][jj][kk]=temp_patch
        
        return patch_data, [nDim1,nDim2,nDim3],range_data
 
    def take_reward_table(self):
        reward_table=(-0.05)*np.ones((self.dim))
        [id1,id2,id3]=np.where(self.isquantum==1)
        reward_table[id1,id2,id3]=5
        #print(id1,id2,id3)
        return reward_table   
        
        
    def __init__(self, starting_pixel_loc):
        # img_row_idx and img_col_idx are the pixel indices, not the block indices
        
        
        
        # this needs to be checked
        #self.golden_dist=[0.02, 0.05] # mean and variance of golden observation
        self.barrier_dist=[0.01,0.005]
        self.sd_dist=[0.2,0.2]
        
        self.dd_dist=[0.02, 0.05]
        
        # action space
        self.K=6
        
        # observation space
        #path='vu_drl\data\scanc5c9_0_512_lowbias.npz'
        #path="data\RL_3d_wrwln_bias18.npz"
        #path='data\I1_174.txt'

        try:
            temp=np.load('../data/arr_0.npy')
        except:
            temp=np.load('../vu_drl/data/arr_0.npy')
            
        data=temp.tolist()["data"]
        
        data=(data-np.min(data))/(np.max(data)-np.min(data))
        
        self.image=data.reshape((128,128,59))
        
        #data=np.load(path)
        #self.image=data['img'][0] # 32 x 32 patches
        
        #self.image=np.genfromtxt(path,unpack=True)
                
        #self.image=np.rot90(np.rot90(data['arr_0'].item()['data'][0,:,:,0])) # 8 x8 patches

        self.img_dim=self.image.shape
        
        # padding to four direction
        #self.extended_image=np.pad(self.image, (16, 16), 'constant',constant_values=0)
        self.extended_image=np.pad(self.image, (16, 16), 'edge')


        #self.data,self.dim,self.range_data=self.load_image(path)
        
        # base on this location, construct the data
        self.data,self.dim,self.range_data=self.construct_block_given_starting_locations(starting_pixel_loc)
        #self.data=np.copy(self.bigpatch_data)
        
        #D is the dimension of each patch
        # self.dim is the dimension of blocks in the images
        self.D=len(self.data[0][0][0])
        
        #self.starting_loc=[np.int(self.dim[0]/2), np.int(self.dim[0]/2)]
        
        self.current_pos=np.copy(self.starting_loc)
        
         #self.isquantum=None
        self.where_is_quantum() # compute self.isquantum
        
        #self.reward_table=self.load_reward_table()
        self.reward_table=self.take_reward_table()
        
        #self.visit_map=np.zeros_like(self.reward_table)
        
       
        #self.train_cnn_model(sess)
        
    def get_state_and_location(self):
        self.current_pos=np.copy(self.starting_loc)
		
        id1,id2,id3=self.current_pos
        self.visit_map=np.zeros_like(self.reward_table)

        return np.reshape(self.data[id1][id2][id3],(-1,54)),np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)
    
    def get_state(self,positions):
        id1,id2,id3=positions  
        #return  self.add_another_dim(self.data[irow][icol])  
        return  np.reshape(self.data[id1][id2][id3],(-1,54))

        
    def current_state(self):
        id1,id2,id3=self.current_pos      
        #return  self.add_another_dim(self.data[irow][icol])
        return  np.reshape(self.data[id1][id2][id3],(-1,54))

    """ # this is 3D
    def plot_current_state(self):
        plt.figure()
        irow,icol=self.current_pos
        plt.imshow(self.data[irow][icol][0])
	
    """
    def get_reward(self,positions):
        id1,id2,id3=positions
        #print(irow,icol,self.reward_table.shape)

        #print(self.reward_table.shape,positions)
        r= self.reward_table[id1,id2,id3]
        r=r-0.5*self.visit_map[id1,id2,id3]
    
        return r
		
    def add_another_dim(self,myinput):
        return np.expand_dims(myinput,axis=0)

    def get_neightborMapIndividual(self,location):
    
        id1,id2,id3=location
        norm_factor=5.0
        output=[]
        # return a 6 dimensional vector
        if id1==0:
            output.append(0)
        else:
            output.append(self.visit_map[id1-1,id2,id3]/norm_factor)
            
        if id1==self.dim[0]-1:
            output.append(0)
        else:
            output.append(self.visit_map[id1+1,id2,id3]/norm_factor)
            
        if id2==0:
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2-1,id3]/norm_factor)
            
        if id2==self.dim[1]-1:
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2+1,id3]/norm_factor)

        if id3==0:
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2,id3-1]/norm_factor)
            
        if id3==self.dim[2]-1:
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2,id3+1]/norm_factor)

        # replace zero by -1
        output2=[-1/norm_factor if o==0 else o*1 for o in output ]
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
        #4: Decrease dim 3
        #5: Increase dim 3

        if action==0:
            if self.current_pos[0]==0:
                print("cannot decrease d1")
            else:
                self.current_pos[0]=self.current_pos[0]-1
        elif action==1:
            if self.current_pos[0]==self.dim[0]-1:
                print("cannot increase d1")
            else:
                self.current_pos[0]=self.current_pos[0]+1
        elif action==2:
            if self.current_pos[1]==0:
                print("cannot decrease d2")
            else:
                self.current_pos[1]=self.current_pos[1]-1
        elif action==3:
            if self.current_pos[1]==self.dim[1]-1:
                print("cannot decrease d2")
            else:
                self.current_pos[1]=self.current_pos[1]+1
        elif action==4:
            if self.current_pos[2]==0:
                print("cannot decrease d3")
            else:
                self.current_pos[2]=self.current_pos[2]-1
        elif action==5:
            if self.current_pos[2]==self.dim[2]-1:
                print("cannot decrease d3")
            else:
                self.current_pos[2]=self.current_pos[2]+1   
        else:
            print("action is 0-6")
            
        id1,id2,id3=self.current_pos
        
        r=self.get_reward(self.current_pos)
        
        self.visit_map[id1,id2,id3]+=1


        done=False
  
        obs=self.data[id1][id2][id3]
        
        if self.isquantum is None:
            self.where_is_quantum() # compute self.isquantum
            
        if self.isquantum[id1,id2,id3]==1:
            done=True
        
        loc_x=np.copy(self.current_pos)
    
        return obs, r, done, loc_x
    
    def Sym_KL(self, a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)
    
        return 0.5*np.sum(np.where(a != 0, a * np.log(a / b), 0)) + 0.5*np.sum(np.where(b != 0, b * np.log(b /a), 0))

    def predict_score(self,input_block_feat): # input_block_feat includes [mean, std]
        
        dist=self.Sym_KL(input_block_feat,self.golden_dist)
        if dist<0.004:
            return True
        else:
            return False
        
    def predict_scoreQuantum(self,input_block_feat): # input_block_feat includes [mean, std]
        
        # distance to golden
        distQ=self.Sym_KL(input_block_feat,self.dd_dist)
        distS=self.Sym_KL(input_block_feat,self.sd_dist)
        distB=self.Sym_KL(input_block_feat,self.barrier_dist)
        
        idxMin=np.argmin([distQ,distS,distB])
        # distance to Barrier
        #distB=self.Sym_KL(input_block_feat,self.dd_dist)
        #if dist<0.004:
        if idxMin==0 and distQ<0.008:
        #if dist<0.004:
            return True,distQ
        else:
            return False,distQ
        
    def predict_scoreBarrier(self,input_block_feat): # input_block_feat includes [mean, std]
        
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
        
    def where_is_quantum(self):
        # return a map telling the quantum location
        ndim1,ndim2,ndim3=self.dim
        self.isquantum=np.zeros(self.dim)
        for ii in range(ndim1):
            for jj in range(ndim2):
                for kk in range(ndim3):
                    obs=np.reshape(self.data[ii][jj][kk],(27,2)) # for predicting quantum, we use multiple smaller blocks
                    #print(ii,jj)
                    countQ=0
                    countB=0
                    for uu in range(27):
                        #predicted_value=self.cnnmodel.predict(obs[uu])
                        
                        isQuantum,scoreQ=self.predict_scoreQuantum(obs[uu])             
                        isBarrier,scoreB=self.predict_scoreBarrier(obs[uu])             
                    
                        if isQuantum is True:
                        #sumscore+=score
                            countQ+=1
                        if isBarrier is True:
                            countB+=1
                            
                    if (countQ==1 or countQ==2 or countQ==3) and countB>=16:
                        self.isquantum[ii,jj,kk]=1
                        #break
        #print("sum quantum=",np.sum(self.isquantum))
        return self.isquantum
                
    
    def reset_at_rand_loc(self):
        self.current_pos=[np.random.randint(0,self.dim[0]),np.random.randint(0,self.dim[1]),np.random.randint(0,self.dim[2])]
		
        id1,id2,id3=self.current_pos
        self.visit_map=np.zeros_like(self.reward_table)
        
        return self.data[id1][id2][id3],np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)

    def reset(self):
        self.current_pos=np.copy(self.starting_loc)
		
        id1,id2,id3=self.current_pos
        self.visit_map=np.zeros_like(self.reward_table)

        return self.data[id1][id2][id3],np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)

       # return  np.expand_dims(self.data[irow][icol],axis=0)

"""
def plot_3d_contour(x_dim, y_dim, x_steps, y_steps, scalar_field, file_path):
    from matplotlib import cm

    fig = plt.figure()

    x, y = np.mgrid[-x_dim/2:x_dim/2:x_steps*1j, -y_dim/2:y_dim/2:y_steps*1j]
    v_min = np.min(scalar_field)
    v_max = np.max(scalar_field)

    ax = fig.gca(projection='3d')

    cset = ax.contourf(x, y, scalar_field, zdir='z', offset=v_min, cmap=cm.coolwarm)
    cset = ax.contourf(x, y, scalar_field, zdir='x', offset=-x_dim/2-1, cmap=cm.coolwarm)
    cset = ax.contourf(x, y, scalar_field, zdir='y', offset=y_dim/2+1, cmap=cm.coolwarm)

    ax.plot_surface(x, y, scalar_field, rstride=10, cstride=10, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_xlim(-x_dim/2-1, x_dim/2+1)
    ax.set_ylabel('Y')
    ax.set_ylim(-y_dim/2-1, y_dim/2+1)
    ax.set_zlabel('Z')
    ax.set_zlim(v_min, v_max)

    plt.savefig(file_path + '.jpg')
    plt.close()
"""
