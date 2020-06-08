# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:36:00 2019

@author: Vu
"""

#reset
import sys

sys.path.append('../')
sys.path.append('/home/sebastian/Documents/ml/binary_classifier')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools
from tqdm import tqdm
import pickle   
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import models

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
    
    
class Quantum_T4_2D: # 2D
    # in the test environment, we start at a particular location
    # then, we start from there, we can not pre-define the blocks.
    
       
    def possible_actions_from_location(self,location=None):
        
        if location is None:
            location =self.current_pos
        irow,icol=location
        
        possible_actions=[]
        
        if self.isRepeat is True:
            if irow>0: #decrease d1
                possible_actions.append(0)
            if irow<self.dim[0]-1: #increase d1
                possible_actions.append(1)
            if icol>0: #decrease d2
                possible_actions.append(2)
            if icol<self.dim[1]-1: #increase d2
                possible_actions.append(3)
            if irow<(self.dim[0]-1) and  icol<(self.dim[1]-1): #decrease d1 and d2
                possible_actions.append(4)
            if (irow>0) and  (icol>0): #increase d1 and  d2
                possible_actions.append(5)
        else:
            
            if irow>0 and self.visit_map[irow-1,icol]==0: #up
                possible_actions.append(0)
            if irow<self.dim[0]-1 and self.visit_map[irow+1,icol]==0: #down
                possible_actions.append(1)
            if icol>0 and self.visit_map[irow,icol-1]==0: #left
                possible_actions.append(2)
            if icol<self.dim[1]-1 and self.visit_map[irow,icol+1]==0: # right
                possible_actions.append(3)
            if irow<(self.dim[0]-1) and  icol<(self.dim[1]-1) and self.visit_map[irow+1,icol+1]==0: #decrease d1 and d2
                possible_actions.append(4)
            if (irow>0) and  (icol>0) and self.visit_map[irow-1,icol-1]==0: #increase d1 and  d2
                possible_actions.append(5)

        #possible_actions=[0,1,2,3,4,5]
        return possible_actions
        
    def construct_block_given_starting_locations(self,starting_pixel_locs):
                
        w1=self.bh
        w2=self.bw
        
        [idx1,idx2]=starting_pixel_locs
        
        img=self.image
        
        #img=self.image
        extended_img=self.extended_image
        
        self.max_d1,self.max_d2=img.shape
        #print('Shape for patching',self.imgheight,self.imgwidth)
        
        self.current_pos=np.copy([np.int(idx1/w1),np.int(idx2/w2)])
        self.starting_loc=np.copy(self.current_pos)
        
        # scale the data to 0-1
        range_data=[np.min(extended_img),np.max(extended_img)]
        
        nDim1=math.ceil(self.max_d1/w1)
        nDim2=math.ceil( self.max_d2/w2)

        count=0

        patch_data=[0]*nDim1
        image_largepatch_data=[0]*nDim1
        image_smallpatch_data=[0]*nDim1
        
        MaxExtImg_d1=extended_img.shape[0]
        MaxExtImg_d2=extended_img.shape[1]
        
        maxMu,minMu,maxSig,minSig=0,10,0,10

        for ii in range(nDim1):
            patch_data[ii]=[0]*nDim2
            image_largepatch_data[ii]=[0]*nDim2
            image_smallpatch_data[ii]=[0]*nDim2

            for jj in range(nDim2):
               
                # expand to 64x64
                patch=extended_img[max(0,ii*w1-w1):min(ii*w1+w1+w1,MaxExtImg_d1),
                                   max(0,jj*w2-w2):min(jj*w2+w2+w2,MaxExtImg_d2)] #2D
                count+=1
                
                size_patch=patch.shape
                
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
                
                #patch_data[ii][jj][kk]=[]
                temp_patch=[]
                image_smallpatch_data[ii][jj]=[0]*len(pp_blocks)

                for kk,mypp in enumerate(pp_blocks): # 27 items
                    #print(kk)
                    temp=patch[mypp[0][0]:mypp[0][1],
                               mypp[1][0]:mypp[1][1]]
                                #mypp[2][0]:mypp[2][1]]
                        
                    # image denoise
                    #temp2 = ndimage.median_filter(temp, int(w1/8))
                    temp2=temp

                    #patch_data[ii][jj][kk]+=[np.mean(temp),np.std(temp)]  
                    temp_patch+=[np.mean(temp2),np.std(temp2)]  
                    image_smallpatch_data[ii][jj][kk]=np.copy(temp2)
                    
                    minMu=min(minMu,np.mean(temp2))
                    maxMu=max(maxMu,np.mean(temp2))
                    minSig=min(minSig,np.std(temp2))
                    maxSig=max(maxSig,np.std(temp2))
                    
                patch_data[ii][jj]=temp_patch
                image_largepatch_data[ii][jj]=patch
        
        #print("minMu",minMu,"maxMu",maxMu,"minSig",minSig,"maxSig",maxSig)
        #minMu,maxMu,minSig,maxSig=0,0.9,0,0.39

        self.minMu=minMu
        self.maxMu=maxMu
        self.minSig=minSig
        self.maxSig=maxSig
        
        # normalize patch_data
        for ii in range(nDim1):
            for jj in range(nDim2):
                temp=patch_data[ii][jj]
                temp=np.asarray(temp)
                temp=np.reshape(temp,(-1,2))
                if self.maxMu==self.minMu:
                    print("maxMu==minMu")
                    
                if self.maxSig==self.minSig:
                    print("maxSig==minSig")
                temp[:,0]=(temp[:,0])/(self.maxMu-self.minMu)
                temp[:,1]=(temp[:,1])/(self.maxSig-self.minSig)
                patch_data[ii][jj]=np.copy(temp.ravel())
        
        return patch_data, [nDim1,nDim2],range_data,image_largepatch_data,image_smallpatch_data
 
    def take_reward_table(self):
        reward_table=(-0.05)*np.ones((self.dim))
        [id1,id2]=np.where(self.isquantum==1)
        reward_table[id1,id2]=5
        return reward_table   
        
        
    def __init__(self, file_name="",starting_pixel_loc=[0,0],bh=18,bw=18,isRepeat=True,offset=0.0):
        # img_row_idx and img_col_idx are the pixel indices, not the block indices
        
        self.isRepeat=isRepeat #allow reselect visited location
        
        #print("file name=",file_name)
        
        self.bw=bw #block width
        self.bh=bh #block height
        
        # this needs to be checked
        #self.dd_dist=[0.02, 0.05] # mean and variance of golden observation
        
        # this needs to be checked
        self.barrier_dist=[0.02,0.1]
        self.sd_dist=[0.08,0.2]
        #self.dd_dist=[0.03, 0.12]
        self.dd_dist=[0.03, 0.2]
        # action space
        self.K=6
        
        #self.offset=2.0e-10 # basel 1
        #self.offset=0.0 #basel 2
        self.offset = offset
        
        #resolution=300
        #window=300
        
        # load multiple data scan into the memory
        try:
            strFile="../data/{}.p".format(file_name)
            #strFile="../data/{}.p".format(file_name)
            
            self.image = pickle.load( open(strFile, "rb" ) )
        except:
            strFile="data/{}.p".format(file_name)
            
            self.image = pickle.load( open(strFile, "rb" ) )

        self.image=self.image+self.offset
        # find min positive
        idxPos=np.where(self.image>0)
        min_pos=np.min(self.image[idxPos])
        
        
        
        idxNegative=np.where(self.image<0)
        self.image[idxNegative]=min_pos

        self.image=(self.image-np.min(self.image))/(np.max(self.image)-np.min(self.image))

        self.img_dim=self.image.shape
        
        # padding to four direction
        #self.extended_image=np.pad(self.image, (16, 16), 'constant',constant_values=0)
        self.extended_image=np.pad(self.image, ( int(self.bh/2), int(self.bw/2)), 'edge')


        #self.data,self.dim,self.range_data=self.load_image(path)
        
        # base on this location, construct the data
        self.data,self.dim,self.range_data,self.image_largepatch_data,self.image_smallpatch_data=\
        self.construct_block_given_starting_locations(starting_pixel_loc)
        #self.data=np.copy(self.bigpatch_data)
        
        #D is the dimension of each patch
        # self.dim is the dimension of blocks in the images
        self.D=len(self.data[0][0])
        
        #self.starting_loc=[np.int(self.dim[0]/2), np.int(self.dim[0]/2)]
        
        self.current_pos=np.copy(self.starting_loc)
        self.pre_classify()
         #self.isquantum=None
        self.where_is_quantum() # compute self.isquantum
        
        #self.reward_table=self.load_reward_table()
        self.reward_table=self.take_reward_table()
        
        self.visit_map=np.zeros_like(self.reward_table)
        
       
        #self.train_cnn_model(sess)

    def pre_classify(self):

        self.mid_point_x = math.floor(len(self.image[:, 0]) / 2.0)
        self.mid_point_y = math.floor(len(self.image[0, :]) / 2.0)
        self.trace_x = self.image[self.mid_point_x, :]
        self.trace_y = self.image[:, self.mid_point_y]
        self.trace_range = max(self.trace_x) - min(self.trace_x)
        self.threshold_1 = self.trace_range * 0.3
        self.threshold_2 = self.trace_range * 0.05

    def get_state_and_location(self):
        #self.current_pos=np.copy(self.starting_loc)
        id1,id2=self.current_pos
        self.visit_map=np.zeros_like(self.reward_table)
        return np.reshape(self.data[id1][id2],(-1,2*9)),np.copy(self.current_pos)

    def get_state(self,positions):
        id1,id2=positions  
        #return  self.add_another_dim(self.data[irow][icol])  
        return  np.reshape(self.data[id1][id2],(-1,2*9))

        
    def current_state(self):
        id1,id2=self.current_pos      
        #return  self.add_another_dim(self.data[irow][icol])
        return  np.reshape(self.data[id1][id2],(-1,2*9))

 
    def get_reward(self,positions):
        id1,id2=positions

        r= self.reward_table[id1,id2]
        r=r-0.5*self.visit_map[id1,id2]
    
        return r
		
    def add_another_dim(self,myinput):
        return np.expand_dims(myinput,axis=0)

    def get_neightborMapIndividual(self,location):
    
        id1,id2=location
    
        norm_factor=5.0
        output=[]
        # return a 6 dimensional vector
        if id1==0: #decrease d1
            output.append(0)
        else:
            output.append(self.visit_map[id1-1,id2]/norm_factor)
            
        if id1==self.dim[0]-1: # increase d1
            output.append(0)
        else:
            output.append(self.visit_map[id1+1,id2]/norm_factor)
            
        if id2==0: # decrease d2
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2-1]/norm_factor)
            
        if id2==self.dim[1]-1: # increase d2
            output.append(0)
        else:
            output.append(self.visit_map[id1,id2+1]/norm_factor)
            
        if id1<self.dim[0]-1 and id2<self.dim[1]-1: # decrease d1 and decrease d2
            output.append(self.visit_map[id1+1,id2+1]/norm_factor)
        else:
            output.append(0)
            
        if id1>0 and id2>0: # increase d1 and increase d2
            output.append(self.visit_map[id1-1,id2-1]/norm_factor)
        else:
            output.append(0)

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
        flagoutside=0
        #flagRepeat=0
        
        if action==0:
            if self.current_pos[0]==0:
                flagoutside=1
                print("cannot decrease d1")
            else:
                self.current_pos[0]=self.current_pos[0]-1
        elif action==1:
            if self.current_pos[0]==self.dim[0]-1:
                flagoutside=1
                print("cannot increase d1")
            else:
                self.current_pos[0]=self.current_pos[0]+1
        elif action==2:
            if self.current_pos[1]==0:
                flagoutside=1
                print("cannot decrease d2")
            else:
                self.current_pos[1]=self.current_pos[1]-1
        elif action==3:
            if self.current_pos[1]==self.dim[1]-1:
                flagoutside=1
                print("cannot decrease d2")
            else:
                self.current_pos[1]=self.current_pos[1]+1
        elif action==4:
            if self.current_pos[0]<self.dim[0]-1 and self.current_pos[1]<self.dim[1]-1:
                self.current_pos[1]=self.current_pos[1]+1
                self.current_pos[0]=self.current_pos[0]+1
            else:
                flagoutside=1
                print("cannot increase both d1 and d2")

        elif action==5:
            if self.current_pos[0]>0 and self.current_pos[1]>0:
                self.current_pos[1]=self.current_pos[1]-1
                self.current_pos[0]=self.current_pos[0]-1
            else:
                flagoutside=1
                print("cannot decrease both d1 and d2")
        else:
            print("action is 0-6")
        
        id1,id2=self.current_pos
        
        if flagoutside==1:
            loc_x=np.copy(self.current_pos)
            r=-8 # terminate
            done=True
            obs=self.data[id1][id2]
        else:
            if self.visit_map[id1,id2] == 1:
                r = 0
                obs = np.zeros_like(self.data[id1][id2])
                done = False
                loc_x = np.copy(self.current_pos)
                return obs, r, done, loc_x

            r=self.get_reward(self.current_pos)
            self.visit_map[id1,id2]+=1
    
            done=False
      
            obs=self.data[id1][id2]
            
            if self.isquantum is None:
                self.where_is_quantum() # compute self.isquantum
                
            if self.isquantum[id1,id2]==1:
                "Quantum found"
                r += 100
                done=True
                #print('**************************')
                #print('--------------------------')
                #print("BIAS TRIANGLE FOUND")
                #print(self.current_pos)
                #print('--------------------------')
                #print('**************************')
            
            loc_x=np.copy(self.current_pos)
    
        return obs, r, done, loc_x
    
    
    def KL(self,p, q):
        [mu1,sig1]=p
        [mu2,sig2]=q
        
        temp1=np.log(sig2/sig1)
        temp2=(sig1**2+(mu1-mu2)**2)/(2*sig2**2)-0.5
       
        return temp1+temp2

    def Sym_KL(self, a, b):
        #a = np.asarray(a, dtype=np.float)+epsilon
        #b = np.asarray(b, dtype=np.float)+epsilon
        return 0.5*self.KL(a,b)+0.5*self.KL(b,a)
        #return 0.5*np.sum(np.where(a != 0, a * np.log(a / b), 0)) \
            #+ 0.5*np.sum(np.where(b != 0, b * np.log(b /a), 0))

    def predict_scoreQuantum(self,input_block_feat): # input_block_feat includes [mean, std]
        # distance to golden
        
        distQ=self.Sym_KL(input_block_feat,self.dd_dist)
        distS=self.Sym_KL(input_block_feat,self.sd_dist)
        distB=self.Sym_KL(input_block_feat,self.barrier_dist)
        
        idxMin=np.argmin([distQ,distS,distB])
        # distance to Barrier
       
        if idxMin==0 and distQ<0.005:
            return True,distQ
        else:
            return False,distQ
        
    def predict_scoreBarrier(self,input_block_feat): # input_block_feat includes [mean, std]
        
        # distance to golden
        
        # distance to Barrier
        distQ=self.Sym_KL(input_block_feat,self.dd_dist)
        distS=self.Sym_KL(input_block_feat,self.sd_dist)
        distB=self.Sym_KL(input_block_feat,self.barrier_dist)
        
        idxMin=np.argmin([distQ,distS,distB])        
        if idxMin==2:
            return True,distB
        else:
            return False,distB
        
    def compute_entropy_block(self,myblock):
        # myblock is a block image
        w,h=myblock.shape
        myblock = myblock.reshape((myblock.shape[0] * myblock.shape[1], 1))

        # perform KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(myblock)
        
        idxCluster_One=np.where(kmeans.labels_==1)[0]
        idxCluster_Zero=np.where(kmeans.labels_==0)[0]

        idx_2d_Zero=[idxCluster_Zero/w,idxCluster_Zero%w]

        idx_2d_One=[idxCluster_One/w,idxCluster_One%w]
        
        stdOne=np.std(idx_2d_One,axis=1)   
        stdZero=np.std(idx_2d_Zero,axis=1) 
        
        return stdZero, stdOne
        
    def check_where_is_quantum_2d(self,ii,jj):
        obs=np.reshape(self.data[ii][jj],(3**len(self.dim),2)) # for predicting quantum, we use multiple smaller blocks

        countQ=0
        countB=0
        #sumscore=0
        for uu in range(3**len(self.dim)): # 3**len(self.dim) = 9
            
            isQuantum,scoreQ=self.predict_scoreQuantum(obs[uu])         
            isBarrier,scoreB=self.predict_scoreBarrier(obs[uu])             
            
            if isQuantum is True:
                
                stdZero,stdOne=self.compute_entropy_block(self.image_smallpatch_data[ii][jj][uu])
                #print("Quantum found")
                if  np.max(stdOne)<8 and np.min(stdOne)>1:
                    #print(stdOne)

                    #sumscore+=score
                    countQ+=1
            if isBarrier is True:
                countB+=1
                
        if (countQ==1 or countQ==2 or countQ==3) and countB>=5:
            return 1
        else:
            return 0
       
    def normalise(self,x):
        y = tf.image.per_image_standardization(x)
        return y

    def load_cnn(self):
        model_binary_classifier = models.load_model('/home/sebastian/Documents/ml/binary_classifier/bias_triangle_binary_classifier.h5')
        return model_binary_classifier 
        
    def check_for_bias_triangle(self,ii,jj):
        statistics = self.data[ii][jj]
        means = statistics[:9]
        for mean in means:
            if (mean > self.threshold_2) and (mean < self.threshold_1):
                self.threshold_test[ii, jj] += 1

        if self.threshold_test[ii,jj] == 0:
            return 0

        #print("Passed pre-classification")
        large_patch = self.image_largepatch_data[ii][jj]
        x, y =np.shape(large_patch)
        test_image= self.normalise(tf.image.resize(np.array(large_patch).reshape(-1, x, y, 1), (32,32)))
        self.prediction[ii,jj] =  self.model_binary_classifier.predict(test_image,steps=1)
        #print("Classification ",self.prediction[ii,jj])
        #.imshow(large_patch)
        #plt.show()
        if self.prediction[ii,jj] > 0.5:
            return 1
        else:
            return 0
                
    def where_is_quantum(self):
        self.model_binary_classifier = self.load_cnn()
        # return a map telling the quantum location
        ndim1,ndim2=self.dim
        self.isquantum=np.zeros(self.dim)
        self.threshold_test=np.zeros(self.dim)
        self.prediction = np.zeros(self.dim)
        for ii in tqdm(range(ndim1)):
            for jj in range(ndim2):
                #self.isquantum[ii,jj]=self.check_where_is_quantum_2d(ii,jj)
                self.isquantum[ii,jj]=self.check_for_bias_triangle(ii,jj)
                
        return self.isquantum
                
    
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

        #print("shape of data",np.shape(self.data))
        #print("id1",id1)
        #print("id2", id2)
        return self.data[id1][id2],np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)

       # return  np.expand_dims(self.data[irow][icol],axis=0)



# test 3D model
       


