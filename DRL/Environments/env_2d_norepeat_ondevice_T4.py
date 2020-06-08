# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:36:00 2019
@author: Vu
"""


import sys

sys.path.append('../')
sys.path.append('P:/09.QuantumProjects/Pygor')
import Pygor

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from tf_cnn_classification_quantum_images import CNNQuantum
import math
import itertools
from tqdm import tqdm
import gc      
from sklearn.cluster import KMeans

reward_table=[]
counter=0

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
    
    
class OnDevice_2D_Norepeat_T4: # 2D
    # in the test environment, we start at a particular location
    # then, we start from there, we can not pre-define the blocks.
    
    def possible_actions_from_location(self,location=None):
        
        if location is None:
            location =self.current_pos
        else:
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
            if irow>0 and  icol<(self.dim[1]-1): #decrease d1 and decrease d2
                possible_actions.append(4)
            if (irow<self.dim[0]-1) and  (icol>0): #increase d1 and increase d2
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
            if irow>0 and  icol<(self.dim[1]-1) and self.visit_map[irow-1,icol+1]==0: #decrease d1 and decrease d2
                possible_actions.append(4)
            if (irow<self.dim[0]-1) and  (icol>0) and self.visit_map[irow+1,icol-1]==0: #increase d1 and increase d2
                possible_actions.append(5)

        """
        if id3>0: #decrease d3
            possible_actions.append(4)
        if id3<self.dim[2]-1: #increase d3
            possible_actions.append(5)
        """

        #possible_actions=[0,1,2,3,4,5]
        return possible_actions
        
    
    def plot_full_image(self,counter):
        
        halfWindow=int(self.WindowSz/2)
        w_from=[self.initial_voltage[0]-self.bw*halfWindow,self.initial_voltage[1]-self.bw*halfWindow]
        w_to=[self.initial_voltage[0]+self.bw*halfWindow,self.initial_voltage[1]+self.bw*halfWindow]
        
        
        f=plt.figure(figsize=(15,15))
        emptyimage=np.zeros((self.bw*self.WindowSz,self.bh*self.WindowSz))
        for ii in range(self.dim[0]): # all rows
            for jj in range(self.dim[1]): # all columns
                if self.visit_map[ii,jj]>0:
                    
                    emptyimage[ii*self.bw:(ii+1)*self.bw,jj*self.bh:(jj+1)*self.bh]=\
                        self.data_img[ii][jj][4]
    
        plt.imshow(emptyimage,vmin=0,vmax=0.6,extent=[ w_to[1],w_from[1], w_to[0],w_from[0]])
        plt.xlabel("C9",fontsize=18)
        plt.ylabel("C5",fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        strPath='path/full_img_{:d}.pdf'.format(counter)
        
        f.tight_layout()
        f.savefig(strPath,boxes_inches="tight")  
        f.clf()
        plt.close()
        gc.collect()   
                
    def fill_data_overlap(self,current_loc,c5_val,c9_val):
        # reusing the overlapping block to reduce the cost of measurement
        H=np.int(self.bh)

        A,B,C,D=np.zeros((H,H)),np.zeros((H,H)),np.zeros((H,H)),np.zeros((H,H))


        # please note that the img_bigblock has been added offset
        id1,id2=current_loc
        if id1>0 and self.visit_map[id1-1,id2]>0:#Up: decrease C5
            A=np.copy(self.img_bigblock[id1-1][id2][H:,:H])
            B=np.copy(self.img_bigblock[id1-1][id2][H:,H:])
            
        if id2>0 and self.visit_map[id1,id2-1]>0:#Left, increase C9
            A=np.copy(self.img_bigblock[id1][id2-1][:H,H:])
            C=np.copy(self.img_bigblock[id1][id2-1][H:,H:])
            
        if id1<self.dim[0]-1 and self.visit_map[id1+1,id2]>0:#Down, increase C5
            C=np.copy(self.img_bigblock[id1+1][id2][:H,:H])
            D=np.copy(self.img_bigblock[id1+1][id2][:H,H:])
            
        if id2<self.dim[1]-1 and self.visit_map[id1,id2+1]>0:#Right, decrease C9
            B=np.copy(self.img_bigblock[id1][id2+1][:H,:H])
            D=np.copy(self.img_bigblock[id1][id2+1][H:,:H])
        if id1>0 and id2<self.dim[1]-1 and self.visit_map[id1-1,id2+1]>0:#Right, Up, decrease C5, decrease C9
            B=np.copy(self.img_bigblock[id1-1][id2+1][H:,:H])
        if id1<self.dim[0]-1 and id2>0 and self.visit_map[id1+1,id2-1]>0:#Left, Down, increase C5, increase C9
            C=np.copy(self.img_bigblock[id1+1][id2-1][:H,H:])

            
        if np.sum(A)==0:
            A=self.pygor.do2d(self.gate_names[1],c9_val-np.int(self.bw),c9_val,np.int(self.bw),
                              self.gate_names[0],c5_val-np.int(self.bh),c5_val,np.int(self.bh))
            A=np.copy(A.data[0])#+self.offset
        if np.sum(B)==0:
            B=self.pygor.do2d(self.gate_names[1],c9_val,c9_val+np.int(self.bw),np.int(self.bw),
                              self.gate_names[0],c5_val-np.int(self.bh),c5_val,np.int(self.bh))
            B=np.copy(B.data[0])#+self.offset

        if np.sum(C)==0:
            C=self.pygor.do2d(self.gate_names[1],c9_val-np.int(self.bw),c9_val,np.int(self.bw),
                              self.gate_names[0],c5_val,c5_val+np.int(self.bh),np.int(self.bh))
            C=np.copy(C.data[0])#+self.offset

        if np.sum(D)==0:
            D=self.pygor.do2d(self.gate_names[1],c9_val,c9_val+np.int(self.bw),np.int(self.bw),
                              self.gate_names[0],c5_val,c5_val+np.int(self.bh),np.int(self.bh))
            D=np.copy(D.data[0])#+self.offset

             
        top=np.hstack((A,B))
        bottom=np.hstack((C,D))
        new_data=np.vstack((top,bottom))
            
        return new_data                
      
        
    def __init__(self, name="T4",bh=18,bw=18,isRepeat=True):
        # img_row_idx and img_col_idx are the pixel indices, not the block indices
        self.isRepeat=isRepeat #allow reselect visited location

        defaults_gate=[147.79,-1458.17628565,  -283.97105228,  -622.40207719, -1871.07883712,\
       -276.39614676,  -679.4902697 , -1013.81260623]
        
        #-962.40868202,  -822.30385845,  -802.15568511,  -142.18028428, -1747.07073912,  -742.81312161, -1729.35970301
        #  -861.71015786,  -192.0956257 ,  -430.84084329, -1282.81330064, -812.66953594, -1801.0843591 ,  -738.3556092 ,  -306.16635879
       
        #initial_gate_c8_c12=[ -718.55425722, -679.4902697]
        #initial_gate_c8_c12=[-652.40207719,-699.4902697]
        
        self.bw=bw #block width
        self.bh=bh #block height
        
        if name=="T4":
            print("T4")
            defaults_gate=[30, -873.4068932049555, -1003.3107070097195, -340.0,
                           -1807.2635262868484, -1856.8117974535132, -1867.595781932562, 
                           -416.0, -990.5233920645587]
            defaults_gate=[30, -873.4068932049555, -1003.3107070097195, -280.0,
                           -1807.2635262868484, -1856.8117974535132, -1867.595781932562, 
                           -486.0, -990.5233920645587]
            
            
            defaults_gate=[30, -852.91749095,  -989.38046736,  -290, -1719.22163011, 
               -1703.85973708, -1860.66172595,  -290,  -986.43953877    ]
            
            self.pygor=Pygor.Experiment(mode='none', xmlip="http://129.67.86.107:8000/RPC2")
            gates=['c1','c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']

            self.pygor.setvals(gates,defaults_gate)
            initial_gate_cu_cv=[ defaults_gate[3], defaults_gate[7]]
            
            self.gate_names=["c5","c9"]

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
        
        self.barrier_dist=[0.02,0.1]
        self.sd_dist=[0.08,0.24]
        self.dd_dist=[0.03, 0.15]
        # action space
        self.K=6
        
        self.offset=2e-10
            
        #idxNegative=np.where(self.image<0)
        #self.image[idxNegative]=self.image[idxNegative]*(-1)
        
        # for normalizing purpose           
        self.minMu,self.maxMu,self.minSig,self.maxSig=0,0.9,0,0.39
        
        #1.825628008326605e-12
        
        
        self.minCurrent=1.0037832448105646e-13
        #self.minCurrent=1.9e-10
        #self.maxCurrent=1.28198307779363e-09
        self.maxCurrent=1.02e-9

        #self.offset=-1.8862508749766253e-12
        
        # action space
        self.K=6
        
        WindowSz=30
        self.WindowSz=WindowSz
 
        # create an empty visit_map of [50x50]
        self.visit_map=np.zeros((WindowSz,WindowSz))
        self.isquantum=np.zeros((WindowSz,WindowSz))
        
        # start at the middle
        self.current_pos=np.copy([int(WindowSz/2),int(WindowSz/2)])
        self.starting_pos=[int(WindowSz/2),int(WindowSz/2)]
        self.initial_voltage=initial_gate_cu_cv # initial gate voltage of C8 and C12

        # base on this location, construct the data
        self.data=[0]*WindowSz
        self.data_ori=[0]*WindowSz

        self.data_img=[0]*WindowSz
        self.data_img_ori=[0]*WindowSz
        self.img_bigblock=[0]*WindowSz

        for ii in range(WindowSz):
            self.data[ii]=[0]*WindowSz
            self.data_ori[ii]=[0]*WindowSz
            self.data_img[ii]=[0]*WindowSz
            self.data_img_ori[ii]=[0]*WindowSz
            self.img_bigblock[ii]=[0]*WindowSz
            
        #self.data=np.copy(self.bigpatch_data)
        self.dim=[WindowSz,WindowSz]

        data=self.get_data(self.current_pos,pre_pos=None)
        
        
        self.data[self.current_pos[0]][self.current_pos[1]]=data
        #self.data_img_ori[25][25]=np.copy(img)
        #self.data_img[25][25]=(img-self.minCurrent)/(self.maxCurrent-self.minCurrent)
        
        #D is the dimension of each patch
        # self.dim is the dimension of blocks in the images
        self.D=len(self.data[self.current_pos[0]][self.current_pos[1]])
        
        self.visit_map[self.current_pos[0],self.current_pos[1]]=1
        
        done=self.is_state_quantum(data,self.data_img[self.current_pos[0]][self.current_pos[1]])     
        if done:
            self.isquantum[self.current_pos[0],self.current_pos[1]]=1
            print("Found Qubit")
            print(self.data[self.current_pos[0]][self.current_pos[1]])
            return
        
    
    def get_data(self,positions,pre_pos=None):
        # positions and previous positions
        
        gate_names=["c5","c9"]
        
        res_w=int(self.bw*1)
        res_h=int(self.bh*1)
        
        if pre_pos is not None:
            pre_id1,pre_id2=pre_pos
            
        id1,id2=positions  
        
        if id1<=self.starting_pos[0]: # decrease C5
            c5_val=self.initial_voltage[0]-self.bh*(self.starting_pos[0]-id1)
        else:
            c5_val=self.initial_voltage[0]+self.bh*(id1-self.starting_pos[0])
        
        if id2<=self.starting_pos[1]: # decrease C9
            c9_val=self.initial_voltage[1]+self.bw*(self.starting_pos[1]-id2)
        else:
            c9_val=self.initial_voltage[1]-self.bw*(id2-self.starting_pos[1])
        
        
        minc8=int(c5_val-self.bw)
        maxc8=int(c5_val+self.bw)
        minc12=int(c9_val-self.bh)
        maxc12=int(c9_val+self.bh)
        w_from=[minc8,minc12]
        w_to=[maxc8,maxc12]
        
        global counter    # Needed to modify global copy of globvar
        counter+= 1
        
        """
        temp = self.pygor.do2d(gate_names[1],minc12,maxc12,2*res_h,
                               gate_names[0],minc8,maxc8,2*res_w)
        temp.data[0]=np.asarray(temp.data[0])+self.offset
        data_2d_bk=temp.data[0]
        
        
        f=plt.figure()
        plt.imshow(data_2d_bk,extent=[ w_to[1],w_from[1], w_to[0],w_from[0]])
        strTitle="c5_{:.3f}_c9_{:.3f}".format(c5_val,c9_val)
        plt.xlabel("C9")
        plt.ylabel("C5")
        plt.title(strTitle)
        plt.colorbar()
        strPath="path/step_{:d}_c5_{:d}_c9_{:d}_bk.pdf".format(counter,np.int(-1*c5_val),np.int(-1*c9_val))
        f.savefig(strPath)
        f.clf()
        plt.close()
        gc.collect()
        """
        
        pre_pos=None
        if pre_pos is None: # full scane
            minc8=int(c5_val-self.bw)
            maxc8=int(c5_val+self.bw)
            minc12=int(c9_val-self.bh)
            maxc12=int(c9_val+self.bh)
            
            temp = self.pygor.do2d(gate_names[1],minc12,maxc12,2*res_h,
                                   gate_names[0],minc8,maxc8,2*res_w)#extra 1 point
            
            data_2d=np.asarray(temp.data[0])#+self.offset
            #data_2d=data_2d[:,:-1]
        else:
            data_2d=self.fill_data_overlap(current_loc=[id1,id2],c5_val=c5_val,c9_val=c9_val)
        
            
        #data_2d = self.pygor.do2d("c8",c8_val-self.bw,c8_val+self.bw,2*self.bw,
                                    #"c12",c12_val-self.bh,c12_val+self.bh,2*self.bh)
        
        #img_2d_ori=data_2d+self.offset
        img_2d_ori=data_2d
        
        print("c5 [",minc8,",",maxc8,"]","c9 [",minc12,",",maxc12,"]")
        f=plt.figure()
        plt.imshow(img_2d_ori,extent=[ w_to[1],w_from[1], w_to[0],w_from[0]])
        strTitle="c5_{:.3f}_c9_{:.3f}".format(c5_val,c9_val)
        plt.xlabel("C9")
        plt.ylabel("C5")
        plt.title(strTitle)
        plt.colorbar()
        
        
        strPath="path/step_{:d}_c5_{:d}_c9_{:d}.pdf".format(counter,np.int(-1*c5_val),np.int(-1*c9_val))
        f.savefig(strPath)
        f.clf()
        plt.close()
        gc.collect()   

        # clip to zero
        #idx = np.where(img_2d_ori<0)
        #img_2d_ori[idx]=self.minCurrent
        
        self.img_bigblock[id1][id2]=np.copy(img_2d_ori)


        # normalize data_2d: 0->1
        img_2d_ori=np.copy(img_2d_ori)+self.offset

        img_2d_ori=np.clip(img_2d_ori, self.minCurrent, self.maxCurrent)
        
        """
        if np.min(img_2d_ori)<self.minCurrent:
            print("np.min(data_2d)<self.minCurrent",np.min(img_2d_ori))
            self.minCurrent=np.min(img_2d_ori)
            
        if np.max(img_2d_ori)>self.maxCurrent:
            print("np.max(data_2d)>img_2d_ori.maxCurrent",np.max(img_2d_ori))
            self.maxCurrent=np.max(img_2d_ori)
        """
        
        #img_2d=(img_2d_ori-self.minCurrent)/(self.maxCurrent-self.minCurrent)
        
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
        img=[0]*len(pp_blocks)
        img_ori=[0]*len(pp_blocks)
        #temp_patch=[]
        #temp_patch_ori=[]
        for kk,mypp in enumerate(pp_blocks): # 27 items
            #print(kk)
            temp_ori=img_2d_ori[mypp[0][0]:mypp[0][1],mypp[1][0]:mypp[1][1]]
            
            temp=(temp_ori-self.minCurrent)/(self.maxCurrent-self.minCurrent)

            mymean=np.mean(temp)/(self.maxMu-self.minMu)
            mystd=np.std(temp)/(self.maxSig-self.minSig)
            mymean_ori=np.mean(temp)
            mystd_ori=np.std(temp)
            
            #temp_patch+=[mymean,mystd]  
            #temp_patch_ori+=[mymean_ori,mystd_ori]  

            data[kk]=np.copy([mymean,mystd])
            data_ori[kk]=np.copy([mymean_ori,mystd_ori]  )
            img[kk]=np.copy(temp)
            img_ori[kk]=np.copy(temp_ori)

            
        self.data[id1][id2]=np.copy(data)
        self.data_ori[id1][id2]=np.copy(data_ori)
        self.data_img[id1][id2]=img
        self.data_img_ori[id1][id2]=img_ori
        
        f, axarr  = plt.subplots(3,3,figsize=(6,6))
        f.tight_layout() # Or equivalently,  "plt.tight_layout()"

        for ii in range(3):
            for jj in range(3):
                axarr[ii,jj].imshow(img[ii*3+jj],vmin=0,vmax=0.6) 
                stTitle="{:.1f}_{:.1f}".format(100*data[ii*3+jj][0],100*data[ii*3+jj][1])

                axarr[ii,jj].set_title(stTitle)
                
        
        strPath="path/step_{:d}_blocks_c5_{:d}_c9_{:d}.pdf".format(counter,np.int(-1*c5_val),np.int(-1*c9_val))
        f.savefig(strPath)
        f.clf()
        plt.close()
        gc.collect()   
        
        self.plot_full_image(counter)
        
        return data
        
        
    def get_state_and_location(self):
        
        #self.current_pos=np.copy(self.starting_loc)
		
        id1,id2=self.current_pos
        #self.visit_map=np.zeros_like(self.reward_table)

        return np.reshape(self.data[id1][id2],(-1,2*9)),np.copy(self.current_pos)
        #return self.add_another_dim(self.data[id1][id2][id3]),np.copy(self.current_pos)
    
    def get_state(self,positions):
        id1,id2=positions  
        if self.data[id1][id2]==0: # not measured yet
            data=self.get_data(self.current_pos,pre_pos=None)
            
            self.data[id1][id2]=data
        return  np.reshape(self.data[id1][id2],(-1,9*2))

        
    def current_state(self):
        id1,id2=self.current_pos      
        #return  self.add_another_dim(self.data[irow][icol])
        #return  np.reshape(self.data[id1][id2],(-1,2*9))
        return  np.reshape(self.data[id1][id2],(9,2))

		
    def add_another_dim(self,myinput):
        return np.expand_dims(myinput,axis=0)

    def get_neightborMapIndividual(self,location):
    
        id1,id2=location
    
        output=[]
        # return a 6 dimensional vector
        norm_factor=5.0
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
            
        if id1>0 and id2<self.dim[1]-1: # decrease d1 and decrease d2
            output.append(self.visit_map[id1-1,id2+1]/norm_factor)
        else:
            output.append(0)

            
        if id1<self.dim[0]-1 and id2>0: # increase d1 and increase d2
            output.append(self.visit_map[id1+1,id2-1]/norm_factor)
        else:
            output.append(0)
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
        
        pres_pos=np.copy(self.current_pos)
        if action==0:
            print("action",action, "Up (Decrease C5)")

            if self.current_pos[0]==0:
                flagoutside=1
                #print("cannot decrease d1")
            else:
                self.current_pos[0]=self.current_pos[0]-1
        elif action==1:
            print("action",action, "Down (Increase C5)")
            
            if self.current_pos[0]==self.dim[0]-1:
                flagoutside=1
                #print("cannot increase d1")
            else:
                self.current_pos[0]=self.current_pos[0]+1
        elif action==2:
            print("action",action, "Left (Increase C9)")
            
            if self.current_pos[1]==0:
                flagoutside=1
                #print("cannot decrease d2")
            else:
                self.current_pos[1]=self.current_pos[1]-1
        elif action==3:
            print("action",action, "Right (Decrease C9)")
            
            if self.current_pos[1]==self.dim[1]-1:
                flagoutside=1
                #print("cannot decrease d2")
            else:
                self.current_pos[1]=self.current_pos[1]+1
        elif action==4:
            print("action",action, "Up Right (Decrease C9 Decrease C5)")

            if self.current_pos[0]>0 and self.current_pos[1]<self.dim[1]-1:
                self.current_pos[1]=self.current_pos[1]+1
                self.current_pos[0]=self.current_pos[0]-1
            else:
                flagoutside=1
                #print("cannot decrease both d1 and d2")

        elif action==5:
            print("action",action, "Left Down (Increase C9 Increase C5)")
            
            if self.current_pos[0]<self.dim[0]-1 and self.current_pos[1]>0:
                self.current_pos[1]=self.current_pos[1]-1
                self.current_pos[0]=self.current_pos[0]+1
            else:
                flagoutside=1
                #print("cannot increase both d1 and d2")

        else:
            print("action is 0-6")
   
       
        id1,id2=self.current_pos
        
        
        self.visit_map[id1,id2]+=1

        done=False
  
        if self.data[id1][id2]==0: # not measured yet
            data=self.get_data(self.current_pos,pres_pos)
            
            self.data[id1][id2]=data
            #self.data_img_ori[id1][id2]=np.copy(img)
            #self.data_img[id1][id2]=(img-self.minCurrent)/(self.maxCurrent-self.minCurrent)
            
        obs=self.data[id1][id2]
        
        done=self.is_state_quantum(obs,self.data_img[id1][id2])     
        if done:
            self.isquantum[id1,id2]=1
            print("Found Qubit")
        
        loc_x=np.copy(self.current_pos)
    
        return obs, done, loc_x
    
    def Sym_KL(self, a, b):
        eps=1e-8
        a = np.asarray(a, dtype=np.float)+eps
        b = np.asarray(b, dtype=np.float)+eps
    
        return 0.5*np.sum(np.where(a != 0, a * np.log(a / b), 0)) + 0.5*np.sum(np.where(b != 0, b * np.log(b /a), 0))

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
        # convert to original scale

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
        
    def is_state_quantum(self,state,image):
        
        countQ,countB=0,0
        for uu in range(9): # 3**len(self.dim) = 9
            #predicted_value=self.cnnmodel.predict(obs[uu])
            isQuantum,scoreQ=self.predict_scoreQuantum(state[uu])             
            isBarrier,scoreB=self.predict_scoreBarrier(state[uu])             
            
            #print(scoreQ,state[uu])
            
            if isQuantum is True:
                stdZero,stdOne=self.compute_entropy_block(image[uu])
                
                print(stdOne)
                
                if  np.max(stdOne)<6 and np.min(stdOne)>0.8:

                    #sumscore+=score
                    countQ+=1
            if isBarrier is True:
                countB+=1
                
        if (countQ==1 or countQ==2 or countQ==3) and countB>=5:
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

