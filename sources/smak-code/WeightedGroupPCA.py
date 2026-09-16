#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:24:52 2023

@author: samwebb
"""


import tkinter
import Pmw

import numpy as np
import packaging.version as VRS
import sklearn
import sblite
import time

import globalfuncs
from MasterClass import MasterClass
import MyGraph
import PCAAnalysisMathClass
import PmwTtkButtonBox



#check sklHasFactorAnalysis
if VRS.Version(sklearn.__version__)>VRS.parse("0.13.0"):
    print ('sklearn version > 0.13.0')
    from sklearn.decomposition import FactorAnalysis
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    sklHasFactorAnalysis=True
else:
    sklHasFactorAnalysis=False

#check sklHasAdvancedCluster
if VRS.Version(sklearn.__version__)>=VRS.parse("0.17.0"):
    print ('sklearn version > 0.17.0')
    from sklearn import cluster as skcluster
    from sklearn import mixture as skmixture
    try:
        import hdbscan
    except:
        hdbscan = None
        print("hdbscan not installed")
    sklHasAdvancedCluster=True
else:
    sklHasAdvancedCluster=False

#######################################
##  Group Class
#######################################        
        
class wtChannelGroup:
    def __init__(self,master,labels,index,extend=True):
        
        self.index=index
        g1=Pmw.Group(master,tag_text='Group '+str(index),tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        if extend:
            mode=tkinter.EXTENDED
        else:
            mode=tkinter.SINGLE
        self.grpch=Pmw.ScrolledListBox(g1.interior(),labelpos='n',label_text='Select Channel',items=labels,listbox_selectmode=mode,
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=tkinter.DISABLED,listbox_height=5,
                                           hull_background='#d4d0c8',label_background='#d4d0c8')
        self.grpch.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')

        l = 'Weighting'
        sv=1
        self.wtCFvar=tkinter.Scale(g1.interior(),label=l,background='#d4d0c8',width=20,length=100,from_=0,to=1000,orient=tkinter.VERTICAL,resolution=0.1,command=tkinter.DISABLED)
        self.wtCFvar.pack(side=tkinter.TOP,fill='both',expand=1,padx=5,pady=10)
        self.wtCFvar.set(sv)
            
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')
        

class GroupPCAParams:
    
    def __init__(self,maindisp,PCAcompMAXFixed,PCAcompMAXNO,status,dataFileBuffer,activeFileBuffer,addchannel,saveEvectFromWtPCA):
        
        self.status=status
        self.dataFileBuffer=dataFileBuffer
        self.activeFileBuffer=activeFileBuffer
        self.PCAcompMAXFixed=PCAcompMAXFixed
        self.PCAcompMAXNO=PCAcompMAXNO
        self.maindisp=maindisp
        self.addchannel=addchannel
        self.saveEvectFromWtPCA=saveEvectFromWtPCA

class GroupPCA(MasterClass):

    def _create(self):
    
        self.win = Pmw.MegaToplevel(self.imgwin)
        self.win.title('Weighted Group PCA Window')
        self.win.userdeletefunc(func=self.kill)    
 
        self.PCAlastevect=None
        self.PCAlastprop=None
        self.PCAlastchans=None   

        if sklHasFactorAnalysis:
            analysisOptions=['sPCA','CCIPCA','FA','NMF',"FastICA","SiVM",'Dictionary','LDA','Kmeans','Cancel']
        else:
            analysisOptions=['sPCA','CCIPCA','NMF',"FastICA","SiVM",'Kmeans','Cancel']
        if sklHasAdvancedCluster:
            analysisOptions.pop(analysisOptions.index('Kmeans'))
            analysisOptions.pop(analysisOptions.index('Cancel'))
            manifoldOptions=['Iso','MDS','tSNE']
            clusterOptions=['Kmeans','Gaussian','Cancel'] 
            if False:
                analysisOptions.extend(manifoldOptions)
            analysisOptions.extend(clusterOptions)

        print("self.mapdata.labels: ",self.mapdata.labels)
        print("done")
    
        self.WeightPCADialog=self.win
        
        h=self.WeightPCADialog.interior()
        h.configure(background='#d4d0c8')
        
        self.gpgroup=[]
        self.groupMaster=Pmw.ScrolledFrame(h,hscrollmode='dynamic',vscrollmode='static',usehullsize=1,hull_width=520,hull_height=500,hull_background='#d4d0c8')
        self.groupMaster.pack(side=tkinter.TOP,padx=1,pady=1,expand='yes',fill='both')
        self.groupMaster.component("frame").configure(background='#d4d0c8')
        self.groupMaster.component("clipper").configure(background='#d4d0c8')
        gfm=self.groupMaster.interior()      
        
        self.gpgroup.append(wtChannelGroup(gfm,self.mapdata.labels,1))
        self.gpgroup.append(wtChannelGroup(gfm,self.mapdata.labels,2)) 
        
        #analysis type
        self.gppcatype = Pmw.ComboBox(h,history=0,selectioncommand=tkinter.DISABLED,hull_background='#d4d0c8',
                                      labelpos='w',label_text="PCA Type: ",label_background='#d4d0c8',hull_width=50) 
        self.gppcatype.pack(side=tkinter.TOP,padx=3,pady=2)  
        self.gppcatype.setlist(analysisOptions)
        
        #multifile checkbox
        self.mfileopt=Pmw.RadioSelect(h,labelpos='w',command=tkinter.DISABLED,label_text='MultiFile Option:    ',buttontype='radiobutton')
        self.mfileopt.add('Single File')
        self.mfileopt.add('All Files')
        self.mfileopt.invoke('Single File')      
        self.mfileopt.pack(side=tkinter.TOP,padx=5,pady=2)
        
        #graph
        self.pcaplot=MyGraph.MyGraph(h,whsize=(3,3),padx=5,pady=5,graphpos=[[.15,.1],[.9,.9]])
        
        #preview button
        b=PmwTtkButtonBox.PmwTtkButtonBox(h,hull_background='#d4d0c8')
        b.add('Preview',command=self.doWtPCAPreview,style='GREEN.TButton',width=10)
        b.add('AutoBalance',command=self.doWtPCABalance,style='BROWN.TButton',width=10)
        b.add('Save PCA',command=self.doWtPCASave,style='SBLUE.TButton',width=10)
        b.add('Add Group',command=self.doWtPCAAdd,style='ORANGE.TButton',width=10)
        b.pack(side=tkinter.TOP,padx=5,pady=5)
        self.WeightPCADialog.show()        
        
       
    def doWtPCAAdd(self):
        n=len(self.gpgroup)+1
        self.gpgroup.append(wtChannelGroup(self.groupMaster.interior(),self.mapdata.labels,n)) 
 
    def checkMultiNames(self,name,iters):
        for nbuf in iters:
            if name in self.ps.dataFileBuffer[nbuf]['data'].labels:
                return False
        return True   
        
    def checkValid(self):
        if self.gppcatype.get()=='': return False
        #check each group
        i=1
        self.masterchans=[]
        self.groupchans={}
        self.changain={}
        for grp in self.gpgroup:
            if len(grp.grpch.getcurselection())==0:
                globalfuncs.setstatus(self.ps.status,'Channels for Group '+str(i)+' is empty')
                return False
            self.masterchans.extend(grp.grpch.getcurselection())
            self.groupchans[i]=grp.grpch.getcurselection()
            for c in grp.grpch.getcurselection():
                self.changain[c]=float(grp.wtCFvar.get())
            i+=1            
        #print (self.changain)

        ptype = self.mfileopt.getcurselection()
        if ptype=='Single File':
            return True
        for buf in list(self.ps.dataFileBuffer.values()):
            for c in self.masterchans:
                if c not in buf['data'].labels:
                    print(c,'missing in',buf['name'])
                    return False
        return True        
    
    def doWtPCASave(self):
        self.doWtPCAPreview()        
        #save
        print('begin export')
        globalfuncs.setstatus(self.ps.status,"Exporting component weights to map dataset...")

        if self.mfileopt.getcurselection()=='Single File':
            iters=[self.ps.activeFileBuffer]
            npost='GWSF'
        else:
            iters=list(self.ps.dataFileBuffer.keys())
            npost='GWMF'

        if self.gppcatype.get() not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian']:           

            cind=0
            for i in range(self.PCAdataStruct.PCAprop.shape[1]):
                noexit=1
                name=self.gppcatype.get()+npost+'Comp'
                while noexit:
                    cind+=1
                    chname=name+str(cind)
                    if self.checkMultiNames(chname,iters): # not in self.mapdata.labels:
                        noexit=0
                name=name+str(cind)
                dataFull=self.PCAdataStruct.PCAprop[:,i]
                ##print data
                startindex=0
                for nbuf in iters:
                    info=self.PCAdatafileInfo[nbuf]
                    data=np.array(dataFull[startindex:startindex+info['len']])
                    startindex+=info['len']
                    if info['zoom'][0:4]!=[0,0,-1,-1]:
                        nd=np.zeros(info['shape'],dtype=np.float32)
                        pm=nd[info['zoom'][1]:info['zoom'][3],info['zoom'][0]:info['zoom'][2]]
                        #data=self.PCArawdata[0,:]
                        data=data[:len(np.ravel(pm))]
                        ##print "prop",i,sum(data)
                        data=np.reshape(data,pm.shape)
                        #data=ones(pm.shape)
                        nd[info['zoom'][1]:info['zoom'][3],info['zoom'][0]:info['zoom'][2]]=data
                        data=nd[::-1,:]
                    else:
                        data=data[:info['shape'][0]*info['shape'][1]] #len(ravel(self.mapdata.data.get(0)))]
                        ##print "prop",i,sum(data)
                        ##print data.shape,self.mapdata.data.get(0).shape
                        data=np.reshape(data,(info['shape']))
                        data=data[::-1,:]
                    self.ps.addchannel(data,name,fbuffer=nbuf)

        if self.gppcatype.get() in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian']:
            cind=0
            noexit=1
            name=self.gppcatype.get()+npost+'Clusters'
            while noexit:
                cind+=1
                chname=name+str(cind)
                if self.checkMultiNames(chname,iters): # not in self.mapdata.labels:
                    noexit=0
            name=name + str(cind)
            dataFull=self.PCAdataStruct.PCAKcluster
            ##print data
            startindex=0
            for nbuf in iters:
                info=self.PCAdatafileInfo[nbuf]
                data=np.array(dataFull[startindex:startindex + info['len']])
                startindex +=info['len']
                if info['zoom'][0:4]!=[0,0,-1,-1]:
                    nd=np.zeros(info['shape'],dtype=np.float32)
                    pm=nd[info['zoom'][1]:info['zoom'][3],info['zoom'][0]:info['zoom'][2]]
                    #data=self.PCArawdata[0,:]
                    data=data[:len(np.ravel(pm))]
                    ##print "prop",i,sum(data)
                    data=np.reshape(data,pm.shape)
                    #data=ones(pm.shape)
                    nd[info['zoom'][1]:info['zoom'][3],info['zoom'][0]:info['zoom'][2]]=data
                    data=nd[::-1,:]
                else:
                    data=data[:info['shape'][0]*info['shape'][1]] #len(ravel(self.mapdata.data.get(0)))]
                    ##print "prop",i,sum(data)
                    ##print data.shape,self.mapdata.data.get(0).shape
                    data=np.reshape(data,(info['shape']))
                    data=data[::-1,:]
                self.addchannel(data,name,fbuffer=nbuf)


        if self.gppcatype.get() in ['sPCA','CCIPCA','FA','NMF',"FastICA","SiVM"]:
            self.ps.saveEvectFromWtPCA(self.PCAlastevect,self.PCAlastprop,self.PCAlastchans)
            
            
        
    def doWtPCAPreview(self):
        #check valid
        if not self.checkValid():
            return
        
        self.PCAlastevect=None
        self.PCAlastprop=None
        self.PCAlastchans=None
        
        #assemble data
        pcadata=[]
        self.PCAdatafileInfo={}
        if self.mfileopt.getcurselection()=='Single File':
            iters=[self.ps.activeFileBuffer]
            self.ps.dataFileBuffer[self.ps.activeFileBuffer]['zoom']=self.ps.maindisp.zmxyi
        else:
            iters=list(self.ps.dataFileBuffer.keys())
        for c in self.masterchans:
            ndfin=[]
            for nbuf in iters:
                buf=self.ps.dataFileBuffer[nbuf]
                dataind=buf['data'].labels.index(c)+2
                #worry about zooms
                dr=buf['data'].data.get(dataind)[::-1,:]#[::-1,:,dataind]
                ##and masks???
                ##if len(self.mask.mask)!=0 and self.usemaskinimage:
                ##    dr=self.mask.mask[::-1,1]*dr
                if buf['zoom'][0:4]!=[0,0,-1,-1]:
                    dr=dr[buf['zoom'][1]:buf['zoom'][3],buf['zoom'][0]:buf['zoom'][2]]
                nd=np.ravel(dr)*float(self.changain[c])
                info={}
                info['len']=len(nd)
                info['zoom']=buf['zoom']
                info['shape']=buf['data'].data.get(0).shape
                self.PCAdatafileInfo[nbuf]=info
                print(c,sum(nd))
                ndfin.extend(nd)
            pcadata.append(ndfin)
        pcarawdata=np.array(pcadata,dtype=np.float64)
        if not self.ps.PCAcompMAXFixed:
            PCAcompMAXNO=pcarawdata.shape[0]
        else:
            PCAcompMAXNO=self.ps.PCAcompMAXNO
        pcarawdata=np.transpose(pcarawdata)
        print ('ERRCH: ',self.ps.maindisp.zmxyi)        
        self.PCAdataStruct = PCAAnalysisMathClass.PCADataStructure(pcarawdata,PCAcompMAXNO,self.imgwin,
                                                                   pcaft=self.mfileopt.getcurselection(),
                                                                   dx = self.mapdata.data.get(0)[::-1,:],
                                                                   dy = self.mapdata.data.get(1)[::-1,:],
                                                                   zmxyi= self.ps.maindisp.zmxyi)
        
        globalfuncs.setstatus(self.ps.status,"WORKING ON PCA")        
        #try new pca
        ntm=[]
        ntm.append(time.process_time())
        self.PCAdataStruct.donewPCA(pcatype=self.gppcatype.get())
        ntm.append(time.process_time())
        print('PCA complete in '+str(ntm[1]-ntm[0])+' seconds')

        self.grpnorms=[]
        if self.gppcatype.get() in ['sPCA','CCIPCA','FA','NMF',"FastICA","SiVM"]: #if result not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','SiVM','Gaussian']:
            
            evnorm = np.sum(abs(self.PCAdataStruct.PCAevect),axis=0)

            i=0
            for iv,ch in self.groupchans.items():
                self.grpnorms.append(np.sum(evnorm[i:i+len(ch)])/len(ch))
                i+=len(ch)
        
        if self.gppcatype.get() not in ['AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','LDA','SiVM','Gaussian','Dictionary','Iso','MDS','tSNE',]:
            globalfuncs.setstatus(self.ps.status,"Checking for negative eigenvectors")
            for i in range(self.PCAdataStruct.PCAprop.shape[1]):
                dmax=max(self.PCAdataStruct.PCAevect[i,:])
                dmin=min(self.PCAdataStruct.PCAevect[i,:])
                if abs(dmin)>abs(dmax):
                    #need to inverse
                    self.PCAdataStruct.PCAevect[i,:]=-self.PCAdataStruct.PCAevect[i,:]
                    #need to adjust wt matrix:
                    z=np.transpose(self.PCAdataStruct.PCAprop)
                    t=z[i,:]
                    z[i,:]=-t
                    self.PCAprop=np.transpose(z)
            print('sizecheck VDX',self.PCAdataStruct.PCAevect.shape,self.PCAdataStruct.PCAeval.shape,self.PCAdataStruct.PCAprop.shape)

        if self.gppcatype.get() not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian']:         
            self.PCAlastevect=self.PCAdataStruct.PCAevect
            self.PCAlastprop=self.PCAdataStruct.PCAprop
            self.PCAlastchans=self.masterchans

        #print the vectors
        #if len(self.PCAdataStruct.PCAprop.shape) > 1:
        #    for i in range(self.PCAdataStruct.PCAprop.shape[1]):
        #        print("EV#"+str(i),self.PCAdataStruct.PCAevect[i,:])
                
        self.plotPCAplotVectors()        
        #print group balances
        for i,j in enumerate(self.grpnorms):
            print ('Group:',i,j)

        globalfuncs.setstatus(self.ps.status,"Done!")
        print('done')            
            

    
    def doWtPCABalance(self):
        cycles = 3
        for pcac in range(cycles):        
            print ('cycle',pcac,' started...')
            self.doWtPCAPreview()  
            # now balance by PCA norms
            cd = max(self.grpnorms)
            adjfact = float(cd)/np.array(self.grpnorms)
            
            i=0
            for g in self.gpgroup:
                nv = float(g.wtCFvar.get())*adjfact[i]
                g.wtCFvar.set(nv)
                i+=1
            print ('cycle',pcac,' complete...')
    
        
    def plotPCAplotVectors(self):  
        print("test in replotPCAplotVectors")
        if self.PCAlastevect is None: return
        #clear old
        self.pcaplot.cleargraphs()
        #make graphs
        palette=sblite.color_palette('hls', n_colors=8)
        colors=palette.as_hex()  # ['blue', 'red', 'green', 'white', 'orange','magenta', 'cyan','brown']
        print('PCApropSp',self.PCAlastprop.shape[1])
        print('PCAlastev',self.PCAlastevect.shape)
        for i in range(self.PCAlastprop.shape[1]):
            yv=self.PCAlastevect[i,:]
            xv=list(range(len(yv)))
            self.pcaplot.plot(tuple(xv),tuple(yv),color=colors[i%len(colors)],text='EV'+str(i))
        self.pcaplot.uselegend(True)
        self.pcaplot.draw()
        
        