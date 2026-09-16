#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:01:58 2023

@author: samwebb
"""

#system imports
import math
import sys
import os
import fnmatch
import time 
import h5py

#third party
import numpy as np
import scipy
from PyMca5.PyMcaIO import ConfigDict as PyCD
from PyMca5.PyMcaPhysics.xrf import Elements as pEle

#local
import globalfuncs
import PCAAnalysisMathClass
from PCAAnalysisClass import pcacompobj
import XanesFitClass
import MultiMassCalibrationClass
import pyMcaFitWrapper

import IR_MathClass


def timeFormat(s):
    return str(s).zfill(2)

def defineMCAfile(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    if type(ps['filename'])!=dict:
        print ('invalid dictionary type')
        return
    fb['mcafn']=ps['filename'][fbn]
    getMCAMetadata(fbl)

def setMCAxraySlope(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    fb['MCAxrayslope']=ps['slope']

def getMCAMetadata(fbl,**kw):
    [fbn,fb]=fbl
    if os.path.splitext(fb['mcafn'])[1] != '.hdf5':
        return
    
    fid=h5py.File(fb['mcafn'])
    if "/main/mcadata" in fid:
        mcadata=fid['/main/mcadata']
    elif "/main/oodata" in fid:
        mcadata=fid['/main/oodata']
    else:
        print('no mcadata found')
        return    
    mcamaxno = mcadata.shape[1]

    if "/main/mcadatacv" in fid:
        fb['MCAxvalues'] = fid['/main/mcadatacv']
    else:
        fb['MCAxvalues'] = np.arange(mcamaxno)*fb['MCAxrayslope']

    
def intMCArange(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    #print (int(ps['binmin']),int(ps['binmax']),int(ps['binint']))
    intvals = list(range(int(ps['binmin']),int(ps['binmax']),int(ps['binint'])))
    if len(intvals)<2:
        print('Inappropriate range entered')
        return
    bvals=intvals.copy()
    tvals=intvals.copy()
    tvals.pop(0)
    tvals.append(int(ps['binmax']))
    for b,t in zip(bvals,tvals):
        
        noexit=1
        name=str(b)+"."+str(t)
        cind=''
        while noexit:
            chname=name+str(cind)
            if chname not in fb['data'].labels: 
                noexit=0
            if noexit:
                if cind=='': cind=0
                cind+=1
        if cind!='':cind='.'+str(cind)
        newname=name+str(cind)                        
        
        MCARebinning(fbl,b,t,newname,ps['addchan'],fbuffer=fbn)


def intMCAbyBin(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    MCARebinning(fbl, int(ps['binmin']), int(ps['binmax']), ps['name'], ps['addchan'])
    
def intMCAbyValue(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    binmin=globalfuncs.find_nearest(fb['MCAxvalues'],int(ps['valmin']))
    binmax=globalfuncs.find_nearest(fb['MCAxvalues'],int(ps['valmax']))    

    MCARebinning(fbl, binmin, binmax, ps['name'], ps['addchan'])

def MCARebinning(fbl,bmin,bmax,name,addchan):
    [fbn,fb]=fbl
    newdata=[]

    #go thru MCA file and integrate
    (fn,ext)=os.path.splitext(fb['mcafn'])
    if ext!=".hdf5":
        print ('please convert to hdf5')
        return

    #have hdf5 data...
    fid=h5py.File(fb['mcafn'])
    if "/main/mcadata" in fid:
        mcadata=fid['/main/mcadata']
    elif "/main/oodata" in fid:
        mcadata=fid['/main/oodata']
    else:
        print('no mcadata found')
        return
    mcamaxno=mcadata.shape[1]
    maxlines=mcadata.shape[0]
    print('hdf',mcamaxno,maxlines)
    
    newdata=np.zeros(fb['data'].nxpts*fb['data'].nypts,dtype=np.float32)
    #MCApixoffs    MCA1stpixoffs
    data=np.sum(mcadata[:,bmin:bmax],axis=1)
    data=data.astype(float)
    
    if len(newdata)<len(data):
        newdata=data[:len(newdata)-1]
    elif len(newdata)>len(data):
        newdata[:len(data)]=data
    else:
        newdata=data
    fid.close()

    if fb['data'].nypts*fb['data'].nxpts>len(newdata): #expand
        while len(newdata)!=fb['data'].nypts*fb['data'].nxpts:
            newdata=np.append(newdata,0)
    newdata=np.array(newdata)
    if fb['data'].nypts*fb['data'].nxpts<len(newdata):
        newdata=np.reshape(newdata[:fb['data'].nypts*fb['data'].nxpts-1],(fb['data'].nypts,fb['data'].nxpts))
    else:
        newdata=np.reshape(newdata,(fb['data'].nypts,fb['data'].nxpts))

    #place new data into main data
    newname=name
    i=1
    while newname in fb['data'].labels:
        newname=name+str(i)
        i+=1    
    addchan(newdata,newname,fbuffer=fbn)
    
def savePCAResult(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']

    [PCAlastevect,PCAlastprop,PCAlastchans] = fb['lastPCAresult']
    if PCAlastevect is None: 
        print ('No recent PCA data')
        return
    #get file name
    t=time.localtime()
    timestamp = timeFormat(t.tm_year)+'_'+timeFormat(t.tm_mon)+'_'+timeFormat(t.tm_mday)+'_'+timeFormat(t.tm_hour)+'_'+timeFormat(t.tm_min)+'_'+timeFormat(t.tm_sec)
    outfn=fb['fname']+'_PCAexportData_'+timestamp+'.out'
    print (outfn)
    #create text
    evtext = PCAlastevect.astype(np.float32).tobytes()
    proptext = PCAlastprop.astype(np.float32).tobytes()
    filetext = '####@PCA####\n@EV\n'
    filetext+= str(PCAlastevect.shape)+'\n'
    filetext+=str(evtext)+'\n'
    filetext+= '@PROP\n'
    filetext+= str(PCAlastprop.shape)+'\n'
    filetext+=str(proptext)+'\n'
    filetext+= '@CHAN\n'
    filetext+= str(len(PCAlastchans))+'\n'
    filetext+= '\n'.join(map(str,PCAlastchans))
    filetext+= '\n####END####\n'
    #write and close
    fid=open(outfn,"w")   
    fid.write(filetext)
    fid.close()    
    
def PCAGeneral(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    print (ps)
    #assemble chans from regex
    chans=[]
    if 'regex' in ps:
        chans = fnmatch.filter(fb['data'].labels,ps['regex'])
    else:
        chans = fb['data'].labels.copy()

    if len(chans)==0:
        print ('no data channels selected...')
        return


    if not checkMultiPCA(ps['dataFB'], chans, ps['checkbox']):
        print('Missing channels, PCA cancelled')
        return
    pcadata=[]
    PCAdatafileInfo={}
    if ps['checkbox']=='Single File':
        iters=[fbn]
    else:
        iters=list(ps['dataFB'].keys())
    for c in chans:
        ndfin=[]
        for nbuf in iters:
            buf=ps['dataFB'][nbuf]
            dataind=buf['data'].labels.index(c)+2
            #worry about zooms
            dr=buf['data'].data.get(dataind)[::-1,:]#[::-1,:,dataind]
            ##and masks???
            ##if len(self.mask.mask)!=0 and self.usemaskinimage:
            ##    dr=self.mask.mask[::-1,1]*dr
            if buf['zoom'][0:4]!=[0,0,-1,-1]:
                dr=dr[buf['zoom'][1]:buf['zoom'][3],buf['zoom'][0]:buf['zoom'][2]]
            nd=np.ravel(dr)
            info={}
            info['len']=len(nd)
            info['zoom']=buf['zoom']
            info['shape']=buf['data'].data.get(0).shape
            PCAdatafileInfo[nbuf]=info
            print(c,sum(nd))
            ndfin.extend(nd)
        pcadata.append(ndfin)
    pcarawdata=np.array(pcadata,dtype=np.float64)
    ##print "PCA data: ",self.PCArawdata.shape
    PCAcompMAXNO=pcarawdata.shape[0]
    cl=int(ps['comps'])
    ml=None
    if 'thresh' in ps:
        ml=float(ps['thresh'])
    pcarawdata=np.transpose(pcarawdata)
    #print ('ERRCH: ',fb['zoom'])
    PCAdataStruct = PCAAnalysisMathClass.PCADataStructure(pcarawdata,PCAcompMAXNO,None,
                                                               pcaft=ps['checkbox'],
                                                               dx = fb['data'].data.get(0)[::-1,:],
                                                               dy = fb['data'].data.get(1)[::-1,:],
                                                               zmxyi= fb['zoom'],cl=cl,ml=ml)
    
    print("WORKING ON PCA")        
    #try new
    ntm=[]
    ntm.append(time.process_time())
    PCAdataStruct.donewPCA(pcatype=ps['oper'])
    ntm.append(time.process_time())
    print('PCA complete in '+str(ntm[1]-ntm[0])+' seconds')
    #compute sum
    if 'PCA' in ps['oper']: #if result not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','SiVM','Gaussian']:
        esum=sum(PCAdataStruct.PCAeval)
        ecsum=[]
        rateind=[]
        evalsq=PCAdataStruct.PCAeval*PCAdataStruct.PCAeval
        for i in range(len(PCAdataStruct.PCAeval)):
            ecsum.append(sum(PCAdataStruct.PCAeval[0:i+1]))
            temp=sum(evalsq[i+1:len(PCAdataStruct.PCAeval)])
            try:
                div=(len(PCAdataStruct.PCAeval)-(i+1))**5
            except:
                div=0
            if div!=0:
                rateind.append(math.sqrt(temp/div))
            else:
                rateind.append(0)
        ecsum=np.array(ecsum)
        varcomp=PCAdataStruct.PCAeval/esum
        varexp=ecsum/esum
    if 'PCA' in ps['oper']: print("IND values: ",rateind)
    print("PCA Analysis complete")

    if ps['oper'] not in ['AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','LDA','SiVM','Gaussian','Dictionary','Iso','MDS','tSNE',]:
        print("Checking for negative eigenvectors")
        for i in range(PCAdataStruct.PCAprop.shape[1]):
            dmax=max(PCAdataStruct.PCAevect[i,:])
            dmin=min(PCAdataStruct.PCAevect[i,:])
            if abs(dmin)>abs(dmax):
                #need to inverse
                PCAdataStruct.PCAevect[i,:]=-PCAdataStruct.PCAevect[i,:]
                #need to adjust wt matrix:
                z=np.transpose(PCAdataStruct.PCAprop)
                t=z[i,:]
                z[i,:]=-t
                PCAprop=np.transpose(z)
        print('sizecheck VDX',PCAdataStruct.PCAevect.shape,PCAdataStruct.PCAeval.shape,PCAdataStruct.PCAprop.shape)
    print('begin export')
    print("Exporting component weights to map dataset...")

    if ps['checkbox']=='Single File':
        iters=[fbn]
        npost='SF'
    else:
        iters=list(ps['dataFB'].keys())
        npost='MF'

    PCAlastevect=None
    PCAlastprop=None
    PCAlastchans=None

    if ps['oper'] not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian'] :          
        PCAlastevect=PCAdataStruct.PCAevect
        PCAlastprop=PCAdataStruct.PCAprop
        PCAlastchans=chans
        cind=0
        for i in range(PCAdataStruct.PCAprop.shape[1]):
            noexit=1
            name=ps['oper']+npost+'Comp'
            while noexit:
                cind+=1
                chname=name+str(cind)
                if checkMultiNames(ps['dataFB'],chname,iters): # not in self.mapdata.labels:
                    noexit=0
            name=name+str(cind)
            dataFull=PCAdataStruct.PCAprop[:,i]
            ##print data
            startindex=0
            for nbuf in iters:
                info=PCAdatafileInfo[nbuf]
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
                ps['addchan'](data,name,fbuffer=nbuf)

    if ps['oper'] in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian']:
        cind=0
        noexit=1
        name=ps['oper']+npost+'Clusters'
        while noexit:
            cind+=1
            chname=name+str(cind)
            if checkMultiNames(ps['dataFB'],chname,iters): # not in self.mapdata.labels:
                noexit=0
        name=name + str(cind)
        dataFull=PCAdataStruct.PCAKcluster
        ##print data
        startindex=0
        for nbuf in iters:
            info=PCAdatafileInfo[nbuf]
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
            ps['addchan'](data,name,fbuffer=nbuf)

        
    #print the vectors
    if len(PCAdataStruct.PCAprop.shape) > 1:
        for i in range(PCAdataStruct.PCAprop.shape[1]):
            print("EV#"+str(i),PCAdataStruct.PCAevect[i,:])
    #save results
    fb['lastPCAresult']=[PCAlastevect,PCAlastprop,chans]
            
def checkMultiNames(dfb, name,iters):
    for nbuf in iters:
        if name in dfb[nbuf]['data'].labels:
            return False
    return True


def checkMultiPCA(dfb, chans,ptype):
    if ptype=='Single File':
        return True
    for buf in list(dfb.values()):
        for c in chans:
            if c not in buf['data'].labels:
                print(c,'missing in',buf['name'])
                return False
    return True

    



def doPyMCAload(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    
    parameters = PyCD.ConfigDict()
    pyfile=os.path.dirname(__file__)+os.sep+"pyMcaConfigs"+os.sep+ps['pyfile']
    parameters.clear()
    parameters.read(pyfile)
    fb['pyMCAsilent']['pyfile']=pyfile
    fb['pyMCAsilent']['params']=parameters
    
def doPyMCAzoomfit(fbl,**kw):
    ps=kw['kw']
    doPyMCAFit(fbl,True,ps['addchan'])
    
def doPyMCAfullfit(fbl,**kw):
    ps=kw['kw']
    doPyMCAFit(fbl,False,ps['addchan'])

def doPyMCAFit(fbl,zoomfit,addchan):
    [fbn,fb]=fbl

    if os.path.splitext(fb['mcafn'])[1] != '.hdf5':
        return

    matsize=fb['data'].data.shape[0]*fb['data'].data.shape[1]
    
    matindex=np.arange(matsize)

    startt=time.process_time()

    fid=h5py.File(fb['mcafn'])
    data=fid['/main/mcadata']
    mcamaxno=data.shape[1]
    maxlines=data.shape[0]
    print('hdf',mcamaxno,maxlines)
    
    binmin=0
    binmax=2048

    if matsize<len(data):
        print("mca file too long? clipping...")
        matindex=matindex[:matsize]

    fitShape=(fb['data'].nypts,fb['data'].nxpts)

    #worry about zoom
    takezoom=False
    if zoomfit and fb['zoom'][0:4]!=[0,0,-1,-1]:  
        takezoom=True
        print("pre-zoom",np.max(matindex))
        tempdata=matindex.reshape((fb['data'].nypts,fb['data'].nxpts))
        tempdata=tempdata[::-1,:]            
        tempdata=tempdata[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]
        print("afterzoom",np.max(tempdata))
        tdsy=(fb['zoom'][3]-fb['zoom'][1])
        tdsx=(fb['zoom'][0]-fb['zoom'][2]) 
        tds=tdsx*tdsy            
        print(tds, tdsx, tdsy, tempdata.shape)
        print(fb['zoom'])
        fitShape=tempdata.shape
        matindex=tempdata.reshape((abs(tds),))  
        matindex=np.sort(matindex)
        print("aftersort",np.max(matindex))

    #look at the size of the datasets...
    chunk=1
    chunkslice=[[(0,-1),fitShape[0]]]
    print(chunk,chunkslice) 
    
    mcaWrap=pyMcaFitWrapper.Wrapper(pkm=fb['pyMCAsilent']['pyfile'])
    #set data
    newDataNames=[]
    for forIter in range(chunk):
        print("fitting chunk",forIter+1)
        if chunk==1:
            if takezoom:
                dataToFit=data[matindex,binmin:binmax]
            else:
                dataToFit=data[:,binmin:binmax]
            fFitShape=fitShape
        else:
            minind=chunkslice[forIter][0][0]
            maxind=chunkslice[forIter][0][1]
            dataToFit=data[matindex[minind:maxind],binmin:binmax]
            fFitShape=(chunkslice[forIter][1],fitShape[1])
        
        mcaWrap.setFastData(dataToFit,fFitShape)
        #fit
        fresult=mcaWrap.doFastFit()            

        print('')
        print('fit'+str(forIter),time.process_time()-startt)        

#                numberFit=list(range(len(fresult['names'])))
        numberFit=list(range(len(fresult._buffers['parameters'])))
        for ch in numberFit:
            if forIter==0:
                #nameroot=fresult['names'][ch]
                nameroot=fresult._labels['parameters'][ch]
                valid=0
                i=0
                newname=nameroot
                while not valid:
                    if newname not in fb['data'].labels:
                        valid=1
                    else:
                        i+=1
                        newname=nameroot+str(i)
                newDataNames.append(newname)

            chdata=fresult._buffers['parameters'][ch][::-1,:]
            chsigma=fresult._buffers['uncertainties'][ch][::-1,:]

            zerodata=np.zeros((fb['data'].nypts,fb['data'].nxpts)) 
            zerosigma=np.zeros((fb['data'].nypts,fb['data'].nxpts))                          
            if chunk==1 and not takezoom:
                zerodata=chdata[::-1,:] 
                zerosigma=chsigma[::-1,:] 
            elif chunk==1 and takezoom:
                zerodata[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]=chdata                
                zerodata=zerodata[::-1,:] 
                zerosigma[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]=chsigma               
                zerosigma=zerosigma[::-1,:] 
            else:
                if takezoom:
                    areadata=zerodata[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]
                    areasigma=zerosigma[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]                                        
                else:
                    areadata=zerodata
                    areasigma=zerosigma
                ishape=areadata.shape
                enditem=ishape[0]*ishape[1]
                #if chunkslice[forIter][0][0]==0:
                #    maxone=enditem-chunkslice[forIter][0][0]
                #else:
                #    maxone=enditem-chunkslice[forIter][0][0]-1                        
                if chunkslice[forIter][0][1]==-1:
                    maxtwo=0
                else:
                    maxtwo=enditem-chunkslice[forIter][0][1]
                maxone=maxtwo+chdata.shape[0]*chdata.shape[1]
                #print maxone,maxtwo
                areadata=areadata.ravel()
                areadata[maxtwo:maxone]=chdata.ravel()
                #areadata=areadata[::-1]
                areadata=areadata.reshape(ishape)
                                    
                areasigma=areasigma.ravel()
                areasigma[maxtwo:maxone]=chsigma.ravel()
                #areasigma=areasigma[::-1]
                areasigma=areasigma.reshape(ishape)
                
                #areadata=areadata[::-1,:]
                #areasigma=areasigma[::-1,:]
                
                if takezoom:
                    zerodata[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]=areadata                 
                    zerosigma[fb['zoom'][1]:fb['zoom'][3],fb['zoom'][0]:fb['zoom'][2]]=areasigma                
                else:
                    zerodata=areadata
                    zerosigma=areasigma
                zerodata=zerodata[::-1,:] 
                zerosigma=zerosigma[::-1,:] 

            if forIter==0:
                addchan(zerodata,globalfuncs.fixlabelname(newname),fbuffer=fbn)
                addchan(zerosigma,globalfuncs.fixlabelname(newname+"-sigma"),fbuffer=fbn)
            
    print('data in',time.process_time()-startt)
    fid.close()

def doPCAonMCAdata(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']

    if os.path.splitext(fb['mcafn'])[1] != '.hdf5':
        return
    
    print('Loading MCA data...')    
    i=0

    PCAhdf5fout=h5py.File(ps['filedir']+os.sep+"mcatemp.hdf5",'w')
    
    pcarawdata,mcamaxno = getHDF5MCAforPCA(fb['mcafn'],PCAhdf5fout,fb['data'],[int(ps['minbin']),int(ps['maxbin'])])
    print("MCA data import complete.")
    PCAdataStruct = PCAAnalysisMathClass.PCADataStructure(pcarawdata,int(ps['comps']),None)


    print("WORKING ON PCA")        
    #try new
    ntm=[]
    ntm.append(time.process_time())
    PCAdataStruct.donewPCA(pcatype='sPCA',MCA=True)
    ntm.append(time.process_time())
    print('PCA complete in '+str(ntm[1]-ntm[0])+' seconds')
    #compute sum
    esum=sum(PCAdataStruct.PCAeval)
    ecsum=[]
    for i in range(len(PCAdataStruct.PCAeval)):
        ecsum.append(sum(PCAdataStruct.PCAeval[0:i+1]))
    ecsum=np.array(ecsum)
    varcomp=PCAdataStruct.PCAeval/esum
    varexp=ecsum/esum
    #clear old results
    compfiles=[]
    compvars={}
    #give results in list window
    evect=np.transpose(PCAdataStruct.PCAevect) #put eigenvectors back to rows
    for c in range(int(ps['comps'])): #not cols in this case -- LIMIT
        #add component...
        wid=pcacompobj()
        #wid.xdat=arange(self.mcamaxno)
        ytemp=np.take(evect,(c,),axis=0) #slice of eigevector
        ytup=tuple(ytemp[0])
        wid.ydat=np.array(ytup)
        wid.xdat=np.arange(len(wid.ydat))+int(ps['minbin'])         
        wid.eigen=PCAdataStruct.PCAeval[c]
        wid.var=varcomp[c]
        wid.vartot=varexp[c]
        wid.ind=0#rateind[c]
        wid.cind=c
        rn=str(c+1)
        name='Comp'+rn
        compfiles.append(name)
        compvars.update({name:wid})

    print("PCA Analysis complete.... saving text files of components")          

    #get components for export
    ind=[]
    data=[]
    for npc in compfiles:
        wid=compvars.get(npc)
        ind.append(wid.cind)
        data.append(wid.ydat)
    if ind==[]:
        print('No components selected')
        [data,nocomp] = [0,0]
    nocomp = len(ind)

    if data==0:
        return
    fn=ps['filedir']+os.sep+globalfuncs.trimdirext(fbn)+'_PCAcomp.dat'
    if fn=='':
        print('Save cancelled')
        return
    fid=open(fn,'w')
    fid.write('# Principal componets for '+globalfuncs.trimdirext(fbn)+'\n')
    fid.write('MCA Bin\t')
    for i in range(nocomp):
        fid.write('Comp'+str(i)+'\t')
    fid.write('\n')
    #parse list now
    for i in range(len(data[0])):
        #setup text
        fid.write(str(i+1)+'\t')
        for j in range(nocomp):
            fid.write(str(data[j][i])+'\t')
        fid.write('\n')
    fid.close()
    print("Active component data saved to file:"+fn)   

    #want to export wieght matrix as a data set for active components...

    print("Exporting component weights to map dataset...")         
    for npc in compfiles:
        wid=compvars.get(npc)
        #export this one...
        noexit=1
        name='PCAMCA_Comp'
        while noexit:
            name=name+str(wid.cind+1)
            if name not in fb['data'].labels:
                noexit=0
        data=PCAdataStruct.PCAprop[:,wid.cind]
        #reshape data and check lengths
        data=data[:len(np.ravel(fb['data'].data.get(0)))]
        ##print data.shape,self.mapdata.data[:,:,0].shape
        data=np.reshape(data,(fb['data'].data.get(0).shape))
        #add to mapdata
        ps['addchan'](data,name,fbuffer=fbn)
    print("Done!")
    PCAhdf5fout.close()

def getHDF5MCAforPCA(fn,temp,mapdata,binrange):
    (PCAminbin,PCAmaxbin)=binrange
    fid=h5py.File(fn)
    if "/main/mcadata" in fid:
        mcadata=fid['/main/mcadata']
    elif "/main/oodata" in fid:
        mcadata=fid['/main/oodata']
    else:
        print('no mcadata found')
        return
    mcamaxno=mcadata.shape[1]
    maxlines=mcadata.shape[0]
    print('hdf',mcamaxno,maxlines)
    
    groupout=temp.create_group("main")
    matsize=mapdata.data.shape[0]*mapdata.data.shape[1]

    pcasize=int(PCAmaxbin)-int(PCAminbin)

    newdata=groupout.create_dataset("mcadata",(matsize,pcasize),maxshape=(None,pcasize),dtype='int',compression="gzip",compression_opts=9)

    if maxlines==matsize:
        newdata=np.array(mcadata[:,PCAminbin:PCAmaxbin])

    if maxlines<matsize:
        newdata[:maxlines,:pcasize]=mcadata[:PCAminbin:PCAmaxbin]

    if maxlines>matsize:
        newdata[:,:pcasize]=mcadata[:matsize,PCAminbin:PCAmaxbin]
    
    fid.close()
    return newdata,mcamaxno     
    

def doXANESfit(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']

    fitXstruct=XanesFitClass.xanesfitobj(None,display=False)

    #get fn
    #read first line
    fid=open(ps['xasfile'],'rU')
    l=fid.readline()
    if l!='SMAK XFIT\n':
        print('Invalid parameter file!')
        fid.close()
        return
    #read data
    numstd=int(fid.readline())
    stdname=[]
    for n in range(numstd):
        stdname.append(str.strip(fid.readline()))
    numdat=int(fid.readline())
    datname=[]
    for n in range(numdat):
        datname.append(str.strip(fid.readline()))
    fid.readline() #data start
    mat={}
    for n in datname:
        l=fid.readline().split()
        mat[n]=l
    fid.close()
    #correlate file names with current data labels
    idok=True 
    for n in datname:
        if n not in fb['data'].labels: idok=False
    if not idok:
        print ('data-standard channel disagreement...')
        return

    #select names and numchans
    fitXnumstds=numstd
    sel=datname.copy()  #this is also fitXdata
    fitXstruct.update(sel,fitXnumstds)
    #place 'em
    print(fitXstruct.columns)
    newX=[]
    newc=['Names']
    for i in range(numstd):
        newc.append(stdname[i])
    newX.append(newc)
    for i in range(numdat):
        newc=[]
        newc.append(datname[i])
        for n in mat[datname[i]]:
            newc.append(n)
        newX.append(newc)
    fitXstruct.columns=newX
    print(fitXstruct.columns)

    print('Performing fitting...')
    #do
    
    stdnum=int(fitXnumstds)
    #make standard matrix and data index
    stdmat=[]
    datind=[]
    bigdat=[]
    for c in fitXstruct.columns[1:]:
        newc=[]
        datind.append(fb['data'].labels.index(c[0])+2)
        bigdat.append(fb['data'].data.get(fb['data'].labels.index(c[0])+2))
        for n in c[1:]:
            newc.append(float(n))
        stdmat.append(newc)
    stdmat=np.array(stdmat)
    bigdat=np.array(bigdat)
    #iterate through data
    t=time.process_time()
    (xlen,ylen)=fb['data'].data.shape[:2]

    resShape=list(fb['data'].data.shape)
    resShape[2]=stdnum
    result=np.zeros(tuple(resShape),dtype=np.float32)
    error=np.zeros(fb['data'].data.shape[:2],dtype=np.float32)
    for i in range(xlen):
        for j in range(ylen):
            pe=0
            #get data for this pixel
            dat=[]
            td=fb['data'].data.getPix(i,j)
            for k in datind:
                dat.append(td[k])#[i,j,k])
            dat=np.array(dat)

            [calc,err]=scipy.optimize.nnls(stdmat,dat)
            err=[err]

            for k in range(stdnum):
                result[i,j,k]=calc[k]*1.0
            try:
                error[i,j]=err[0]
            except:
                #print i,j,err
                if not pe: sys.stdout.write('!')
                error[i,j]=0
        sys.stdout.write('.')
    sys.stdout.write('\n')    
    #add to data    
    csum=0
    for i in range(stdnum):
        ps['addchan'](result[:,:,i],globalfuncs.fixlabelname(fitXstruct.columns[0][i+1]),fbuffer=fbn)
        csum+=result[:,:,i]
    base='sumFIT'
    i=1
    while 1:
        if base+str(i) in fb['data'].labels:
            i=i+1
        else:
            base=base+str(i)
            break     
    ps['addchan'](csum,base,fbuffer=fbn)        
    base='fiterror'
    i=1
    while 1:
        if base+str(i) in fb['data'].labels:
            i=i+1
        else:
            base=base+str(i)
            break
    ps['addchan'](error,base,fbuffer=fbn)
    print(time.process_time()-t)          
    print('Fit complete!')    


def doQuantify(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']

    cfilewid = MultiMassCalibrationClass.CalibResultObject()
    cfilewid.loadFile(ps['quantfile'])
    cfilewid.chlistb = [item for sublist in cfilewid.chlist for item in sublist]


    print (cfilewid.chlistb)
    print (cfilewid.ellist)
    print (set(fb['data'].labels))

    chanquantdict={}
    for e,c in zip(cfilewid.ellist,cfilewid.chlistb):
        chanquantdict[c]=e

    #determine which channels can be quantified...
    quantselchans=list(set(cfilewid.chlistb) & set(fb['data'].labels))
    if len(quantselchans)==0:
        print ('no common channels in standards and data')
        return
    
    print (quantselchans)
    
    #normalize channel?
    normCCSchan = cfilewid.normalize.strip()
    if "None"  in normCCSchan:
        normCCSchan = None
    else:
        if normCCSchan not in fb['data'].labels:
            print ('no i0 channel...')
            return
        
    #now do a bunch of calibrations...
    print('Doing quantitative analysis...')
    #do
    for q in quantselchans:
        #get data
        Aind=fb['data'].labels.index(q)+2     
        (xlen,ylen)=fb['data'].data.shape[:2]

        if normCCSchan is not None:
            iind=fb['data'].labels.index(normCCSchan)+2
            i0dat=fb['data'].data.get(iind)#[:,:,iind]
        else:
            i0dat=np.ones((xlen,ylen),dtype=np.float32)
        #divide by i0
        adata=fb['data'].data.get(Aind)
        
        newdata=np.divide(adata,i0dat, out=np.zeros_like(adata),where=i0dat!=0)
        
        #fw contains the calibration and units data in fw.slope, fw.intc, and fw.units
        #fw for this is: self.cfilewid.chdict[el]
        # el = self.chanquantdict[chan] where chan is q
        fw = cfilewid.chdict[chanquantdict[q]]
        #calculate
        mod = [fw.slope,fw.intc]
        pred = np.poly1d(mod)
        newdata = pred(newdata)
        
        #add data!
        nameroot=q+'-'+fw.units+'-'
        valid=0
        i=1
        while not valid:
            newname=nameroot+str(i)
            if newname not in fb['data'].labels:
                valid=1
            else:
                i+=1
        ps['addchan'](newdata,newname,fbuffer=fbn)

    print('Quantative analysis complete!')     
    
    
def doFTIRload(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    
    #load file
    FTIRgroup = IR_MathClass.IRParamClass()
    FTIRgroup.load(ps['irfile'])       
    print('load complete')
    fb['FTIRgroup']=FTIRgroup

    
def switchWavenumber(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    getMCAMetadata(fbl)
    
def doFTIRcorrections(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    
    if os.path.splitext(fb['mcafn'])[1] != '.hdf5':
        return

    #have hdf5 data...
    fid=h5py.File(fb['mcafn'])
    if "/main/mcadata" in fid:
        mcadata=fid['/main/mcadata']
    elif "/main/oodata" in fid:
        mcadata=fid['/main/oodata']
    else:
        print('no mcadata found')
        return

    if len(fb['MCAxvalues']) != len(mcadata[0]):
        print ('length mismatch')
        MCAx=list(range(len(mcadata[0])))
    else:
        MCAx=fb['MCAxvalues']

    mcadata=np.array(mcadata)
    maxlen=mcadata.shape[1] #in spectra/wn
    maxpoints=mcadata.shape[0]  #in entire file
    print('hdf',maxpoints,maxlen)
          
    if fb['FTIRgroup']==None:
        fb['FTIRgroup']=IR_MathClass.IRParamClass()
    fb['FTIRgroup'].saveCurrent()
    resultx,resulty = IR_MathClass.DoNorm(MCAx ,mcadata,fb['FTIRgroup'])  
    
    newfn=os.path.splitext(fb['mcafn'])[0]+"_ircorr.hdf5"
    newfid=h5py.File(newfn,'w')
    groupout=newfid.create_group("main")

    outmcadata=groupout.create_dataset("mcadata",(resulty.shape),maxshape=(None,resulty.shape[1]),dtype='float',compression="gzip",compression_opts=4)        
    if "/main/xdata" in fid:
        mcadataxv=groupout.create_dataset("xdata",(resulty.shape[1]),maxshape=(resulty.shape[1]),dtype='float',compression="gzip",compression_opts=4)
        mcadataxv[:]=resultx
    outmcadata[:,:]=resulty

    newfid.close()        
    fid.close()
    
    print ('done FTIR',fbn)
    
def doFTIRcorrectionsLoad(fbl,**kw):
    [fbn,fb]=fbl
    ps=kw['kw']
    
    doFTIRcorrections(fbl,**kw)
    newfn=os.path.splitext(fb['mcafn'])[0]+"_ircorr.hdf5"
    fb['mcafn']=newfn
    getMCAMetadata(fbl)
    
    
    
    
    