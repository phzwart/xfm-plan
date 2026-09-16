# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 13:39:19 2016

@author: samwebb
"""

#standard
from inspect import getsourcefile
import os
import os.path
import time

#third party
import numpy as np

import PyMca5.PyMcaPhysics.xrf.ClassMcaTheory as MCAT
from PyMca5.PyMcaPhysics.xrf import ConcentrationsTool
from PyMca5.PyMcaPhysics.xrf import FastXRFLinearFit


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)


class Wrapper:
    def __init__(self,pkm=None):

        if pkm==None:
            path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))+os.sep+"pyMcaConfigs"+os.sep
#            print path
            pkm=path+"defaultPyMCAConfig.cfg"

        continuum=0
        stripflag=1
        maxiter=10
        sumflag=1
        hypermetflag=1
        escapeflag=1
        attenuatorsflag=1
        self.outfile=None

        self.x=None
        self.y0=None
        self.fitdone=False

        self.mcafit = MCAT.McaTheory(initdict=pkm,maxiter=maxiter,sumflag=sumflag,
                        continuum=continuum,escapeflag=escapeflag,hypermetflag=hypermetflag,
                        attenuatorsflag=attenuatorsflag)
        self.config = self.mcafit.configure()

    def setData(self,ydata):
        xmin = self.config['fit']['xmin']
        xmax = self.config['fit']['xmax']
        self.y0= np.array(ydata)
        self.x = np.arange(len(self.y0))*1.0
        self.mcafit.setData(self.x,self.y0,xmin=xmin,xmax=xmax)
        
    def setFastData(self,ydata,yshape):
        print(ydata.shape,yshape)
        self.xmin = self.config['fit']['xmin']
        self.xmax = self.config['fit']['xmax']
        xdepth=len(ydata[0,:])
        self.x = np.arange(xdepth)*1.0
        self.y0= ydata #np.array(ydata,dtype='float32')
        #self.y0.astype('float32')
        self.y0=self.y0.reshape([yshape[0],yshape[1],xdepth])
        print(self.y0.shape)
        #self.mcafit.setData(self.x,self.y0,xmin=xmin,xmax=xmax)     

    def doFastFit(self,fitpass=False,useConc=False):
        self.fitdone=False
        t0=time.time()        
        fitconfig = {}
        fitconfig.update(self.mcafit.configure())
        if fitconfig['peaks'] == {}:
        
            print("No peaks defined.\nPlease configure peaks")
            return [0],1

        fFit=FastXRFLinearFit.FastXRFLinearFit(mcafit=self.mcafit)
        result = fFit.fitMultipleSpectra(x=self.x, 
                                y=self.y0, xmin=self.xmin, xmax=self.xmax,
                                configuration=fitconfig, concentrations=useConc,
                                ysum=None, weight=None, refit=True)
        print('done w/fit')
        
        #print ('info',result._info)
        #print ('buffers',result._buffers)
        #print ('results',result._results)
        #print ('labels',result._labels)
        
        
        return result                    
                            
        
    def doFit(self,verbose=False,fitpass=False,useConc=False):        
        self.fitdone=False
        t0=time.time()
        self.mcafit.estimate()
        if fitpass: excess=12
        else: excess = 2
        
        if verbose:
            print("estimation time ",time.time()-t0)
            print(self.mcafit.PARAMETERS)
            print(self.mcafit.config['peaks'])
            print(self.mcafit.config)
        
        fitconfig = {}
        fitconfig.update(self.mcafit.configure())
        if fitconfig['peaks'] == {}:
        
            print("No peaks defined.\nPlease configure peaks")
            return [0],1
        
        try:
            #fitresult, mcafitresult=mcafit.startfit(digest=1)
            fitresult    = self.mcafit.startfit(digest=0)
            self.mcafitresult = self.mcafit.digestresult(self.outfile)
  
            if useConc: 
                fr={}
                fr['result']=self.mcafitresult    
                concentrationsTool = ConcentrationsTool.ConcentrationsTool(config=self.mcafit.config['concentrations'],fitresult=fr)
                conc = concentrationsTool.processFitResult()


            if verbose: print("fit took ",time.time()-t0)
        except:
            sys.stdout.write('!')
            zerosize = []
            for j in range(len(list(fitconfig['peaks'].keys()))+1+excess):
                zerosize.append(0)
            return zerosize,list(range(len(list(fitconfig['peaks'].keys()))+1+excess)),zerosize
            
            
        fittedpar=fitresult[0]
        chisq    =fitresult[1]
        sigmapar =fitresult[2]
        i = 0
        
        
        if verbose:
            print("chisq = ",chisq)
            for param in self.mcafit.PARAMETERS:
                if i < self.mcafit.NGLOBAL:
                    print(param, ' = ',fittedpar[i],' +/- ',sigmapar[i])
                else:
                    print(param, ' = ',fittedpar[i],' +/- ',sigmapar[i])
                #,'mcaarea = ',areas[i-self.mcafit.NGLOBAL]
                i += 1
            i = 0
            #self.mcafit.digestresult()
            for group in self.mcafitresult['groups']:
                print(group,self.mcafitresult[group]['fitarea'],' +/- ', \
                   self.mcafitresult[group]['sigmaarea'],self.mcafitresult[group]['mcaarea'])
            
            print("##################### ROI fitting ######################")
            print(self.mcafit.roifit(self.mcafit.xdata,self.mcafit.ydata))
        else: 
            sys.stdout.write('.')
        self.fitdone=True
       
            
        RA=[]
        CA=[]
        plist=[]
        if fitpass:
            for param in self.mcafit.PARAMETERS:
                if i<self.mcafit.NGLOBAL:
                    RA.append(fittedpar[i])
                    plist.append(param)
        else:
            for group in self.mcafitresult['groups']:
                RA.append(self.mcafitresult[group]['fitarea']) 
                if useConc and not group.startswith('Scatter'):
                    CA.append(conc['mass fraction'][group])
                elif useConc and group.startswith('Scatter'):
                    CA.append(0.0)
                plist.append(group)
            
        RA.append(chisq)
        CA.append(chisq)
        plist.append("chisq")    
        
        
        return RA,CA,plist

    def getFitData(self):
        if self.fitdone==False:
            print("No fit performed")
            return []
        xw=np.ravel(self.mcafit.xdata)
        ##yfit0 = self.mcafit.mcatheory(self.mcafit.parameters,xw)+np.ravel(mcafit.zz)
        yfit0 = self.mcafitresult['yfit']
        xw = (xw*self.mcafit.parameters[1]+self.mcafit.parameters[0])*1000
        #return x,ydata,estimated,background
        return [xw,np.ravel(self.mcafit.ydata),yfit0,np.ravel(self.mcafit.zz)]
        
    def getFittedParameters(self):
        params={}
        for i in range(len(self.mcafitresult['parameters'])):
            params[self.mcafitresult['parameters'][i]]=self.mcafitresult['fittedpar'][i]
        return params
        