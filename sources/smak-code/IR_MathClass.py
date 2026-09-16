#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:47:27 2023

@author: samwebb
"""

import numpy as np

import octavvs.algorithms.atm_correction as atm
import octavvs.algorithms.baseline as baseline
import octavvs.algorithms.correction as correction
import octavvs.algorithms.mie_correction as mie
import octavvs.algorithms.normalization as normalization
import octavvs.algorithms.util as util

import json
import scipy


#scattering CRMieSc
    #reference: Mean, Percentile, Lignin, Casein, Matrigel
    #max iters: 30, clusters: 30, stabilize
#Denoise
    #window size (odd-9), poly order: 3
#range select:
    #min/max 800-1800?
#baseline
    #opt: rubberband, concave rubberband, AsLS, arPLS, Assym Trunc Quad
#normalization
    #opt: Mean, total area, wavenumber (@val 1665 def), maximum, vector 

class IRparamDefs:
    def __init__(self):
        self.atmos = ['None', 'Water', 'CO2+Water']
        self.mieref = ['None','Mean', 'Percentile', 'Lignin', 'Casein', 'Matrigel']
        self.base = ['None','Rubberband', 'Concave Rubberband', 'AsLS', 'arPLS']
        self.norm = ['None', 'Mean', 'Total area', 'Wavenumber' , 'Maximum', 'Vector' ]
        self.normtrans = {'None':'None', 'Mean':'mean', 'Total area':'area', 'Wavenumber':'wn' , 'Maximum':'max', 'Vector':'n2'}
        
class IRParamClass:
    def __init__(self):
        self.atmCorr = True
        self.atmRef = None
        self.atmCutCO2 = True
        self.atmExtraIters = 5
        self.atmExtraFactor = 0.25
        
        self.mieReference = 'None'
        self.mieMaxIter = 30
        self.mieClusters = 30
        self.mieStabilize = True
        
        self.denoiseWindow = 9
        self.denoisePolyOrder = 3

        self.rangeSelect=[800,1800]
        
        self.baseline = 'Rubberband'
        self.lam = 1000
        self.p = 0.01
        self.baseiters = 10
        #self.baseAsymPoly = 3
        #self.baseAsymThresh = 0.01

        self.normalization = 'Mean'
        self.normvalue = 1655
        
        self.dict={}
        self.defs=IRparamDefs()
        
        self.stepdata = {'atm':None,'mie':None,'final':None}
        
        self.lastparams=None
        self.lastshape=None
        
    def decodeAtmos(self):
        if self.atmCorr == False:
            return 'None'
        if self.atmCutCO2 == True:
            return 'CO2+Water'
        else:
            return 'Water'
    
    def encodeAtmos(self,value):
        if value == 'None':
            self.atmCorr = False
            self.atmCutCO2 = False
        elif value == 'CO2+Water':
            self.atmCorr = True
            self.atmCutCO2 = True 
        else:
            self.atmCorr = True
            self.atmCutCO2 = False 
            
    def saveCurrent(self):
        self.makeDict()
        self.lastparams=self.dict.copy()
        

    def makeDict(self):
        d={}
        d['corr']=self.atmCorr
        d['ref']=self.atmRef
        d['cutco2']=self.atmCutCO2
        d['xiters']=self.atmExtraIters
        d['xfact']=self.atmExtraFactor
        self.dict['atm']=d
        
        m={}
        m['ref']=self.mieReference
        m['maxiter']=self.mieMaxIter
        m['clusters']=self.mieClusters
        m['stabilize']=self.mieStabilize
        self.dict['mie']=m
        
        dn={}
        dn['window']=self.denoiseWindow
        dn['polyord']=self.denoisePolyOrder
        self.dict['denoise']=dn
        
        self.dict['rangeselect']=self.rangeSelect
        
        bl={}
        bl['baseline']=self.baseline
        bl['lam']=self.lam
        bl['p']=self.p
        bl['baseiters']=self.baseiters
        self.dict['baseline']=bl
        
        n={}
        n['normal']=self.normalization
        n['normvalue']=self.normvalue
        self.dict['normal']=n

    def loadDict(self,d):
        self.dict=d
        
        self.atmCorr = d['atm']['corr']
        self.atmRef = d['atm']['ref']
        self.atmCutCO2 = d['atm']['cutco2']
        self.atmExtraIters = d['atm']['xiters']
        self.atmExtraFactor = d['atm']['xfact']
        
        self.mieReference = d['mie']['ref']
        self.mieMaxIter = d['mie']['maxiter']
        self.mieClusters = d['mie']['clusters']
        self.mieStabilize = d['mie']['stabilize']
        
        self.denoiseWindow = d['denoise']['window']
        self.denoisePolyOrder = d['denoise']['polyord']

        self.rangeSelect=d['rangeselect']
        
        self.baseline = d['baseline']['baseline']
        self.lam = d['baseline']['lam']
        self.p = d['baseline']['p']
        self.baseiters = d['baseline']['baseiters']

        self.normalization = d['normal']['normal']
        self.normvalue = d['normal']['normvalue']

    def save(self,fn):
        self.makeDict()
        exp=json.dumps(self.dict)
        fid=open(fn,'w')
        fid.write(exp)
        fid.close()
        
    def load(self,fn):
        fid=open(fn,'r')
        inp=fid.read()
        fid.close()
        self.loadDict(json.loads(inp))

    def verbose(self):
        print ('Atm has data:',self.stepdata['atm'] is not None)
        print ('Atm changed',self.dict['atm'] != self.lastparams['atm'])
        print ('Mie has data:',self.stepdata['mie'] is not None)
        print ('Mie changed',self.dict['mie'] != self.lastparams['mie'])

def normCallback(a,b):
    pct = float(a)/float(b)*100
    print ("CRMie Scattering Correction ... ",pct,"% complete")        
        
def DoNorm(wn,data,params,force=False):
 
    #shape
    print (data.shape)   
    #dictionary params
    params.makeDict()
    params.verbose()

    #atm corr
    redidatm=False
    if params.lastparams['atm']['corr'] != params.atmCorr or (params.lastshape!=data.shape):
        redidatm=True
    elif params.atmCorr:
        if (params.stepdata['atm'] is None) or (params.dict['atm'] != params.lastparams['atm']) or (params.lastshape!=data.shape):
            (data,b) = atm.atmospheric(wn,data,atm=None,cut_co2=params.atmCutCO2)
            print ('atm corr.fact.',b)
            params.stepdata['atm']=data
            redidatm=True
        else:
            print ('copy old atm')
            data=params.stepdata['atm']
            redidatm=False
            
    redidatm = redidatm or force
    params.lastshape=data.shape            
    #mie scattering
    if params.mieReference != 'None':
        if (params.stepdata['mie'] is None) or (params.dict['mie'] != params.lastparams['mie']) or redidatm:
            if params.mieReference in ['Lignin','Casein','Matrigel']:
                ref = util.load_reference(wn, what=params.mieReference.lower())
            else:
                if len(data.shape)==1:
                    ref=data
                else:
                    ref = data.mean(axis=0)
            data = mie.rmiesc(wn,data,ref,iterations=params.mieMaxIter,clusters=params.mieClusters,progressCallback=normCallback)
            params.stepdata['mie']=data
        else:
            print ('cope old mie')
            data=params.stepdata['mie']

    #denoise
    if params.denoiseWindow > params.denoisePolyOrder and data.shape[1]>=params.denoiseWindow:
        corr = scipy.signal.savgol_filter(data, params.denoiseWindow, params.denoisePolyOrder, axis=1)
        data=corr
        print ('denoise ok')
    else:
        print ('no denoising performed')
        print (params.denoiseWindow,params.denoisePolyOrder,data.shape)

    #range select
    if len(params.rangeSelect) < 2:
        print ('no edit of range')
    else:
        ranges=np.array([[params.rangeSelect[0]],[params.rangeSelect[1]]]).T
        ind = util.find_wn_ranges(wn,ranges)
        ind=ind[0]
        wn=wn[ind[0]:ind[1]]
        data = data [:,ind[0]:ind[1]]
        print ('range select ok',ind)
  
    #baseline
    bl=np.zeros(wn.shape)
    if params.baseline == 'Rubberband':
        bl = baseline.rubberband(wn, data)
    elif params.baseline == 'Concave Rubberband':
        bl = baseline.concaverubberband(wn, data, params.baseiters)
    elif params.baseline == 'AsLS':
        bl = baseline.asls(data,params.lam,params.p)
    elif params.baseline == 'arPLS':
        bl = baseline.arpls(data,params.lam)
    elif params.baseline == 'Assym Trunc Quad': #not implemented currently
        pass
    else:
        data=data
    data=data-bl
    print ('baseline',params.baseline)
        
    #normalization
    if params.normalization != 'None':
        data = normalization.normalize_spectra(params.defs.normtrans[params.normalization], data, wn=wn, wavenum=params.normvalue)
    print ('normalization',params.normalization)
    
    #return results
    return wn,data

