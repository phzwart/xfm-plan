#necessary first imports
import tkinter
import Pmw


root=tkinter.Tk()
Pmw.initialise(root)

#standard library imports
import functools
import importlib.util
import itertools
import math
import os
import os.path
import random
import shutil
import stat
import struct
import sys
import threading
import time
from inspect import getsourcefile



#third party imports
import cv2 as cv
import functools
import h5py
import imreg_dft as ird
from imreg_dft import tiles as irdTiles
import imutils
import json
import matplotlib.colors as matcolors
import matplotlib.pyplot as plt
import numpy as np
#possibly get rid of this
from numpy import array, ones, ravel, transpose, zeros
import numpy.linalg as LinearAlgebra
import packaging.version as VRS
import pickle
from PIL import Image, ImageFile, ImageTk, _imaging
import re
import scipy.optimize
import scipy.stats as Stats
import sklearn
import sklearn.decomposition as skdecomp
import sklearn.ensemble as skensemble
import sklearn.metrics.pairwise as sklPairs
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.transform import rescale as skrescale
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.cluster import MiniBatchKMeans
from sklearn import manifold as skmanifold
import sklearn.metrics.pairwise as sklPairs
import skimage
import tkinter.colorchooser
import tkinter.filedialog
import tkinter.messagebox
import tkinter.simpledialog
from tkinter.ttk import Button, Style, Treeview


#third party continued because py2exe is dumb...
from PIL import JpegImagePlugin, PpmImagePlugin, PngImagePlugin, TiffImagePlugin, WmfImagePlugin, GifImagePlugin, BmpImagePlugin

Image._initialized=1

#blatant fix...
Image.SAVE["JPEG"]=JpegImagePlugin._save
Image.EXTENSION[".jpg"]="JPEG"
Image.SAVE["PPM"]=PpmImagePlugin._save
Image.EXTENSION[".ppm"]="PPM"
Image.SAVE["PNG"]=PngImagePlugin._save
Image.EXTENSION[".png"]="PNG"
Image.SAVE["TIFF"]=TiffImagePlugin._save
Image.EXTENSION[".tif"]="TIFF"
Image.SAVE["WMF"]=WmfImagePlugin._save
Image.EXTENSION[".wmf"]="WMF"
Image.SAVE["GIF"]=GifImagePlugin._save
Image.EXTENSION[".gif"]="GIF"
Image.SAVE["BMP"]=BmpImagePlugin._save
Image.EXTENSION[".bmp"]="BMP"
print("thru PIL inits")

#local imports 
from . import absorptionCalc
from . import ChannelInterpret
from . import ciexyz
from . import colormodels
from . import Deconv
from . import Display
from . import FileTracker
from . import globalfuncs
from . import histclass
from . import ImageGet
from . import ImCmap
from . import ImRadon
from . import LAMSImageGet
from . import MCAImageGet
from . import MomentMathClass
from . import MyGraph
from . import NNLS
from . import parseFormula
from . import sblite
from . import screencapture
from . import ScrollTree
from . import smwPmw
from . import stitch_hdf_mod
from . import sivm
from . import tkFileDialogDir
from . import varimax


#local imports GUI Classes
from . import AdvancedFilteringClass
from . import BeamDeconvolutionClass
from . import BatchAnalyze
from . import BatchDisplay
from . import BatchMode
from . import BatchProcess
from . import ConcentrationStandardClass
from . import CorrelationPlotClass
from . import CustomKernelClass
from . import DataAveragingClass
from . import DataSummaryClass
from . import ExportRegistrationClass
from . import FaderClass
from . import MakeHistogramClass
from . import Mask
from . import MathWindowClass
from . import MomentAnalysisClass
from . import MultiFitObj
from . import MultiMassCalibrationClass
from . import PCAAnalysisClass
from . import PCAAnalysisMathClass
from . import ParticleStatisticsClass
from . import pyMcaFitWrapper
from . import pyMcaParamGUI
from . import QuantThicknessDialog
from . import RadialProfileClass
from . import ThresholdingClass
from . import TriColorWindowClass
from . import XanesFitClass


#local import style 
from . import PmwTtkButtonBox
from . import PmwTtkMenuBar
from . import PmwTtkNoteBook
from . import PmwTtkRadioSelect


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

if VRS.Version(sklearn.__version__)>VRS.parse("0.23.1"):
    print ('sklearn version > 0.23.1,  MacOS may have segmentation faults with clustering...')
        


#check segment anything install
if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("pycocotools") is not None and importlib.util.find_spec("segment_anything") is not None:
    print ('utilize segment anything')
    samIsInstalled=True
    
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from pycocotools import mask as mask_utils
    print ('sam-torchok')
#    mode,path = globalfuncs.getModePath()
#    p = path+os.sep+"sam_vit_h_4b8939.pth"
#    print (os.path.exists(p))
#    t=time.time()
#    samModel = sam_model_registry["default"](checkpoint=p)
#    print ('samok')    
        
#    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
#    from pycocotools import mask as mask_utils
else:
    print("NO TORCH")
    samIsInstalled=False

#all imports are complete
print("imports completed")



global LASTDIR
global MINSIZE
global DEFAULT_HEIGHT

DEFAULT_HEIGHT=255
MINSIZE=300
LASTDIR=1


filepath=os.getcwd()+os.sep



class Unbuffered(object):
   def __init__(self, stream):
       self.stream=stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout=Unbuffered(sys.stdout)


print("done import")

def programabout(root):
    Pmw.aboutversion('3.0.0')
    Pmw.aboutcopyright('Copyright Samuel Webb, 2006\nStanford Synchrotron Radiation Laboratory')
    Pmw.aboutcontact("""SMAK is Sam's Microprobe Analysis toolKit

email: samsxrays.com@gmail.com
web: http://smak.sams-xrays.com

Reference:
Webb S.M. (2011)  "The MicroAnalysis Toolkit: X-ray Fluorescence Image Processing Software."
AIP Conference Proceedings, 1365, pp. 196-199.

XRF Fitting via PyMCA: V.A. Solé, Spectrochim. Acta Part B (2007). 
FTIR Processing via OCTAVVS: C. Troein, Methods Protoc. (2020).""")
    imlogo=Image.open("smak_sm.jpg")
    imlogo.load()
    logo=ImageTk.PhotoImage(imlogo)
    logo.paste(imlogo)
    about=Pmw.AboutDialog(root,applicationname='SMAK',icon_image=logo)
    about.component('icon').image=logo

########### Style
    

SMAKStyle=Style()
SMAKStyle.theme_use('default')

SMAKStyle.configure("OPEN.TButton",foreground='black',background='lightyellow')
SMAKStyle.configure("GREEN.TButton",foreground='snow',background='darkgreen')
SMAKStyle.configure("RED.TButton",foreground='snow',background='indianred')
SMAKStyle.configure("RRED.TButton",foreground='snow',background='red')
SMAKStyle.configure("FIREB.TButton",foreground='snow',background='firebrick4')
SMAKStyle.configure("BROWN.TButton",foreground='snow',background='chocolate3')
SMAKStyle.configure("SBLUE.TButton",foreground='snow',background='steel blue')
SMAKStyle.configure("LGREEN.TButton",foreground='snow',background='palegreen4')
SMAKStyle.configure("OGREEN.TButton",foreground='snow',background='olivedrab4')
SMAKStyle.configure("RBLUE.TButton",foreground='snow',background='royalblue3')
SMAKStyle.configure("MBLUE.TButton",foreground='snow',background='midnightblue')
SMAKStyle.configure("ORANGE.TButton",foreground='snow',background='darkorange')
SMAKStyle.configure("NAVY.TButton",foreground='snow',background='navy')

if sys.platform!="darwin":
    SMAKStyle.configure("TMenubutton",background='#d4d0c8',borderwidth='0',width='',padding='3 3')
else:
    SMAKStyle.configure("TMenubutton",background='white',borderwidth='0',width='',padding='3 3')



def gkern(kernlen=3, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp=np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2]=1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return spFilters.gaussian_filter(inp, nsig)

def neighbors(i,j,l,mx,my):
    m=list(itertools.product(list(range(i-(l-1)/2,i+(l-1)/2+1)),list(range(j-(l-1)/2,j+(l-1)/2+1))))
    rl=[]
    for s in m:
        if s[0]<0 or s[1]<0: rl.append(s)
        if s[0]>=mx or s[1]>=my: rl.append(s)
    for r in rl:
        try: m.remove(r)
        except: 
            print("duplicate?",r,rl)
    return m

#######################################
## XRD Entry classes and funcs
#######################################

class xrdentryobj:
    def __init__(self,mf):
        f=tkinter.Frame(mf,background='#d4d0c8')
        f.pack(side=tkinter.TOP,padx=2,pady=2)
        self.name=Pmw.EntryField(f,labelpos='w',label_text='Name: ',entry_width=15,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.low=Pmw.EntryField(f,labelpos='w',label_text='Low: ',validate='real',entry_width=10,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.hi=Pmw.EntryField(f,labelpos='w',label_text='Hi: ',validate='real',entry_width=10,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.name.pack(side=tkinter.LEFT,padx=5,pady=4)
        self.low.pack(side=tkinter.LEFT,padx=5,pady=4)
        self.hi.pack(side=tkinter.LEFT,padx=5,pady=4)

#######################################
## XANES Fit classes and funcs
#######################################


                

#######################################
## Quantify entry classes
#######################################


class quantfield:
    def __init__(self,name,master,std=0):
        self.name=name
        self.master=master
        self.std=std
        self.gainlist=['1e6','2e6','5e6','1e7','2e7','5e7','1e8','2e8','5e8','1e9','2e9','5e9','1e10','2e10','5e10','1e11']
        #make a series of entry fields...
        w=10
        px=3
        f=tkinter.Frame(master,background='#d4d0c8')
        #channel name
        self.chname=Pmw.EntryField(f,entry_width=w,hull_background='#d4d0c8')
        self.chname.pack(side=tkinter.LEFT,padx=px)
        self.chname.setvalue(name)
        self.chname.component('entry').configure(state=tkinter.DISABLED)        
        #element
        self.element=Pmw.EntryField(f,entry_width=w,hull_background='#d4d0c8')
        self.element.pack(side=tkinter.LEFT,padx=px)
        #standard formula
        self.formula=Pmw.EntryField(f,entry_width=w,hull_background='#d4d0c8')
        self.formula.pack(side=tkinter.LEFT,padx=px)
        #standard concentration
        self.conc=Pmw.EntryField(f,entry_width=w,validate='real',hull_background='#d4d0c8')
        self.conc.pack(side=tkinter.LEFT,padx=px)
        #cts/I0
        self.cts=Pmw.EntryField(f,entry_width=w,validate='real',hull_background='#d4d0c8')
        self.cts.pack(side=tkinter.LEFT,padx=px)
        if not std:
            #sample I0 gain
            self.i0gain=Pmw.ComboBox(f,history=0,selectioncommand=tkinter.DISABLED,hull_background='#d4d0c8')
            self.i0gain.setlist(self.gainlist)
            self.i0gain.pack(side=tkinter.LEFT,padx=px)
        #standard I0 gain
        self.stdi0gain=Pmw.ComboBox(f,history=0,selectioncommand=tkinter.DISABLED,hull_background='#d4d0c8')
        self.stdi0gain.setlist(self.gainlist)
        self.stdi0gain.pack(side=tkinter.LEFT,padx=px)
        f.pack(side=tkinter.TOP,padx=2,pady=2)
        
        

        
#######################################
## PlotMarker entry class
#######################################

class plotMarkerField:
    def __init__(self,master,posfunc,updatefunc,delcb=None):
        self.f=tkinter.Frame(master,bd=2,relief=tkinter.RIDGE,background='#d4d0c8')
        self.posfunc=posfunc
        self.updatefunc=updatefunc
        self.delcb=delcb
        self.textpresent=0
        self.valid=0
        self.markertypes=['sm circle','big circle','sm square','big square','sm emptycircle','big emptycircle','sm emptysquare','big emptysquare','sm triangle','big triangle','text']
        #delete and update button
        b=PmwTtkButtonBox.PmwTtkButtonBox(self.f,hull_background='#d4d0c8')
        b.add('Update',command=self.update,style='GREEN.TButton',width=7)
        b.add('Delete',command=self.deleteline,style='FIREB.TButton',width=7)
        b.pack(side=tkinter.LEFT,padx=3,pady=3,fill='y')        
        #position (x,y)
        smf=tkinter.Frame(self.f,bd=2,relief=tkinter.SUNKEN,background='#d4d0c8')
        self.xpos=Pmw.EntryField(smf,labelpos='w',label_text='x:',entry_width=7,validate='real',hull_background='#d4d0c8',label_background='#d4d0c8')
        self.ypos=Pmw.EntryField(smf,labelpos='w',label_text='y:',entry_width=7,validate='real',hull_background='#d4d0c8',label_background='#d4d0c8')
        self.xpos.pack(side=tkinter.LEFT,padx=2,pady=2)
        self.ypos.pack(side=tkinter.LEFT,padx=2,pady=2)
        #get position button        
        b=Button(smf,text='Get Pos',command=self.getpos,style='ORANGE.TButton',width=7)
        b.pack(side=tkinter.LEFT,padx=2,pady=2)
        smf.pack(side=tkinter.LEFT,padx=3,pady=1,fill='y')
        #type (sm/big circle, sm/big square, sm/big triangle, text)
        self.marker=Pmw.ComboBox(self.f,scrolledlist_items=self.markertypes,dropdown=1,history=0,listheight=300,
                        selectioncommand=self.checktype)
        self.marker.pack(side=tkinter.LEFT,padx=2,pady=3)
        #color
        self.color='#FFFFFF'
        self.colorsample=tkinter.Canvas(self.f,width=10,height=10,bg=self.color)
        self.colorsample.bind(sequence="<ButtonPress>",func=self.colask)     
        self.colorsample.pack(side=tkinter.LEFT,padx=3,pady=3)
        self.f.pack(side=tkinter.TOP,padx=2,pady=2,fill='x')

    def validate(self):
        self.valid=1
        if not self.xpos.valid(): self.valid=0
        if not self.ypos.valid(): self.valid=0
        if self.marker.getvalue()==():
            self.valid=0
            return
        if self.marker.getvalue()[0] not in self.markertypes: self.valid=0
        if self.textpresent and self.textfield.getvalue()=='': self.valid=0
    
    def update(self):
        self.validate()
        if self.valid: self.updatefunc(self)

    def getpos(self):
        self.posfunc(self)

    def colask(self,*args):
        new=tkinter.colorchooser.askcolor(self.color)
        if new==(None,None):
            return
        self.color=new[1]
        self.colchange()

    def colchange(self):
        self.colorsample.configure(bg=self.color)
        self.update()

    def checktype(self,*args):
        if self.marker.getvalue()==():
            if self.textpresent:
                self.textfield.destroy()
                self.textpresent=0
            self.update()
            return
        if self.textpresent==0 and self.marker.getvalue()[0]=='text':
            #create text box
            self.textfield=Pmw.EntryField(self.f,labelpos='w',label_text='Text:',entry_width=5)
            self.textfield.pack(side=tkinter.LEFT,fill='x',padx=3,pady=3)
            self.textpresent=1
            self.update()
            return
        if self.textpresent and self.marker.getvalue()[0]!='text':
            self.textfield.destroy()
            self.textpresent=0
        self.update()
 
    def deleteline(self,cb=1):
        #delete stuff
        self.f.destroy()
        #do call back
        if self.delcb is not None and cb: self.delcb(self)

    def output(self):
        out={}
        out['xpos']=self.xpos.getvalue()
        out['ypos']=self.ypos.getvalue()
        out['mark']=self.marker.getvalue()[0]
        out['color']=self.color
        out['text']=''
        if self.textpresent:
            out['text']=self.textfield.getvalue()
        return out




class HDFConstruct:
    def __init__(self, fnbase, chantuple):

        self.fnbase=fnbase
        self.chans=chantuple
        

#######################################
## Crop Class
#######################################


class cropslide:
    def __init__(self,master,dir,total,startval=None):
        self.total=total
        f=tkinter.Frame(master,background='#d4d0c8')
        f.pack(side=tkinter.TOP,padx=5,pady=5,fill='both')
        if total>=0:
            l=tkinter.Label(f,text=dir+' pixels to add:',background='#d4d0c8')
        else:
            l=tkinter.Label(f,text=dir+' pixels to crop:',background='#d4d0c8')
        l.pack(side=tkinter.TOP,padx=5,pady=2)
        inf=tkinter.Frame(f)
        inf.pack(side=tkinter.TOP)
        if dir=='Vertical':
            tl='Top'
            br='Bottom'
        else:
            tl='Left'
            br='Right'
        if startval is None:
            initval=int(abs(total)/2)
        else:
            initval=startval
        oinitval=abs(total)-initval
        self.topleft=Pmw.Counter(inf,labelpos='n',label_text=tl,datatype='integer',orient='horizontal',
                                 entry_width=4,entryfield_value=initval,entryfield_validate={'validator' : 'integer',
                        'min' : 0, 'max' : abs(total)},
                                 hull_background='#d4d0c8')
        self.topleft.component('entry').configure(state=tkinter.DISABLED)
        self.topleft.pack(side=tkinter.LEFT,padx=5,pady=2)
        self.botright=Pmw.Counter(inf,labelpos='n',label_text=br,datatype='integer',orient='horizontal',
                                 entry_width=4,entryfield_value=oinitval,entryfield_validate={'validator' : 'integer',
                        'min' : 0, 'max' : abs(total)},
                                  hull_background='#d4d0c8')
        self.botright.pack(side=tkinter.LEFT,padx=5,pady=2)
        self.botright.component('entry').configure(state=tkinter.DISABLED)        
        self.topleft.component('uparrow').bind(sequence='<Button-1>',func=self.changetl,add='+')
        self.topleft.component('downarrow').bind(sequence='<Button-1>',func=self.changetl,add='+')
        self.botright.component('uparrow').bind(sequence='<Button-1>',func=self.changebr,add='+')
        self.botright.component('downarrow').bind(sequence='<Button-1>',func=self.changebr,add='+')

    def changetl(self,event):
        nv=int(self.topleft.getvalue())
        self.botright.setvalue(abs(self.total)-nv)

    def changebr(self,event):
        nv=int(self.botright.getvalue())
        self.topleft.setvalue(abs(self.total)-nv)

    


def makenewpadline(data,newy,sh):
    newl=zeros(newy,dtype=np.float32)
    ylen=len(data)
    for j in range(newy):
        if j<sh:
            newl[j]=data[0]
        elif j>ylen-1:
            newl[j]=data[ylen-1]
        else:
            newl[j]=data[j-sh]
    return newl


            
#filterConvolve moved to globalfuncs

def subfilterconvolve(data,filter,z=0):
    #print data.shape,filter.shape
    t=data*filter
    s=sum(ravel(t))
    if z:
        return abs(s)
    else:
        return s

def shiftvert(d,s):
    if s>0:
        nr=zeros((s,d.shape[1]),dtype=np.float32)
        d=np.concatenate((d,nr),axis=0)
        d=d[s:,:]
    elif s<0:
        nr=zeros((-s,d.shape[1]),dtype=np.float32)
        d=np.concatenate((nr,d),axis=0)
        d=d[:d.shape[0]+s,:]
    return d

def shifthorz(d,s):
    s=-s
    if s>0:
        nr=zeros((d.shape[0],s),dtype=np.float32)
        d=np.concatenate((d,nr),axis=1)
        d=d[:,s:]
    elif s<0:
        nr=zeros((d.shape[0],-s),dtype=np.float32)
        d=np.concatenate((nr,d),axis=1)
        d=d[:,:d.shape[1]+s]  
    return d  

           
#############################
##   Main Class
#############################
        
class Main:
    def __init__(self,master):
        root=master
        self.root=root
        self.root.title("""Sam's Microprobe Analysis Kit""")
        self.imgwin=imgwin=master
        imgwin.protocol("WM_DELETE_WINDOW", self.killmain)
        #toggle MCA variable
        self.viewMCAplottoggle=tkinter.IntVar()
        self.viewMCAplottoggle.set(0)
        #toggle scalebar variable
        self.showscalebar=tkinter.IntVar()
        self.showscalebar.set(0)
        self.showscalebarText=tkinter.IntVar()
        self.showscalebarText.set(1)
        #toggle LargeFileEditor
        self.LargeFileEdit=tkinter.StringVar()
        self.xyflip=tkinter.IntVar()
        self.xyflip.set(0)
        self.contoursOn=tkinter.IntVar()
        self.contoursOn.set(0)
        self.CMYKOn=tkinter.IntVar()
        self.CMYKOn.set(0)
        self.nullMaskCalc=tkinter.IntVar()
        self.nullMaskCalc.set(0)
        self.isStatCalcMultiFile=tkinter.IntVar()
        self.isStatCalcMultiFile.set(0)
        self.showClusterVectors=tkinter.IntVar()
        self.showClusterVectors.set(0)
        self.PCAplotVectors=tkinter.IntVar()
        self.PCAplotVectors.set(1)
        self.XFITOPT=tkinter.StringVar()
        self.HDFCOMPRESS=tkinter.StringVar()
        self.MCASUMOPT=tkinter.StringVar()
        self.MCAXAXIS=tkinter.StringVar()
        self.dispScaleFactor=tkinter.StringVar()
        self.defaultSaveType=tkinter.StringVar()
        #self.tcimageexists=0        
        self.cfimageexists=0
        self.mfimageexists=0
        self.corplotexist=0
        self.movieView=None
        self.megaGraph=None
        self.multiCorGraph=None
        self.dataAxisAveragingDialog = None
        self.samSegmentInitialized = False

        #master class window variables
        self.correlationPlot=CorrelationPlotClass.CorrelationPlot(self.imgwin)
        print("corr initialized")
        self.mathWindow=MathWindowClass.MathWindow(self.imgwin)
        print("math window initialized")
        self.regWindow=ExportRegistrationClass.ExportRegWindow(self.imgwin)
        print("export registration window initialized")
        self.triColorWindow=TriColorWindowClass.TriColorWindow(self.imgwin)
        print("tricolor window initialized")
        self.dataSummaryWindow = DataSummaryClass.DataSummary(self.imgwin)
        print("data compress window initialized")
        self.momentWindow=MomentAnalysisClass.MomentAnalysisWindow(self.imgwin)
        print("moment window initialized")
        self.histogramWindow = MakeHistogramClass.MakeHistogramWindow(self.imgwin)
        print("histogram window initialized")
        self.xanesFitWindow=XanesFitClass.XanesFitWindow(self.imgwin)
        print("xanes fit window initialized")
        self.thresholdingWindow = ThresholdingClass.ThresholdingWindow(self.imgwin)
        print("thresholding window initialized")
        self.beamDeconvolutionWindow = BeamDeconvolutionClass.BeamDeconvolutionWindow(self.imgwin)
        print("BeamDeconvolution window initialized")
        self.particleStatisticsWindow = ParticleStatisticsClass.ParticleStatisticsWindow(self.imgwin)
        print("particleStatistics window initialized")
        self.editConcStandardsWindow = ConcentrationStandardClass.ConcentrationStandardWindow(self.imgwin)
        print("particleStatistics window initialized")
        self.multiMassWindow = MultiMassCalibrationClass.MultiMassCalibrationWindow(self.imgwin)
        print("multi mass window initialized")
        self.advancedFilterWindow = AdvancedFilteringClass.AdvancedFilteringWindow(self.imgwin)
        print("advanced filter window initialized")
        self.customKernelWindow = CustomKernelClass.CustomKernelWindow(self.imgwin)
        print("custom kernel window initialized")
        self.scriptEditorWindow = BatchMode.ScriptEditorWindow(self.imgwin)
        

        #Display
        self.maindisp=Display.Display(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,tcrefresh=self.tcrefresh,callback=[None,None,self.displaySpectrumGraph,self.dispAddMarker],rowcb=self.returnPRS,sf=self.dispScaleFactor)
        self.maindisp.main.withdraw()
        #Menu bar
        menubar=PmwTtkMenuBar.PmwTtkMenuBar(imgwin)
        if os.sys.platform=='win32': menubar.component('hull').configure(bg='#d4d0c8')
        #file menu
        menubar.addmenu('File','')
        menubar.addmenuitem('File','command',label='Open Single',command=self.fileme)
        menubar.addmenuitem('File','command',label='Open Many',command=self.multifileme)
        menubar.addmenuitem('File','command',label='Open Special',command=self.filemeSpecial)        
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Import From File',command=self.importchannel)
        menubar.addmenuitem('File','command',label='Rapid Import From File',command=self.importchannel_rapid)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Export to tab',command=self.exportchannelToTab)
        menubar.addmenuitem('File','command',label='Export with Resize',command=self.exportchannelWithResize)
        menubar.addmenuitem('File','command',label='Export with Channel Registration',command=self.exportchannelWithRegistration)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Convert SSRL Mosaic',command=self.createSSRLMosaic)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='XRD Import',command=self.importXRD)
        menubar.addmenuitem('File','command',label='Create from MCA',command=self.createDataFromMCA)
        menubar.addmenuitem('File','command',label='Open from LA-MS',command=self.createDataFromLAMS)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Close File',command=self.closeFileTab)
        menubar.addmenuitem('File', 'command', label='Close All Files', command=self.closeAllFileTabs)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Set temp file Directory',command=self.setTempDir)
        menubar.addmenuitem('File','separator')
        menubar.addcascademenu('File','Large File Edits')
        menubar.addmenuitem('Large File Edits','radiobutton',label='None',command=tkinter.DISABLED,variable=self.LargeFileEdit)        
        menubar.addmenuitem('Large File Edits','radiobutton',label='Edit',command=tkinter.DISABLED,variable=self.LargeFileEdit)
        menubar.addmenuitem('Large File Edits','radiobutton',label='Select',command=tkinter.DISABLED,variable=self.LargeFileEdit)        
        menubar.addmenuitem('Large File Edits','radiobutton',label='Single',command=tkinter.DISABLED,variable=self.LargeFileEdit)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Create SUPER file header',command=self.createSUPERhead)        
        menubar.addmenuitem('File','separator')        
        menubar.addmenuitem('File','command',label='Save Data',command=self.saveprocdata)

        menubar.addcascademenu('File','Default Save Type')
        menubar.addmenuitem('Default Save Type','radiobutton',label='HDF5',command=tkinter.DISABLED,variable=self.defaultSaveType)        
        menubar.addmenuitem('Default Save Type','radiobutton',label='ASCII',command=tkinter.DISABLED,variable=self.defaultSaveType)        
        menubar.addmenuitem('Default Save Type','radiobutton',label='Both',command=tkinter.DISABLED,variable=self.defaultSaveType)        

        menubar.addmenuitem('File','separator')                
        menubar.addmenuitem('File','command',label='Save Display',command=self.savedisplayasjpg)
        menubar.addmenuitem('File','command',label='Save Hi-Res Display',command=self.savedisplayasHD)
        menubar.addmenuitem('File','command',label='Save Display as Array',command=self.savedisplayasNPY)
        menubar.addmenuitem('File','command',label='Save Many Displays',command=self.asksavemanydisplays)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Save TriColor Display',command=self.savetcdisplayasjpg)        
        menubar.addmenuitem('File','command',label='Save Hi-Res TriColor Display',command=self.savetcdisplayasHD)
        menubar.addmenuitem('File','separator')        
        menubar.addmenuitem('File','command',label='Export X-Y Plot',command=self.savexyplotdata)
#        menubar.addmenuitem('File','command',label='Export Data as Text',command=self.exporttextdata)        
        menubar.addmenuitem('File','command',label='Save CT sections',command=self.saveCTsections)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Save Recent PCA Analysis',command=self.savePCArecent)
        menubar.addmenuitem('File','command',label='Load Recent PCA Analysis',command=self.loadPCArecent)
        menubar.addmenuitem('File','separator')

        menubar.addmenuitem('File','command',label='Rename Selected Channel',command=self.askrenamechannel)
        menubar.addmenuitem('File','command',label='Remove Selected Channel',command=self.askremovechannel)
        menubar.addmenuitem('File','command',label='Remove Multiple Channels',command=self.askremovemultichannel)
        menubar.addmenuitem('File','separator')
        menubar.addmenuitem('File','command',label='Export scan area coordinates',command=self.exportScanArea)
        menubar.addmenuitem('File','command',label='Export scan area to dataqueue',command=self.exportScanAreaDQ)
        menubar.addmenuitem('File','separator')        
        menubar.addmenuitem('File','command',label='Quit',command=self.killmain)
        menubar.addmenu('View','')
        menubar.addmenuitem('View','command',label='Image Map',command=self.showmap)
        menubar.addmenuitem('View','command',label='Correlation Plot',command=self.startcorplot)
        menubar.addmenuitem('View','command',label='Cross Correlation Plot',command=self.startmegaCrossPlot)
        menubar.addmenuitem('View','command',label='TriColor Plot',command=self.starttcplot)
        menubar.addmenuitem('View','separator')
        menubar.addmenuitem('View','checkbutton',label='Convert Palette to CMYK',command=self.doCMYK,variable=self.CMYKOn)
        menubar.addmenuitem('View','separator')
        menubar.addmenuitem('View','command',label='Change Min Size',command=self.changeMIN)
        menubar.addcascademenu('View','Apply Scaling Factor')
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='0.1x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='0.25x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='0.5x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='0.75x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='1x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='1.5x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        menubar.addmenuitem('Apply Scaling Factor','radiobutton',label='2x',command=self.domapimagefromscaselect,variable=self.dispScaleFactor)
        self.dispScaleFactor.set('1x')
        menubar.addmenuitem('View','separator')
        menubar.addmenuitem('View','command',label='Crossfader',command=self.startCrossFader)
        menubar.addmenuitem('View','command',label='Multifader',command=self.startMultiFader)
        menubar.addmenuitem('View','separator')
        menubar.addmenuitem('View','command',label='View Header',command=self.viewheader)

        menubar.addmenu('Process','')  
        menubar.addmenuitem('Process','command',label='Open Math Window',command=self.startmathwin)
        menubar.addmenuitem('Process','separator')
        menubar.addmenuitem('Process', 'command', label='Perform Log Transforms', command=self.startlogTrans)
        menubar.addmenuitem('Process', 'command', label='Perform Vert Flips', command=self.startvertTrans)
        menubar.addmenuitem('Process', 'command', label='Perform Horz Flips', command=self.starthorzTrans)
        menubar.addmenuitem('Process', 'command', label='Normalize to Time', command=self.startTimeTrans)
        menubar.addmenuitem('Process','separator')

        menubar.addmenuitem('Process','command',label='Thresholding',command=self.dothreshold)            
        menubar.addmenuitem('Process','command',label='Advanced Filtering',command=self.startadvfilterwin)
        menubar.addmenuitem('Process','command',label='Custom Kernel Convolution',command=self.startcustomkernels)
        menubar.addmenuitem('Process','command',label='Calculate Row Balance',command=self.rowBalanceCalc)
        menubar.addmenuitem('Process','command',label='Apply Row Balance',command=self.rowBalance)
        menubar.addmenuitem('Process','separator')  

        menubar.addmenuitem('Process','command',label='Use Current Channel as Mask',command=self.useImageasMask)
        menubar.addmenuitem('Process','command',label='Create New channel from Mask',command=self.createNewChannelFromMask)
        menubar.addmenuitem('Process','command',label='Add Mask to Channel',command=self.addMaskToChannel)
        menubar.addmenuitem('Process','command',label='Remove Mask from Channel',command=self.delMaskToChannel)
        menubar.addmenuitem('Process','command',label='Invert Mask Selection',command=self.invertMaskSelection)
        menubar.addmenuitem('Process','command',label='Create Mask from Zoom',command=self.makeMaskFromZoom)
        menubar.addmenuitem('Process','separator')      

        menubar.addmenuitem('Process','command',label='Resolution Deconvolution',command=self.startDeconv)       
        menubar.addmenuitem('Process','command',label='Change Resolution',command=self.startResChange)                 
        menubar.addcascademenu('Process','File Transformation Operations')
        menubar.addmenuitem('File Transformation Operations','command',label='Rotate File',command=self.tf_rotateFile)         
        menubar.addmenuitem('File Transformation Operations','command',label='Vertical Flip File',command=self.tf_vertflipFile)         
        menubar.addmenuitem('File Transformation Operations','command',label='Horizontal Flip File',command=self.tf_horzflipFile)         
        menubar.addmenuitem('Process','separator')

        menubar.addmenuitem('Process','command',label='Interpolate Missing Rows',command=self.doRowInterpolation)
        menubar.addmenuitem('Process','command',label='Interpolate Missing Rows (All Channels)',command=self.doRowInterpolationAll)
        menubar.addmenuitem('Process','command',label='Manual Pix Row Shift',command=self.startPixRowShift)
        menubar.addmenuitem('Process','command',label='Pix Interlace Correction',command=self.startPixInterlace)        
        menubar.addmenuitem('Process','command',label='Automatic PixRow Image Shift',command=self.startAutoPixRowShift)                
        menubar.addmenuitem('Process','command',label='Low Signal Shift',command=self.startLowSignalShift)                        
        menubar.addmenuitem('Process','command',label='Remove Columns',command=self.startRemoveColumns)        
        menubar.addmenuitem('Process','separator')
#        menubar.addmenuitem('Process','command',label='Stack FT Align/Shifts',command=self.startAlignStack)                 
        menubar.addmenuitem('Process','command',label='Stack Alignment',command=self.startAlignStackSequence)                 
        menubar.addmenuitem('Process','command',label='Apply Last Alignment',command=self.startApplyAlignStackSequence)   
        menubar.addmenuitem('Process','separator')
#        menubar.addmenuitem('Process','separator') 

        menubar.addmenuitem('Process','command',label='Sum MultiElement',command=self.multisum)
        menubar.addmenuitem('Process','command',label='Channel Stitching',command=self.openchanstitch)
        menubar.addmenuitem('Process','separator')
        menubar.addcascademenu('Process','Deadtime')
        menubar.addmenuitem('Deadtime','command',label='Apply Deadtime Correction',command=self.deadtimecorrection)
        menubar.addmenuitem('Deadtime','command',label='Write SIXPACK Deadtime File',command=self.writeSIXPACKdtf)


        menubar.addmenu('Analyze','')  
        menubar.addmenuitem('Analyze','command',label='XANES Fitting',command=self.startXANESfitting)
        menubar.addcascademenu('Analyze','XANES Fit Options')
        menubar.addmenuitem('XANES Fit Options','radiobutton',label='LS Fit',command=tkinter.DISABLED,variable=self.XFITOPT)
        menubar.addmenuitem('XANES Fit Options','radiobutton',label='NNLS Fit-A',command=tkinter.DISABLED,variable=self.XFITOPT)
        menubar.addmenuitem('XANES Fit Options','radiobutton',label='NNLS Fit-B',command=tkinter.DISABLED,variable=self.XFITOPT)
        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','command',label='Set I0 Channel',command=self.seti0channel)
        menubar.addmenuitem('Analyze','command',label='Set Time Channel',command=self.setTIMEchannel)
        menubar.addmenuitem('Analyze','separator')
        menubar.addcascademenu('Analyze','Quantification')
        menubar.addmenuitem('Quantification','command',label='Standard Calibrant Editor',command=self.startStandardEditor)
        menubar.addmenuitem('Quantification','command',label='Quantitative Calibration',command=self.startMultiMass)
        menubar.addmenuitem('Quantification','command',label='Do Quantify',command=self.doQuantify)
        menubar.addmenuitem('Quantification','separator')
        menubar.addmenuitem('Quantification','checkbutton',label='Use MultiFile Definitions',variable=self.isStatCalcMultiFile)
        menubar.addmenuitem('Quantification','separator')
        menubar.addcascademenu('Quantification','Legacy Quant')        
        menubar.addmenuitem('Legacy Quant','command',label='Quantitative Analysis',command=self.quantoptions)
        menubar.addmenuitem('Legacy Quant','command',label='Homogenous Thickness Correction',command=self.quantThickness)        
        menubar.addmenuitem('Legacy Quant','command',label='Add Data to QuantFile',command=self.addquantcalc)
        
        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','command',label='PCA Analysis on MCAs',command=self.startPCA)
        menubar.addmenuitem('Analyze','command',label='PCA Analysis on Channels',command=self.startPCAmini)
        menubar.addmenuitem('Analyze','command',label='Fix PCA MAX',command=self.editPCAMAX)
        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','checkbutton',label='Report Clustering Vectors',variable=self.showClusterVectors)
        menubar.addmenuitem('Analyze','checkbutton',label='Plot PCA Vectors',variable=self.PCAplotVectors)
        menubar.addmenuitem('Analyze','command',label='Show Recent PCA Vectors',command=self.replotPCAplotVectors)
        menubar.addmenuitem('Analyze','command',label='PCA Vector Analysis',command=self.startPCAVectorAnalysis)
        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','command',label='Save Recent PCA Analysis',command=self.savePCArecent)
        menubar.addmenuitem('Analyze','command',label='Load Recent PCA Analysis',command=self.loadPCArecent)
        
        
        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','command',label='Average Data Along Axis',command=self.datacompressplot)
        menubar.addmenuitem('Analyze','command',label='Make Radial Profile',command=self.makeradialstart)      
        menubar.addmenuitem('Analyze','separator')        
        menubar.addmenuitem('Analyze','command',label='Data Summary',command=self.datacompresssummary)
        menubar.addmenuitem('Analyze','checkbutton',label='Contour Summary',command=self.datacontoursToggle,variable=self.contoursOn)
        menubar.addmenuitem('Analyze','command',label='Moment Analysis',command=self.startdatamoments)
        menubar.addmenuitem('Analyze','command',label='Make Histogram',command=self.starthistogram)
        menubar.addmenuitem('Analyze','separator')        
        menubar.addmenuitem('Analyze','command',label='Spectrum Maker',command=self.startspectrumMaker)
        menubar.addmenuitem('Analyze','command',label='Export ROI Spectra',command=self.exportFFSpectra)
        menubar.addmenuitem('Analyze','command',label='Export ROI/PCA Spectra',command=self.exportFFPCASpectra)
        menubar.addmenuitem('Analyze','command',label='Full Field Maker',command=self.startFFMaker)

        menubar.addmenuitem('Analyze', 'separator')
        menubar.addmenuitem('Analyze', 'command', label='Particle Watershed Statistics', command=self.startPartWaterStats)
        menubar.addmenuitem('Analyze', 'command', label='Particle Statistics', command=self.startPartStats)
        menubar.addmenuitem('Analyze', 'command', label='Masked Statistics', command=self.startMaskStats)
        menubar.addcascademenu('Analyze','Segmentation')
        menubar.addmenuitem('Segmentation','command',label='Random Forest Segmentation',command=self.RFsegment)        
        if samIsInstalled is True:
            menubar.addmenuitem('Segmentation','separator')            
            menubar.addmenuitem('Segmentation','command',label='Initialize SAM',command=self.initSAMParams)
            menubar.addmenuitem('Segmentation','command',label='Create AutoMasks',command=self.startSAMAuto)
            menubar.addmenuitem('Segmentation','command',label='Detail Mask',command=self.startSAMDetailed)
            menubar.addmenuitem('Segmentation','separator')
            menubar.addmenuitem('Segmentation','command',label='Use Last Segmentation Result',command=self.reuseSAMResult)
            menubar.addmenuitem('Segmentation','separator')
            menubar.addmenuitem('Segmentation','command',label='Save Segmentation Result',command=self.saveSAMResulttoFile)
            menubar.addmenuitem('Segmentation','command',label='Load Segmentation Result',command=self.loadSAMResultfromFile)
        
        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','command',label='Set Minimum Particle Pixels',command=self.setPartROIThresh)
        menubar.addmenuitem('Analyze','command',label='Set Watershed Pixel Distance',command=self.setPartWaterThresh)
        menubar.addmenuitem('Analyze','checkbutton',label='Use NULL Mask',variable=self.nullMaskCalc)
        menubar.addmenuitem('Analyze','checkbutton',label='Do MultiFile Statistics',variable=self.isStatCalcMultiFile)

        menubar.addmenuitem('Analyze','separator')
        menubar.addmenuitem('Analyze','command',label='3D Stereo Image Maker',command=self.open3Dmaker)
        #menubar.addmenuitem('Analyze','separator')
        #menubar.addmenuitem('Analyze','command',label='CT Options',command=self.showCToptions)
        #menubar.addmenuitem('Analyze','command',label='Compute CT Section',command=self.doCTsection)

        menubar.addmenu('Workflow','')        
        menubar.addmenuitem('Workflow','command',label='Workflow Editor',command=self.defineCustomBatch)
        menubar.addmenuitem('Workflow','command',label='Reload Workflows',command=self.reloadAddinBatch)
        menubar.addmenuitem('Workflow','command',label='Save Workflows',command=self.saveAddinBatch)
        #menubar.addmenuitem('Workflow','separator')
        #menubar.addmenuitem('Workflow','command',label='EdgeRemoval',command=self.batchEdgeRemoval)
        #menubar.addmenuitem('Workflow','command',label='Create ROI mask from EdgeRemoval',command=self.batchEdgeRemovalMask)
        #menubar.addmenuitem('Workflow','command',label='Blur Ka Channels',command=self.batchBlurAll)
        menubar.addmenuitem('Workflow','separator')


        menubar.addmenu('Legend','')        
        menubar.addmenuitem('Legend','command',label='View Colormap',command=self.viewcolormap)
        menubar.addmenuitem('Legend','command',label='View TriColormap',command=self.viewtricolormap)
        menubar.addmenuitem('Legend','separator')
        menubar.addmenuitem('Legend','command',label='Save Colormap',command=self.savecolormap)
        menubar.addmenuitem('Legend','command',label='Save TriColormap',command=self.savetricolormap)
        menubar.addmenuitem('Legend','separator')
        menubar.addmenuitem('Legend','command',label='Plot Marker Options',command=self.plotmarkermain)
        menubar.addmenuitem('Legend','separator')        
        menubar.addmenuitem('Legend','checkbutton',label='Show Scalebar',command=self.doscaleup,variable=self.showscalebar)
        menubar.addmenuitem('Legend','checkbutton',label='Show Scalebar Text',command=self.doscaleup,variable=self.showscalebarText)
        menubar.addmenuitem('Legend','command',label='Place Scalebar',command=self.placethescalebar)
        menubar.addmenuitem('Legend','command',label='Default Scalebar Position',command=self.defaultscalebar)
        
        menubar.addmenu('MCA Spectra','')        
        menubar.addmenuitem('MCA Spectra','command',label='Define MCA file',command=self.getMCAfile)
        menubar.addmenuitem('MCA Spectra','separator')
        menubar.addcascademenu('MCA Spectra','MCA Sum Options')
        menubar.addmenuitem('MCA Sum Options','radiobutton',label='Average',command=tkinter.DISABLED,variable=self.MCASUMOPT)
        menubar.addmenuitem('MCA Sum Options','radiobutton',label='Sum',command=tkinter.DISABLED,variable=self.MCASUMOPT)
        menubar.addmenuitem('MCA Spectra','separator')
        menubar.addcascademenu('MCA Spectra','MCA View Options')
        menubar.addmenuitem('MCA View Options','radiobutton',label='Bins',command=tkinter.DISABLED,variable=self.MCAXAXIS)
        menubar.addmenuitem('MCA View Options','radiobutton',label='Spectrum',command=self.setMCAslopeValue,variable=self.MCAXAXIS)
        menubar.addmenuitem('MCA Spectra','separator')        
        menubar.addmenuitem('MCA Spectra','command',label='Correct MCA file',command=self.MCAdeglitch)
        menubar.addmenuitem('MCA Spectra','command',label='Set pixel per line offset',command=self.MCAsetpixofs)
        menubar.addmenuitem('MCA Spectra','command',label='Set pixel first line offset',command=self.MCAset1stpixofs)
        menubar.addmenuitem('MCA Spectra','separator')
        ##LEGACY
        ##menubar.addmenuitem('MCA Spectra','command',label='Translate HDF file',command=self.HDFtoMCA)

        menubar.addmenuitem('MCA Spectra','command',label='Construct Xspress3 HDFs',command=self.constructFullXspressHDF)
        menubar.addcascademenu('MCA Spectra','HDF5 Write Compression')
        menubar.addmenuitem('HDF5 Write Compression','radiobutton',label='None',command=tkinter.DISABLED,variable=self.HDFCOMPRESS)
        menubar.addmenuitem('HDF5 Write Compression','radiobutton',label='GZIP 4',command=tkinter.DISABLED,variable=self.HDFCOMPRESS)
        menubar.addmenuitem('HDF5 Write Compression','radiobutton',label='GZIP 9',command=tkinter.DISABLED,variable=self.HDFCOMPRESS)
        menubar.addmenuitem('HDF5 Write Compression','radiobutton',label='LZF',command=tkinter.DISABLED,variable=self.HDFCOMPRESS) 
        menubar.addmenuitem('MCA Spectra','command',label='Edit bidirection of HDFs',command=self.swapdirXspressHDF)
        menubar.addmenuitem('MCA Spectra','separator') 
        menubar.addmenuitem('MCA Spectra','command',label='Translate to Binary MCA',command=self.MCAwritebinary)
        menubar.addmenuitem('MCA Spectra','command',label='Export MCA to HDF',command=self.MCAwriteHDF5)
        menubar.addmenuitem('MCA Spectra','separator')        
        menubar.addmenuitem('MCA Spectra','command',label='Set MCA multi-file lines',command=self.MCAlinestep)
        menubar.addmenuitem('MCA Spectra','command',label='Split MCA multi-files',command=self.MCAlinesplit)        
        menubar.addmenuitem('MCA Spectra','separator')
        menubar.addmenuitem('MCA Spectra','command',label='Get MCA from mask area',command=self.getMCAfrommask)
        menubar.addmenuitem('MCA Spectra','command',label='Get MCAs from cluster map',command=self.getMCAfromcluster)        
        menubar.addmenuitem('MCA Spectra','separator')
        menubar.addmenuitem('MCA Spectra','command',label='Movie',command=self.makeMCAmovie)
        menubar.addmenuitem('MCA Spectra','separator') 
        menubar.addcascademenu('MCA Spectra','FTIR Support')
        menubar.addmenuitem('FTIR Support','command',label='Set IR Corrections/Normalization',command=self.IRprocessingWindow) 
        menubar.addmenuitem('FTIR Support','command',label='Save IR Settings',command=self.saveIRsettings) 
        menubar.addmenuitem('FTIR Support','command',label='Load IR Settings',command=self.loadIRsettings)                
        menubar.addmenuitem('FTIR Support','separator')         
        menubar.addmenuitem('FTIR Support','command',label='Correct and Save FTIR data',command=self.IRprocessAll)        
        menubar.addmenuitem('MCA Spectra','separator') 
        menubar.addcascademenu('MCA Spectra','Color Support')
        menubar.addmenuitem('Color Support','command',label='Construct HDFs from OOMap',command=self.constructFullXspressHDF)                
        menubar.addmenuitem('Color Support','command',label='Construct HDFs from OOXAS',command=self.constructXASXspressHDF)        
        menubar.addmenuitem('Color Support','separator')         
        menubar.addmenuitem('Color Support','command',label='Convert UV-MCA to RGB',command=self.convertMCAtoRGB)
        menubar.addmenuitem('Color Support','command',label='Convert UV-MCA to XYZ',command=self.convertMCAtoXYZ)  
        menubar.addmenuitem('Color Support','command',label='Convert UV-MCA to Lab',command=self.convertMCAtoLAB)    
        menubar.addmenuitem('Color Support','separator') 
        menubar.addmenuitem('Color Support','command',label='Convert to RGB Textfile',command=self.convertMCAtoRGBtext)
        menubar.addmenuitem('Color Support','command',label='Convert to XYZ Textfile',command=self.convertMCAtoXYZtext)  
        menubar.addmenuitem('Color Support','command',label='Convert to Lab Textfile',command=self.convertMCAtoLABtext) 
        menubar.addmenuitem('MCA Spectra','separator')         
        menubar.addmenuitem('MCA Spectra','checkbutton',label='View MCA',command=tkinter.DISABLED,variable=self.viewMCAplottoggle)
#more?
        menubar.addmenu('Help','',side=tkinter.RIGHT)
        menubar.addmenuitem('Help','command',label='About',command=self.callprogramabout)
        menubar.addmenuitem('Help','separator')
        menubar.addmenuitem('Help','command',label='Help',command=self.showclickhelp)
        menubar.pack(side=tkinter.TOP,fill=tkinter.X)
        self.mainmenubar=menubar
        self.XFITOPT.set('NNLS Fit-A')
        self.LargeFileEdit.set('Edit')
        self.MCASUMOPT.set('Average')
        self.MCAXAXIS.set('Bins')
        self.HDFCOMPRESS.set('GZIP 9')
        self.defaultSaveType.set('HDF5')
        self.currentMCAXraySlope = 10.0
        self.currentMCAXvalues = None

        #notebook
        self.filenb=PmwTtkNoteBook.PmwTtkNoteBook(imgwin,raisecommand=self.clickOnFileTab)
        self.filenb.configure(hull_background='#d4d0c8',hull_width=500,hull_height=30)
        self.filenb.pack(side=tkinter.TOP,fill='both',expand=1,padx=3,pady=3)
        #nbp=self.filenb.add('Individuals',page_background='#d4d0c8')
        
        #Data file handling variables
        self.dataFileBuffer={}
        self.activeFileBuffer=None
        self.workingFileBufferNames=[]
        
        #file bar
        mf=tkinter.Frame(imgwin,background='#d4d0c8')
        mf.pack(side=tkinter.TOP)
        topf=tkinter.Frame(mf,background='#d4d0c8')
        topf.pack(side=tkinter.LEFT)
        filebar=tkinter.Frame(topf, relief=tkinter.SUNKEN,bd=2,background='#d4d0c8')
        self.fileentry=Pmw.EntryField(filebar, label_text="Data File:",labelpos=tkinter.W,validate=None,entry_width=68,command=self.load_data_file,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.fileentry.pack(side=tkinter.LEFT,padx=5,pady=2,fill=tkinter.X)
        b=Button(filebar,text="Open",command=self.multifileme,style='OPEN.TButton',width=7)
        b.pack(side=tkinter.LEFT,padx=2,pady=2)
        b=Button(filebar,text="Load",command=self.load_data_file,style='GREEN.TButton')
        b.pack(side=tkinter.LEFT,padx=2,pady=2)
        filebar.pack(side=tkinter.TOP,padx=2,pady=2,fill=tkinter.X)
        #SCA Channels
        nf=tkinter.Frame(topf,background='#d4d0c8')
        nf.pack(side=tkinter.TOP,fill='both')
        self.datachan=Pmw.ScrolledListBox(nf,labelpos='n',label_text='Data Channels',listbox_height=15,
                        selectioncommand=self.domapimagefromscaselect,listbox_exportselection=0,
                        hull_background='#d4d0c8',label_background='#d4d0c8')
        #bind to move
        self.datachan.bind(sequence="<Up>", func=self.arrowdatachan)
        self.datachan.bind(sequence="<Down>", func=self.arrowdatachan)
        if sys.platform=='darwin':
            self.datachan.component('listbox').bind(sequence="<Button-2>",func=self.domapimagenewwindow)
        else:
            self.datachan.component('listbox').bind(sequence="<Button-3>",func=self.domapimagenewwindow)
        self.datachan.pack(side=tkinter.LEFT,fill='both')
        #Action Buttons
        mbf=tkinter.Frame(nf,background='#d4d0c8')
        mbf.pack(side=tkinter.LEFT,fill='both')
        w=15
        #processed data
        b=PmwTtkButtonBox.PmwTtkButtonBox(mbf,label_text="Proccess File",labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',frame_background='#d4d0c8')
        #tempedit
        #b.add('Import Data From',command=self.importchannel_rapid,style='RED.TButton',width=tkinter.W)
        b.add('Import Data From',command=self.importchannel_rapid,style='RED.TButton',width=w)
        b.add('Save Processed',command=self.saveprocdata,style='SBLUE.TButton',width=w)
        b.pack(side=tkinter.TOP,padx=2,pady=2)
        #Analysis actions
        b=PmwTtkButtonBox.PmwTtkButtonBox(mbf,label_text="Analysis Procedures",labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',frame_background='#d4d0c8')
        b.add('Correlation Plots',command=self.startcorplot,style='BROWN.TButton',width=w)
        b.add('TriColor Plots',command=self.starttcplot,style='BROWN.TButton',width=w)        
        b.pack(side=tkinter.TOP,padx=2,pady=2)
        b=PmwTtkButtonBox.PmwTtkButtonBox(mbf,hull_background='#d4d0c8')
        b.add('Map Math',command=self.startmathwin,style='GREEN.TButton',width=w)
        b.add('PCA Analysis',command=self.startPCAmini,style='GREEN.TButton',width=w)        
        b.pack(side=tkinter.TOP,padx=2,pady=2)
        #Mask Plot actions
        self.usemaskinimage=0
        b=PmwTtkButtonBox.PmwTtkButtonBox(mbf,label_text="Map Mask Actions",labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',frame_background='#d4d0c8')
        b.add('Use Mask',command=self.usemask,style='LGREEN.TButton',width=w)
        b.add('Ignore Mask',command=self.ignoremask,style='RBLUE.TButton',width=w)
        b.pack(side=tkinter.TOP,padx=2,pady=2)        
        #LOGO
        imlogo=Image.open("smak_sm.jpg")
        imlogo.load()
        #imlogo=imlogo.resize((211,252))
        logo=ImageTk.PhotoImage(imlogo)
        canlogo=tkinter.Label(nf,image=logo,bd=2,relief=tkinter.RIDGE,background='#d4d0c8')
        canlogo.image=logo
        canlogo.pack(side=tkinter.LEFT,padx=10,pady=2)
        #Status Bar
        botfr=tkinter.Frame(imgwin,background='#d4d0c8')
        botfr.pack(side=tkinter.TOP,fill=tkinter.X)        
        self.status=tkinter.Label(botfr,text="",bd=2,relief=tkinter.RAISED,anchor=tkinter.W,fg='blue',background='#d4d0c8')
        self.status.pack(side=tkinter.LEFT,fill=tkinter.X,expand=1)
        globalfuncs.setstatus(self.status,"Ready")
        self.mask=Mask.MaskClass()
       
        #could initialize corr plot here
        self.useMaskforCorPlotData=tkinter.IntVar()
        self.useMaskforCorPlotData.set(0)
        self.corplotSQRT=tkinter.IntVar()
        self.corplotSQRT.set(0)
        #JOY COME BACK AND REMOVE THESE
        self.tcmarkerlist={}
        self.tcrangedict={}
        self.plotmarkerexist=0
        self.plotmarkerlist=[]
        self.hasdata=0
        self.tclegendexist=0
        self.IRwinexist=0
        self.MCAviewexist=0
        self.MCAfilename=''
        self.MCAzoomstack=[]
        #MCA data init
        self.clearMCABuffers()
        self.PCAzoomstack=[]
        self.PCAlastprop=None
        self.PCAlastevect=None
        self.PCAlastchans=[]
        self.PCAcompMAXNO=0
        self.PCAcompMAXFixed=False
        self.PCAhdf5fout=None
        self.MCAbindialogexist=0
        self.PCAdatatype='none'         
        #self.PCAviewexist=0
        self.PCAViewWindow=PCAAnalysisClass.PCAFullWindow(self.imgwin)
        print("PCA full window initialized")
        #self.PCArawdata=None
        #self.PCAdataLoaded=0
        self.deadtimedialogexist=0
        self.i0chandialogexist=0
        self.TIMEchandialogexist=0
        #self.fitXANESactive=0
        self.MCAoffset=1
        self.MCAnofiles=1
        self.MCApixoffs=0
        self.MCA1stpixoffs=0
        self.mcamaxno=2048
        self.stitchprev=None
        self.stereoprev=None
        self.crossFader=None
        self.megaX=None
        self.multiFader=None
        self.DTICRchanval=-1
        self.deadtimevalue=None
        self.fullfieldzoom=None
        self.lastAlignStack=[]
        #spectrum maker
        self.spectrumSwitch=0
        self.spectrumXvalues=[]
        self.specX=None
        #ROIs particles
        self.partstatROIflag=0
        self.partROIThresh=5
        self.partWaterThresh=20
        #CT opts
        self.CTdialogexist=0
        self.CTtypevar=0
        self.CTringvar=0
        self.CTfiltervar='Shepp_Logan'
        self.CTcalctype='FT'
        self.CTaccvaluevar=0
        self.CTtopvaluevar=0
        self.CTbotvaluevar=0        
        self.CTairvaluevar=5
        self.CTringwidthvar=5
        self.CTdocentvar=1
        self.CTcentpixvar=0
        self.CTfiltwidth=0
        self.CTfiltd=1
        self.CTdispdict={}
        self.lastSAMmask=None

        self.rat=None
        #deadtime init
        self.dodt=tkinter.IntVar()
        self.dodt.set(0)
        #i0 init
        self.doI0c=tkinter.IntVar()
        self.doI0c.set(0)
        #changes made
        self.changes=0
        self.filedir=FileTracker.FileTracker("smakpath.pth")
        self.workingdir=FileTracker.FileTracker("smakworkpath.pth")
        self.workingdir.get()

        self.pyMCAlastconfig=None
        self.pyMCAparamDialog=None
        self.pyMCAsilent={}
        self.FTIRgroup=None

        self.readAddins(first=True)
        
        self.showpage=True

        #TEMP FOR DEBUG
        #p=pyMcaParamGUI.PyMcaParameterDialog(root)


    def killmain(self):
        #print self.changes
        self.filedir.save()
        self.closeALLwindows()
        rv = self.closeAllFileTabs()
        if rv==-1: return
        self.imgwin.focus_force()
        self.root.destroy()

        
    def closeAllFileTabs(self):
        tlist=list(self.dataFileBuffer.keys())
        for a in tlist:
            self.showpage=False
            self.filenb.selectpage(a)
            if self.dataFileBuffer[a]['changes']:
                if not tkinter.messagebox.askyesno("Changes Made", "Unsaved changes made to data! Discard?"):
                    return -1
            self.closeFileTab(ignore=True)
        self.showpage=True

    def closeALLwindows(self):
        #JOY come back and double check that all habve been correctly updated
        if self.correlationPlot.exist: self.correlationPlot.win.destroy()
        if self.plotmarkerexist: self.killplotmarkerwin()
        if self.PCAViewWindow.exist: self.PCAViewWindow.kill()
        if self.MCAviewexist: self.killMCAplot()
        if self.IRwinexist: self.killIRwin()
        #JOY 
        if self.dataAxisAveragingDialog is not None:
            self.dataAxisAveragingDialog.kill()
        if self.dataSummaryWindow.exist: self.dataSummaryWindow.kill()
        if self.histogramWindow.exist: self.histogramWindow.kill()
        if self.momentWindow.exist: self.momentWindow.kill()
        if self.xanesFitWindow.exist: self.xanesFitWindow.kill()
        if self.particleStatisticsWindow.exist: self.particleStatisticsWindow.kill()

        if self.maindisp.linegraphpresent: self.maindisp.newlineplot.destroy()
        if self.maindisp.linegraph2present: self.maindisp.newlineplot2.destroy()
        if self.maindisp.linegraph3present: self.maindisp.newlineplot3.destroy()
        if self.maindisp.linegraph4present: self.maindisp.newlineplot4.destroy()

        self.correlationPlot.exist=0
        self.plotmarkerexist=0
        self.PCAViewWindow.exist=0
        self.MCAviewexist=0
        self.IRwinexist=0
        self.maindisp.linegraphpresent=0
        self.maindisp.linegraph2present=0
        self.maindisp.linegraph3present=0

    def callprogramabout(self):
        programabout(self.root)

    def setTempDir(self):
        #get dirname
        r=tkFileDialogDir.choose_directory(initialdir=self.workingdir.get())
        #check for os.setp at end
        self.workingdir.set(r)
        self.workingdir.save()

    def arrowdatachan(self,event):
        dlist=self.datachan.get()
        ind=dlist.index(self.datachan.getvalue()[0])#index(ACTIVE)
        if event.keysym=='Down':
            ind=ind+1
        if event.keysym=='Up':
            ind=ind-1
        if ind<0: ind=0
        if ind>len(dlist)-1: ind=len(dlist)-1
        self.datachan.setvalue(dlist[ind])
        self.datachan.see(ind)
        self.domapimagefromscaselect()
    """
    def arrowcpxchan(self,event):
        dlist=self.cpxchan.get()
        ind=dlist.index(self.cpxchan.getvalue()[0])
        if event.keysym=='Down':
            ind=ind+1
        if event.keysym=='Up':
            ind=ind-1
        if ind<0: ind=0
        if ind>len(dlist)-1: ind=len(dlist)-1
        self.cpxchan.setvalue(dlist[ind])
        self.cpxchan.see(ind)
        self.checkcorplotx()

    def arrowcpychan(self,event):
        dlist=self.cpychan.get()
        ind=dlist.index(self.cpychan.getvalue()[0])
        if event.keysym=='Down':
            ind=ind+1
        if event.keysym=='Up':
            ind=ind-1
        if ind<0: ind=0
        if ind>len(dlist)-1: ind=len(dlist)-1
        self.cpychan.setvalue(dlist[ind])
        self.cpychan.see(ind)
        self.checkcorploty()

    """

    # def arrowmathA(self,event):
    #     dlist=self.mathA.get()
    #     ind=dlist.index(self.mathA.getvalue()[0])
    #     if event.keysym=='Down':
    #         ind=ind+1
    #     if event.keysym=='Up':
    #         ind=ind-1
    #     if ind<0: ind=0
    #     if ind>len(dlist)-1: ind=len(dlist)-1
    #     self.mathA.setvalue(dlist[ind])
    #     self.mathA.see(ind)
    #     self.checkmathA()

    # def arrowmathB(self,event):
    #     dlist=self.mathB.get()
    #     ind=dlist.index(self.mathB.getvalue()[0])
    #     if event.keysym=='Down':
    #         ind=ind+1
    #     if event.keysym=='Up':
    #         ind=ind-1
    #     if ind<0: ind=0
    #     if ind>len(dlist)-1: ind=len(dlist)-1
    #     self.mathB.setvalue(dlist[ind])
    #     self.mathB.see(ind)
    #     self.checkmathscalar()

    # def arrowmathop(self,event):
    #     dlist=self.mathop.get()
    #     ind=dlist.index(self.mathop.getvalue()[0])
    #     if event.keysym=='Down':
    #         ind=ind+1
    #     if event.keysym=='Up':
    #         ind=ind-1
    #     if ind<0: ind=0
    #     if ind>len(dlist)-1: ind=len(dlist)-1
    #     self.mathop.setvalue(dlist[ind])
    #     self.mathop.see(ind)
    #     self.checkmathop()

    def saveCurrentFileBufferInfo(self):
        self.dataFileBuffer[self.activeFileBuffer]['data']=self.mapdata
        self.dataFileBuffer[self.activeFileBuffer]['changes']=self.changes
        self.dataFileBuffer[self.activeFileBuffer]['hasdata']=self.hasdata
        self.dataFileBuffer[self.activeFileBuffer]['wfn']=self.workingdir.wfn
        self.dataFileBuffer[self.activeFileBuffer]['name']=self.activeFileBuffer
        self.dataFileBuffer[self.activeFileBuffer]['mcafn']=self.MCAfilename
        self.dataFileBuffer[self.activeFileBuffer]['zoom']=self.maindisp.zmxyi
        self.dataFileBuffer[self.activeFileBuffer]['pm']=[self.plotmarkerlist,self.outputPMlistParam()]
        self.dataFileBuffer[self.activeFileBuffer]['spectrumX']=self.spectrumXvalues
        self.dataFileBuffer[self.activeFileBuffer]['spectrumS']=self.spectrumSwitch
        self.dataFileBuffer[self.activeFileBuffer]['mask']=self.mask
        self.dataFileBuffer[self.activeFileBuffer]['masktype']=self.maindisp.masktype
        self.dataFileBuffer[self.activeFileBuffer]['maskraw']=self.mapavgallpts
        self.dataFileBuffer[self.activeFileBuffer]['tcrange']=self.tcrangedict
        self.dataFileBuffer[self.activeFileBuffer]['MCAxrayslope']=self.currentMCAXraySlope
        self.dataFileBuffer[self.activeFileBuffer]['MCAxvalues']=self.currentMCAXvalues
        self.dataFileBuffer[self.activeFileBuffer]['pyMCAsilent']=self.pyMCAsilent
        self.dataFileBuffer[self.activeFileBuffer]['lastPCAresult']=[self.PCAlastevect,self.PCAlastprop,self.PCAlastchans]
        self.dataFileBuffer[self.activeFileBuffer]['dictFTIR']=self.FTIRgroup
        
    def loadFromCurrentFileBufferInfo(self,nodata=True):
        if not nodata:
            self.mapdata=self.dataFileBuffer[self.activeFileBuffer]['data']
            self.changes=self.dataFileBuffer[self.activeFileBuffer]['changes']
            self.hasdata=self.dataFileBuffer[self.activeFileBuffer]['hasdata']
            self.workingdir.wfn=self.dataFileBuffer[self.activeFileBuffer]['wfn']
            [self.plotmarkerlist,plottemplist]=self.dataFileBuffer[self.activeFileBuffer]['pm']
        self.MCAfilename=self.dataFileBuffer[self.activeFileBuffer]['mcafn']
        self.maindisp.zmxyi=self.dataFileBuffer[self.activeFileBuffer]['zoom']
        self.spectrumXvalues=self.dataFileBuffer[self.activeFileBuffer]['spectrumX']
        self.spectrumSwitch=self.dataFileBuffer[self.activeFileBuffer]['spectrumS']
        self.mask=self.dataFileBuffer[self.activeFileBuffer]['mask']
        self.maindisp.masktype=self.dataFileBuffer[self.activeFileBuffer]['masktype']
        self.mapavgallpts=self.dataFileBuffer[self.activeFileBuffer]['maskraw']
        self.tcrangedict=self.dataFileBuffer[self.activeFileBuffer]['tcrange']
        self.currentMCAXraySlope=self.dataFileBuffer[self.activeFileBuffer]['MCAxrayslope']
        self.currentMCAXvalues=self.dataFileBuffer[self.activeFileBuffer]['MCAxvalues']
        self.pyMCAsilent=self.dataFileBuffer[self.activeFileBuffer]['pyMCAsilent']
        [self.PCAlastevect,self.PCAlastprop,self.PCAlastchans]=self.dataFileBuffer[self.activeFileBuffer]['lastPCAresult']        
        self.FTIRgroup=self.dataFileBuffer[self.activeFileBuffer]['dictFTIR']

    def clickOnFileTab(self,page):
        #save current changing things
        self.saveCurrentFileBufferInfo()
        self.clearallmarker()
        #get new
        self.mapdata=self.dataFileBuffer[page]['data']
        self.changes=self.dataFileBuffer[page]['changes']
        self.hasdata=self.dataFileBuffer[page]['hasdata']
        self.workingdir.wfn=self.dataFileBuffer[page]['wfn']
        self.activeFileBuffer=self.dataFileBuffer[page]['name']
        self.MCAfilename=self.dataFileBuffer[page]['mcafn']
        self.maindisp.zmxyi=self.dataFileBuffer[page]['zoom']
        [self.plotmarkerlist,plottemplist]=self.dataFileBuffer[page]['pm']
        self.spectrumXvalues=self.dataFileBuffer[page]['spectrumX']
        self.spectrumSwitch=self.dataFileBuffer[page]['spectrumS']
        self.mask=self.dataFileBuffer[page]['mask']
        self.maindisp.masktype=self.dataFileBuffer[page]['masktype']
        self.mapavgallpts=self.dataFileBuffer[page]['maskraw']
        self.tcrangedict=self.dataFileBuffer[page]['tcrange']
        self.currentMCAXraySlope=self.dataFileBuffer[page]['MCAxrayslope']
        self.currentMCAXvalues=self.dataFileBuffer[page]['MCAxvalues']
        self.pyMCAsilent=self.dataFileBuffer[page]['pyMCAsilent']
        [self.PCAlastevect,self.PCAlastprop,self.PCAlastchans]=self.dataFileBuffer[page]['lastPCAresult']
        self.FTIRgroup=self.dataFileBuffer[page]['dictFTIR']
        
        print(self.activeFileBuffer,self.mapdata.data.shape)
        if self.showpage:
            self.updatelists()
            self.updatefromclick(plottemplist)
        self.showpage=True

    def closeFileTab(self,ignore=False):
        if not ignore:
            if self.changes==1 or self.dataFileBuffer[self.activeFileBuffer]['changes']==1:
                if not tkinter.messagebox.askyesno("Changes Made", "Unsaved changes made to data! Discard?"):
                    return -1
        if self.mapdata.hasHDF5:
            self.mapdata.hdf5.close() 
        #clear data buffers
        wfn=self.dataFileBuffer[self.activeFileBuffer]['wfn']
        self.workingFileBufferNames.remove(wfn)
        print(self.activeFileBuffer,'is closed')
        rem=[self.activeFileBuffer]
        self.filenb.delete(self.activeFileBuffer)
        del self.dataFileBuffer[rem[0]]
        if len(list(self.dataFileBuffer.keys()))==0:
            self.activeFileBuffer=None
        print(self.activeFileBuffer,'is active')
        self.updatelists()
        
    def checkHadData(self):
        if self.activeFileBuffer!=None: 
            return True
        return False
    
    def fileme(self):
#        if self.changes:
#            if not tkMessageBox.askyesno("Changes Made", "Unsaved changes made to data! Discard?"):
#                return
        globalfuncs.fileget(root,self.fileentry,dir=self.filedir.get())
        self.load_data_file(ignore=1)
        self.fullfieldzoom=None

    def multifileme(self):
        global LASTDIR
        globalfuncs.setstatus(self.status,"Ready")
        #get file name
        fty=[("data files","*.dat"),("HDF5 data files","*.hdf5"),("NXS data files","*.nxs"),("H5 data files","*.h5"),("SUPER files","*.*G"),("all files","*")]
        if LASTDIR==1:
            fty=[fty[1],fty[3],fty[2],fty[0],fty[4],fty[5]]
        if LASTDIR==2:
            fty=[fty[2],fty[1],fty[0],fty[3],fty[4],fty[5]]
        if LASTDIR==3:
            fty=[fty[5],fty[0],fty[1],fty[2],fty[3],fty[4]]
        infile=globalfuncs.ask_for_file(fty,self.filedir.get(),multi=True)
        multfn=self.root.tk.splitlist(infile)
        if multfn==():            
            return
        multfn=list(multfn)
        multfn.sort()
        for impfn in multfn:

            self.impfnmain=os.path.basename(impfn)
            fp=impfn.split('.')
            exten=fp[-1]
            if exten.lower()=='dat': LASTDIR=0
            elif exten[0].upper()=='H': LASTDIR=1
            elif exten[0].upper()=='N': LASTDIR=2
            else: LASTDIR=3
            #load it
            globalfuncs.setstatus(self.status,"LOADING...")
            globalfuncs.entry_replace(self.fileentry,impfn)
            self.load_data_file(ignore=1)
        self.fullfieldzoom=None

    def filemeSpecial(self):
#        if self.changes:
#            if not tkMessageBox.askyesno("Changes Made", "Unsaved changes made to data! Discard?"):
#                return
        special=[] 
        if self.fullfieldzoom is None:        
            #get current zoom
            special.append(self.maindisp.zmxyi)   
            self.fullfieldzoom=list(self.maindisp.zmxyi)
            temp=False
        else:
            special.append(self.fullfieldzoom)
            temp=True
        #ask for frame rate
        skip=tkinter.simpledialog.askinteger(title='Data Frame Skip ',prompt='Enter skip rate to extract data',initialvalue=20)
        if skip<=0 or skip is None:
            skip=1
        special.append(skip)       
       
        fn=globalfuncs.fileget(self.root,self.fileentry,dir=self.filedir.get(),replace=False)
        special.append(fn)
        special.append(temp)
        self.load_data_file(ignore=1,special=special)

    def load_data_file(self,ignore=0,special=None):
#        if not ignore:
#            if self.changes:
#                if not tkMessageBox.askyesno("Changes Made", "Unsaved changes made to data! Discard?"):
#                    return
        if special is None:
            newdataName=self.fileentry.get()
            isTemp=False
        else:
            newdataName=special[2]
            isTemp=special[3]
        #save current info
        if self.activeFileBuffer is not None:
            self.saveCurrentFileBufferInfo()
        dataname=globalfuncs.trimdirext(newdataName)
        if dataname=='':
            print("   ERROR: Empty file name")
            return
        if not os.path.exists(newdataName):
            print("    File not found")
            globalfuncs.setstatus(self.status,"LOADING...failed.   File not found")
            return
        #read file
#        if self.hasdata and not isTemp:
#            if self.mapdata.hasHDF5:
#                self.mapdata.hdf5.close()                
        self.filedir.set(os.path.dirname(newdataName))
        shortfn=os.path.splitext(os.path.basename(newdataName))[0]
        shortfn=shortfn.replace('_','-')
        if shortfn in self.dataFileBuffer:
            i=1
            while True:
                if shortfn+'-'+str(i) in self.dataFileBuffer:
                    i+=1
                else:
                    shortfn=shortfn+'-'+str(i)
                    break
        globalfuncs.setstatus(self.status,"LOADING...")
        wfn=1
        while True:
            if wfn in self.workingFileBufferNames:
                wfn+=1
            else:
                break
        self.workingdir.wfn=wfn
        self.workingFileBufferNames.append(wfn)        
    
        newmap=ImageGet.ImageGet((newdataName),self.root,self.LargeFileEdit.get(),temp=isTemp,workdir=self.workingdir,special=special)
        if newmap is None:
            globalfuncs.setstatus(self.status,"LOADING...failed")
            return
        newmap.cleanString()
        self.activeFileBuffer=shortfn    

        if special is None or special[3] is False:
                        
            self.mapdata=newmap            
            self.cleanup_load_data()
        else:
            #add channels
            for c in newmap.labels:
                ind=1
                ok=False
                basename=c
                newname=globalfuncs.fixlabelname(basename)
                while not ok:
                    
                    if newname in self.mapdata.labels:
                        newname=globalfuncs.fixlabelname(basename+'_'+str(ind))
                        ind+=1
                    else:
                        ok=True
                #add the channel
                impind=newmap.labels.index(c)+2
                preimpdata=newmap.data.get(impind) 
                #print preimpdata.shape                           
                self.addchannel(preimpdata,newname)        
            


    def cleanup_load_data(self):
        #make indices
        self.mapdata.mapindex=zeros(self.mapdata.data.shape[:2],dtype=np.int32)
        id=0
        for i in range(self.mapdata.data.shape[0]):
            for j in range(self.mapdata.data.shape[1]):
                self.mapdata.mapindex[i,j]=id
                id=id+1
        haddata=self.checkHadData()
        self.hasdata=1
        #place SCAS
        for l in range(len(self.mapdata.labels)):
            self.mapdata.labels[l]=globalfuncs.fixlabelname(self.mapdata.labels[l])
        self.datachan.setlist(self.mapdata.labels)
        try:
            self.datachan.setvalue(self.mapdata.labels[1])
        except:
            self.datachan.setvalue(self.mapdata.labels[0])
        #make header/comments
        self.header=Pmw.TextDialog(self.imgwin,title='File Comments',defaultbutton=0,scrolledtext_usehullsize=1,
                                   scrolledtext_hull_width=300,scrolledtext_hull_height=100)
        self.header.withdraw()
        self.header.clear()
        if self.mapdata.comments!='':
            self.header.insert('end',self.mapdata.comments)
            self.header.insert('end','Energy: '+str(self.mapdata.energy))
        else:
            self.header.insert('end','No comments available')        
        #clear mask...
        self.clearcpmask(replot=False)
        self.mapavgallpts=[]
        #clear MCA info
        if self.MCAviewexist: self.killMCAplot()
        self.MCAviewexist=0
        if self.IRwinexist: self.killIRwin()
        self.IRwinexist=0
        self.clearMCABuffers()#self.MCArawdata=[]
        self.MCAfilename=''        
        if self.PCAViewWindow.exist: 
            self.PCAViewWindow.kill()
        if self.plotmarkerexist:
            self.killplotmarkerwin()
        self.PCArawdata=None
        if self.PCAhdf5fout is not None:
            self.PCAhdf5fout.close()
            self.PCAhdf5fout=None
        self.PCAdataLoaded=0
        #plotmarkers
        self.plotmarkerlist=[]
        self.maindisp.markerlist={}
        self.tcmarkerlist={}
        self.tcrangedict={}
        #clear CT info
        self.CTdispdict={}        
        #clear XANES fit info
        #self.fitXANESactive=0
        #clear histograms
        if self.histogramWindow.exist:
            self.histogramWindow.kill()
        #and moments
        if self.momentWindow.exist:
            self.momentWindow.kill()
        if self.dataSummaryWindow.exist:
            self.dataSummaryWindow.kill()
            #self.dataSummaryWindow=None
        #make new mask
        self.mask=Mask.MaskClass()
        
        #mapit
        #self.maindisp.clearzoom(disp=0)
        self.maindisp.defineZoom()
        #i0 channel?
        self.seti0channel(wd=1)
        self.setTIMEchannel(wd=1)

        #spectrum maker
        self.spectrumSwitch=0
        self.spectrumXvalues=[]
        if self.particleStatisticsWindow.exist:
            self.particleStatisticsWindow.kill()
        ##self.domapimage()
        if haddata:
            self.updatelists()
        self.xyflip.set(0)
#        self.contoursOn.set(0)
        self.changes=0
        self.currentMCAXraySlope = 10.0
        self.currentMCAXvalues = None
        self.pyMCAsilent={}
        self.PCAlastprop=None
        self.PCAlastevect=None
        self.PCAlastchans=[]
        self.FTIRgroup=None

        print('LOAD',self.activeFileBuffer,self.mapdata.data.shape)
        dfinfo={}
        dfinfo['data']=self.mapdata
        dfinfo['changes']=self.changes
        dfinfo['hasdata']=self.hasdata
        dfinfo['wfn']=self.workingdir.wfn
        dfinfo['name']=self.activeFileBuffer
        dfinfo['fname']=self.fileentry.get()
        dfinfo['mcafn']=self.MCAfilename
        dfinfo['zoom']=self.maindisp.zmxyi
        dfinfo['pm']=[self.plotmarkerlist,[]]
        dfinfo['spectrumX']=self.spectrumXvalues
        dfinfo['spectrumS']=self.spectrumSwitch
        dfinfo['mask']=self.mask
        dfinfo['masktype']=self.maindisp.masktype
        dfinfo['maskraw']=self.mapavgallpts
        dfinfo['tcrange']=self.tcrangedict
        dfinfo['MCAxrayslope']=self.currentMCAXraySlope
        dfinfo['MCAxvalues']=self.currentMCAXvalues
        dfinfo['pyMCAsilent']=self.pyMCAsilent
        dfinfo['lastPCAresult']=[self.PCAlastevect,self.PCAlastprop]
        dfinfo['dictFTIR']=self.FTIRgroup
        
        self.dataFileBuffer[self.activeFileBuffer]=dfinfo
        self.filenb.add(self.activeFileBuffer,page_background='#d4d0c8')
        self.filenb.selectpage(self.activeFileBuffer)
        #print self.dataFileBuffer.keys()

    def createSSRLMosaic(self):
        globalfuncs.setstatus(self.status,"Ready")
        #get file name
        fty=[("HDF5 data files","*.hdf5"),("all files","*")]
        infile=globalfuncs.ask_for_file(fty,self.filedir.get(),multi=True)
        multfn=self.root.tk.splitlist(infile)
        if multfn==():
            print('no files selected')
            globalfuncs.setstatus(self.status,"No files selected.")
            return
        multfn=list(multfn)
        multfn.sort()
        newfns=[]
        for impfn in multfn:
            #load it
            globalfuncs.setstatus(self.status,"Forming mosaic...")
            mosaicfn=stitch_hdf_mod.stitch_hdf(impfn)
            globalfuncs.setstatus(self.status,"Mosaic complete.")
            newfns.append(mosaicfn)
    
        #load new file?
        if tkinter.messagebox.askyesno(title="Mosaic Builder",message="Load new moasic file(s)?"):
            for mfn in newfns:
                self.fileentry.setvalue(mfn)
                self.load_data_file(ignore=1)

        globalfuncs.setstatus(self.status,"Ready")

    def createSUPERhead(self):
        self.SUPbuttondict={}
        #create dialog:
        self.SUPERheaddialog=Pmw.Dialog(self.imgwin,title='Create SUPER file header',buttons=('Accept','Cancel'),
                                        command=self.doSUPERdialog)
        h=self.SUPERheaddialog.interior()

        fb1=tkinter.Frame(h,background='#d4d0c8')
        fb1.pack(side=tkinter.TOP,padx=2,pady=5)
        fb2=tkinter.Frame(h,background='#d4d0c8')
        fb2.pack(side=tkinter.TOP,padx=2,pady=5)
        #starting file
        self.SUPstart=Pmw.EntryField(fb1, label_text="Starting File:",labelpos=tkinter.W,validate=None,entry_width=50,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.SUPstart.pack(side=tkinter.LEFT,padx=5,pady=2,fill=tkinter.X)
        b=Button(fb1,style='OPEN.TButton',width=7)
        b.pack(side=tkinter.LEFT,padx=2,pady=2)
        b.bind('<Button-1>',self.SUPfileme)
        self.SUPbuttondict[b]=self.SUPstart
        #ending file
        self.SUPend=Pmw.EntryField(fb2, label_text="Ending File:",labelpos=tkinter.W,validate=None,entry_width=50,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.SUPend.pack(side=tkinter.LEFT,padx=5,pady=2,fill=tkinter.X)
        b=Button(fb2,style='OPEN.TButton',width=7)
        b.pack(side=tkinter.LEFT,padx=2,pady=2)
        b.bind('<Button-1>',self.SUPfileme)
        self.SUPbuttondict[b]=self.SUPend
        Pmw.alignlabels([self.SUPstart,self.SUPend])
        
        self.SUPERheaddialog.show()

    def SUPfileme(self,event):
        #event.widget.config(state=ACTIVE)
        globalfuncs.fileget(self.SUPERheaddialog,self.SUPbuttondict[event.widget],dir=self.filedir.get())
        #event.widget.config(state=tkinter.NORMAL)

    def doSUPERdialog(self,result):
        #check validity
        if result=='Cancel':
            self.SUPERheaddialog.destroy()
            return
        if self.SUPstart=='' or self.SUPend=='':
            return
        #for number too
        dirroot=os.path.dirname(self.SUPstart.get())
        self.filedir.set(dirroot)
        flist=os.listdir(dirroot)
        try:
            snum=int(os.path.splitext(self.SUPstart.get())[1][1:])
            srt=os.path.splitext(os.path.split(self.SUPstart.get())[1])[0]
            enum=int(os.path.splitext(self.SUPend.get())[1][1:])
        except:
            print('non valid super names')
            return
        if snum>enum:
            print('non valid sequence')
            return
        for cnum in range(snum,enum+1):
            if srt+'.'+cnum.zfill(3) not in flist:
                print('non continuous sequence')
                return
        print('all valid')
        #start to make header:
        outfile=''
        #open first file
        fidin=open(self.SUPstart.get(),'r')
        inlines=fidin.read().split('\n')
        fidin.close()
        getz=0
        for l in inlines:
            if len(l)<1: continue
            if l.split()[0]=='#F':
                outfn=l.split()[1]+'G'
                outfile=outfile+'#F '+outfn+'\n'
            if l.split()[0] in ['#D','#C','#H','#T','#N','#S']:
                outfile=outfile+l+'\n'
            if len(l.split()[0])==3 and l.split()[0][0:2]=='#P':
                outfile=outfile+l+'\n'
            if 'ZAXIS' in l.split():
                getz=1
                getzind=l.split().index('ZAXIS')
                continue
            if getz:
                getz=0
                zval=l.split()[getzind]
        outfile=outfile+'#L ZAXIS\n'
        outfile=outfile+zval+'\n'
        for cnum in range(snum+1,enum+1):
            fidin=open(dirroot+'/'+srt+'.'+cnum.zfill(3))
            inlines=fidin.read().split('\n')
            fidin.close()
            getz=0
            for l in inlines:
                if len(l)<1: continue
                if 'ZAXIS' in l.split():
                    getz=1
                    getzind=l.split().index('ZAXIS')
                    continue
                if getz:
                    getz=0
                    zval=l.split()[getzind]
            outfile=outfile+zval+'\n'            

        #write header result
        fn=dirroot+'/'+outfn
        fid=open(fn,'w')
        fid.write(outfile)
        fid.close()
       
        self.SUPERheaddialog.destroy()
        #load file
        self.fileentry.setvalue(fn)
        self.load_data_file()

    def viewheader(self):
        globalfuncs.setstatus(self.status,"Ready")
        if self.hasdata:
            self.header.show()
        else:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')


    def exportchannelWithResize(self):
        self.exportchannelToTab(resize=True)

    def exportchannelToTab(self,resize=False):
        self.exportChannelDoResize=resize
        #export multiple channels
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if len(list(self.dataFileBuffer.keys()))<2:
            print('Only one file open, no other tab to import to')
            globalfuncs.setstatus(self.status,'Only one file open, no other tab to import to')
            return            
        self.multiexportdialog=Pmw.SelectionDialog(self.imgwin,title="Export Multiple Channels",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.selectexportdestination)
        self.multiexportdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        
    def selectexportdestination(self,result):
        self.exGoner=self.multiexportdialog.getcurselection()
        self.multiexportdialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        self.exportdestdialog=Pmw.SelectionDialog(self.imgwin,title="Export Multiple Channels",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Destination Tab',scrolledlist_items=list(self.dataFileBuffer.keys()),
                                                   command=self.exportmultichannel)
         
    def exportmultichannel(self,result):
        goner=self.exGoner
        dest=self.exportdestdialog.getcurselection()
        self.exportdestdialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        if len(dest)<1:
            globalfuncs.setstatus(self.status,'No action taken')
            return
        dest=dest[0]            
        print (goner,dest)
        #check matching data sizes
        if not self.exportChannelDoResize and self.mapdata.data.shape[:2]!=self.dataFileBuffer[dest]['data'].data.shape[:2]:
            print('WARNING: Imported data has different size')
            globalfuncs.setstatus(self.status,'WARNING: Imported data has different size')        
            return
        
        for going in goner:           
            ind=1
            ok=False
            if int(self.mapdata.energy)>1:
                basename=going+str(self.mapdata.energy)
            else:
                basename=going
            newname=globalfuncs.fixlabelname(basename)
            while not ok:                
                if newname in self.dataFileBuffer[dest]['data'].labels:
                    newname=globalfuncs.fixlabelname(basename+'_'+str(ind))
                    ind+=1
                else:
                    ok=True
            print(self.mapdata.energy,going,basename,newname)
            #add the channel
            impind=self.mapdata.labels.index(going)+2
            preimpdata=self.mapdata.data.get(impind)
            #check for resize
            if self.exportChannelDoResize:
                destsize=self.dataFileBuffer[dest]['data'].data.get(0).shape
                impsize=preimpdata.shape
                if destsize!=impsize:
                    wr=float(destsize[0])/float(impsize[0])
                    hr=float(destsize[1])/float(impsize[1])
                    print (destsize,impsize,wr,hr)
                    if wr<hr:
                        sd=imutils.resize(preimpdata,height=destsize[0])
                        nd=np.zeros(destsize,dtype=np.float32)
                        nd[:,0:sd.shape[1]]=sd
                    elif wr>hr:
                        sd=imutils.resize(preimpdata,width=destsize[1])
                        nd=np.zeros(destsize,dtype=np.float32)
                        nd[0:sd.shape[0],:]=sd
                    else: #same aspect ratio
                        nd=imutils.resize(preimpdata,width=destsize[0])
                    preimpdata=nd                   
            self.addchannel(preimpdata,newname,fbuffer=dest)

    def exportchannelWithRegistration(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if len(list(self.dataFileBuffer.keys()))<2:
            print('Only one file open, not other tab to import to')
            globalfuncs.setstatus(self.status,'Only one file open, not other tab to import to')
            return  
        if self.regWindow.exist:
            self.regWindow.win.show()   
        else:
            ps=ExportRegistrationClass.ExportRegWindowParams(Display.DisplayParams(self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,self.dispScaleFactor,self.status,self.maindisp.zmxyi), self.dataFileBuffer, self.activeFileBuffer, None, self.addchannel)
            self.regWindow.create(self.mapdata, ps)
   
        
    def dothreshold(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return


        self.threshDialog=Pmw.Dialog(self.imgwin,title="Thresholding",buttons=('Done','Preview','Save'),defaultbutton='Done',
                                     command=self.enterThreshFilter)
        h=self.threshDialog.interior()
        lf=tkinter.Frame(h,background='#d4d0c8')
        lf.pack(side=tkinter.LEFT,fill='both')
        #data selection
        self.threshsel=Pmw.ScrolledListBox(lf,labelpos='n',label_text='Select Channel',items=self.mapdata.labels,listbox_selectmode=tkinter.SINGLE,
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=self.selectthreshfilterdata,listbox_height=15,
                                           hull_background='#d4d0c8',label_background='#d4d0c8')
        self.threshsel.pack(side=tkinter.TOP,padx=4,pady=5,fill='both')
        #new data name
        self.newthreshfiltername=Pmw.EntryField(lf,labelpos='w',label_text='New Channel Name: ',entry_width=15,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.newthreshfiltername.pack(side=tkinter.TOP,fill=tkinter.X,pady=10)

        #data character
        g1=Pmw.Group(lf,tag_text='Data Properties',hull_background='#d4d0c8',tag_background='#d4d0c8',ring_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        self.threshDataProps=tkinter.Label(g1.interior(),text="\n\n\n",background='#d4d0c8')
        self.threshDataProps.pack(side=tkinter.TOP,padx=5,pady=4,fill='both')
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')        

        rf=tkinter.Frame(h,background='#d4d0c8')
        rf.pack(side=tkinter.LEFT,fill='both',padx=5)
        l=tkinter.Label(rf,text='Threshold Parameters',bd=2,relief=tkinter.RAISED,background='#d4d0c8')
        l.pack(side=tkinter.TOP,fill=tkinter.X,expand=1,pady=5)

        #filter type
        g1=Pmw.Group(rf,tag_text='Type',hull_background='#d4d0c8',tag_background='#d4d0c8',ring_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        self.threshfiltertype=Pmw.RadioSelect(g1.interior(),buttontype='radiobutton',orient='vertical',command=self.selectthreshfilterdata,hull_background='#d4d0c8')
        for text in ('TruncMax','TruncMin','Binary','InvBinary','Otsu','ThreshZero','InvThreshZero','Adapt','InvAdapt','Canny'):
            self.threshfiltertype.add(text,background='#d4d0c8')
        self.threshfiltertype.setvalue('TruncMax')
        self.threshfiltertype.pack(side=tkinter.TOP,padx=3,pady=3)
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')
        
        #level
        self.threshLevelvar=tkinter.DoubleVar()
        self.threshLevelvar.set(50)
        self.threshLevel=tkinter.Scale(rf,label='Threshold Level (%)',background='#d4d0c8',variable=self.threshLevelvar,width=20,length=150,from_=0,to=100,orient=tkinter.HORIZONTAL,resolution=0.01)
        self.threshLevel.pack(side=tkinter.TOP,fill='both',expand=1,padx=5,pady=10)

        #level
        self.threshLevelvar2=tkinter.DoubleVar()
        self.threshLevelvar2.set(50)
        self.threshLevel2=tkinter.Scale(rf,label='Threshold Level2 (%)',background='#d4d0c8',variable=self.threshLevelvar2,width=20,length=150,from_=0,to=100,orient=tkinter.HORIZONTAL,resolution=0.01,state=tkinter.DISABLED,fg='gray70')
        self.threshLevel2.pack(side=tkinter.TOP,fill='both',expand=1,padx=5,pady=10)

        #value
        self.threshValue=Pmw.EntryField(rf,labelpos='w',label_text='Threshold Value (num): ',entry_width=15,validate='real',value=0,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.threshValue.pack(side=tkinter.TOP,fill=tkinter.X,pady=10)        

        self.threshDialog.show()
       
##        val=tkSimpleDialog.askfloat(title='Channel Threshold',prompt=text)
##        if val is None:
##            print 'No Change'
##            globalfuncs.setstatus(self.status,'No Change')
##            return
##        newv=tkSimpleDialog.askfloat(title='Channel Threshold',prompt='Replace threshold values with:')
##        if newv is None:
##            print 'No Change'
##            globalfuncs.setstatus(self.status,'No Change')
##            return
##        newdata=where(greater(self.mapdata.data[:,:,datind],val),newv,self.mapdata.data[:,:,datind])
##        self.mapdata.data[:,:,datind]=newdata
##        self.changes=1
##        self.domapimage()

    def cancelimport(self):
        print('Import cancelled')
        self.importneedcrop=0
        self.doneimp=1
        self.donefileimp=1
        self.abortimp=1
        globalfuncs.setstatus(self.status,'Import cancelled')

    def importchannel_rapid(self):
        self.importchannel(rapid=1)

    def importchannel(self,rapid=0):
        self.rapidimport=rapid
        self.impchanlist=[]
        self.abortimp=0
        self.doneimp=0
        self.donefileimp=0
        global LASTDIR
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #add channel from file
        #get file name
        fty=[("data files","*.dat"),("HDF5 data files","*.hdf5"),("NXS data files","*.nxs"),("H5 data files","*.h5"),("SUPER files","*.*G"),("all files","*")]
        if LASTDIR==1:
            fty=[fty[1],fty[3],fty[2],fty[0],fty[4],fty[5]]
        if LASTDIR==2:
            fty=[fty[2],fty[1],fty[0],fty[3],fty[4],fty[5]]
        if LASTDIR==3:
            fty=[fty[5],fty[0],fty[1],fty[2],fty[3],fty[4]]
        infile=globalfuncs.ask_for_file(fty,self.filedir.get(),multi=True)
        multfn=self.root.tk.splitlist(infile)
        if multfn==():
            self.cancelimport()
            return
        multfn=list(multfn)
        multfn.sort()
        for impfn in multfn:
            if self.abortimp: return
            self.donefileimp=0
            self.impfnmain=os.path.basename(impfn)
            fp=impfn.split('.')
            exten=fp[-1]
            if exten.lower()=='dat': LASTDIR=0
            elif exten[0].upper()=='H': LASTDIR=1
            elif exten[0].upper()=='N': LASTDIR=2
            else: LASTDIR=3
            #load it
            globalfuncs.setstatus(self.status,"LOADING...")
            self.impdata=ImageGet.ImageGet(impfn,self.LargeFileEdit.get(),workdir=self.workingdir,temp=True)
            self.importneedcrop=0
            self.impdata.cleanString()

            #check for matching shapes
            if self.impdata.data.shape[:2]!=self.mapdata.data.shape[:2]:
                print('WARNING: Imported data has different size')
                globalfuncs.setstatus(self.status,'WARNING: Imported data has different size')
                if self.rapidimport:
                    self.cancelimport()
                    return
                if tkinter.messagebox.askokcancel('Import Data','Import data has different size!\nCrop data on import?'):
                    self.importneedcrop=1
                else:
                    self.cancelimport()
                    return
            #select channel
            if self.impchanlist==[] or self.rapidimport==0:
                self.impchandialog=Pmw.SelectionDialog(self.imgwin,title='Import Selection:',buttons=('OK','Cancel'),
                                              defaultbutton='OK',scrolledlist_labelpos='n',
                                              label_text='Select channel to import:\n'+self.impfnmain,
                                              scrolledlist_items=self.impdata.labels,command=self.importchannel_b)
                self.impchandialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
                try:
                    self.impchandialog.setvalue(self.impchanlist)
                except:
                    print('previous list does not exist in next file... defaults')
                    self.impchanlist=[]
                    self.impchandialog.setvalue(self.impchanlist)
                while not self.donefileimp:
                    self.impchandialog.focus_set()
                    time.sleep(0.2)
                    self.root.update()
            else:
                self.importchannel_c_rapid(fn=impfn)

    def importchannel_b(self,result):
        self.impchandialog.withdraw()
        if result=='Cancel' or self.impchandialog.getcurselection()==():
            self.cancelimport()
            return

        #name for channel in file
        self.impchanlist=self.impchandialog.getcurselection()

        if self.rapidimport:
            self.donefileimp=1
            self.importchannel_c_rapid()
            return

        for c in self.impchanlist:
            self.impchan=c
            self.doneimp=0
            self.chandialog=Pmw.PromptDialog(self.imgwin,title='Import Selection:',buttons=('OK','Cancel'),
                                             defaultbutton='OK',label_text='Enter new channel name:\n'+self.impfnmain,
                                             entryfield_labelpos='n',command=self.importchannel_c)
            if self.impdata.energy>1:
                self.chandialog.setvalue(str(c)+str(self.impdata.energy))
            else:
                self.chandialog.setvalue(str(c))
            self.chandialog.component('entryfield_entry').focus_set()
            while not self.doneimp:
                time.sleep(0.1)
                self.root.update()
            
        self.donefileimp=1            
        self.impdata.hdf5.close()

    def importchannel_c(self,result):
        self.chandialog.withdraw()
        if result=='Cancel' or self.chandialog.get()=='':
            self.cancelimport()
            return
        #check for uniqueness
        newname=globalfuncs.fixlabelname(self.chandialog.get())
        if newname in self.mapdata.labels:
            print('Enter unique channel name')
            globalfuncs.setstatus(self.status,'Enter unique channel name')
            self.importchannel_b('OK')
            return
        #add the channel
        impind=self.impdata.labels.index(self.impchan)+2
        #impind=self.impdata.labels.index(self.impchan[0])+2
        self.preimpdata=self.impdata.data.get(impind)
        #default for rough size
        self.impsizechange=array(self.mapdata.data.shape[:2])-array(self.preimpdata.shape)      
        
        if self.importneedcrop:
            self.importcropdata()
            print(self.impsizechange)
            if self.impsizechange.any():#if self.impsizechange!=array([0,0]):
                self.cancelimport()
                return                
        self.addchannel(self.preimpdata,newname)
        self.doneimp=1

    def importchannel_c_rapid(self,fn=None):
        #check for uniqueness
        for chi in self.impchanlist:
            self.impchan=chi #self.impchanlist[0] #only do first channel for brevity
            ind=1
            ok=False
            if int(self.impdata.energy)>1:
                basename=self.impchan+str(self.impdata.energy)
            else:
                basename=self.impchan
            newname=globalfuncs.fixlabelname(basename)
            while not ok:
                
                if newname in self.mapdata.labels:
                    newname=globalfuncs.fixlabelname(basename+'_'+str(ind))
                    ind+=1
                else:
                    ok=True
            print(self.impdata.energy,self.impchan,basename,newname)
            #add the channel
            impind=self.impdata.labels.index(self.impchan)+2
            #impind=self.impdata.labels.index(self.impchan[0])+2
            self.preimpdata=self.impdata.data.get(impind)
            #default for rough size
            self.impsizechange=array(self.mapdata.data.shape[:2])-array(self.preimpdata.shape)      
                        
            self.addchannel(self.preimpdata,newname)
        self.doneimp=1
        self.impdata.hdf5.close()

    def importcropdata(self):
                
        #check for coordinate registration?
        print(self.impsizechange)
        print(self.impdata.xvals[0],self.impdata.yvals[0])
        xnear=globalfuncs.find_nearest(self.mapdata.xvals,self.impdata.xvals[0])
        ynear=globalfuncs.find_nearest(self.mapdata.yvals,self.impdata.yvals[0])
        print(xnear,self.mapdata.xvals[xnear])
        print(ynear,self.mapdata.yvals[ynear])        
        
        self.cropprev=None
        self.importCropdialog=Pmw.Dialog(self.imgwin,title="Import Data - Crop",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                        command=self.importCdone)
        h=self.importCropdialog.interior()
        if self.impsizechange[0]-ynear>0:
            sv=self.impsizechange[0]-ynear
        else:
            sv=0
        self.icvert=cropslide(h,'Vertical',self.impsizechange[0],startval=sv)
        self.ichorz=cropslide(h,'Horizontal',self.impsizechange[1],startval=xnear)
        b=PmwTtkButtonBox.PmwTtkButtonBox(h)
        b.add('Preview',command=self.importCpreview,style='GREEN.TButton',width=10)
        b.pack(side=tkinter.TOP,padx=5,pady=5)
        self.importCropdialog.show()        

    def importCpreview(self):
        preview=self.doImpCrop()
        if self.cropprev is None:
            self.cropprev=Display.Display(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,main=0,sf=self.dispScaleFactor)
        self.cropprev.placeData(transpose(preview[::-1,:]),transpose(self.mapdata.mapindex[::-1,:]),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,domask=0,mask=[],datlab='Import Preview')
        self.cropprev.main.lift()
        #self.importCropdialog.lift()
        #self.imgwin.iconify()

    def doImpCrop(self):
        preview=self.preimpdata
        if self.impsizechange[0]>0: #add
            zt=zeros((int(self.icvert.topleft.getvalue()),preview.shape[1]),dtype=np.float32)
            zb=zeros((int(self.icvert.botright.getvalue()),preview.shape[1]),dtype=np.float32)
            preview=np.concatenate((zb,preview,zt),axis=0)
        if self.impsizechange[0]<0: #crop
            preview=preview[int(self.icvert.botright.getvalue()):preview.shape[0]-int(self.icvert.topleft.getvalue()),:]
        if self.impsizechange[1]>0: #add
            zt=zeros((preview.shape[0],int(self.ichorz.topleft.getvalue())),dtype=np.float32)
            zb=zeros((preview.shape[0],int(self.ichorz.botright.getvalue())),dtype=np.float32)
            preview=np.concatenate((zt,preview,zb),axis=1)
        if self.impsizechange[1]<0: #crop
            preview=preview[:,int(self.ichorz.topleft.getvalue()):preview.shape[1]-int(self.ichorz.botright.getvalue())]
        return preview
    
    def importCdone(self,result):
        if result=='OK':
            self.preimpdata=self.doImpCrop()      
        self.impsizechange=array(self.mapdata.data.shape[:2])-array(self.preimpdata.shape)
        self.importCropdialog.withdraw()
        if self.cropprev is not None: self.cropprev.main.destroy()
        self.cropprev=None
        self.imgwin.deiconify()
        
    def addchannel(self,data,name,fbuffer=None):
        #assumes validity of channel and data!
        if fbuffer is None:
            buf=self.dataFileBuffer[self.activeFileBuffer]
            self.changes=1
        else:
            buf=self.dataFileBuffer[fbuffer]
        buf['data'].labels.append(name)
        ##newdat=zeros((self.mapdata.data.shape[0],self.mapdata.data.shape[1],self.mapdata.data.shape[2]+1),Float)
        ##newdat[:,:,:-1]=self.mapdata.data.get
        ##newdat[:,:,-1]=data
        ##self.mapdata.data=newdat
        buf['data'].data.addChannel(data)
        buf['data'].channels=buf['data'].channels+1
        self.updatelists()
        buf['changes']=1 #self.changes=1

    def updatefromclick(self,plottemplist):
        #display and corplot?
        self.maindisp.domask=0
        self.domapimage()
        self.addmarkersback(plottemplist)
        #self.updatePMall()
        if self.correlationPlot.exist: self.correlationPlot.checkcorplot()
        if self.histogramWindow.exist: self.makeHupdate()
        if self.momentWindow.exist: self.momentWindow.domoments()

    def updatelistsClear(self):
        self.datachan.setlist([])
        self.datachan.select_clear(0)
        # correlation
        if self.correlationPlot.exist:
            self.correlationPlot.cpxchan.setlist([])
            self.correlationPlot.cpychan.setlist([])
        if self.triColorWindow.exist:
            self.triColorWindow.killtcplot()
        # maths:
        if self.mathWindow.exist:
            self.mathWindow.kill()
        # xanes fit
        if self.xanesFitWindow.exist:
            self.xanesFitWindow.kill()
            #self.fitXANESactive=0
        if self.histogramWindow.exist:
            self.histogramWindow.kill()
        if self.momentWindow.exist:
            self.momentWindow.kill()
        globalfuncs.setstatus(self.status, "Ready")

    def updatelists(self):
        if self.activeFileBuffer is None:
            self.updatelistsClear()
            return
        #check all widgets and update labels if needed.
        #main:
        temp=self.datachan.getvalue()[0]
        self.datachan.setlist(self.mapdata.labels)
        try:
            self.datachan.setvalue(temp)
        except:
            self.datachan.select_clear(0)
            self.datachan.select_set(0)
        #correlation
        if self.correlationPlot.exist:
            try:
                temp=self.correlationPlot.cpxchan.getvalue()[0]
                self.correlationPlot.cpxchan.setlist(self.mapdata.labels)
                self.correlationPlot.cpxchan.setvalue(temp)
            except:
                self.correlationPlot.cpxchan.setlist(self.mapdata.labels)                
            try:
                temp=self.correlationPlot.cpychan.getvalue()[0]
                self.correlationPlot.cpychan.setlist(self.mapdata.labels)
                self.correlationPlot.cpychan.setvalue(temp)
            except:
                self.correlationPlot.cpychan.setlist(self.mapdata.labels)
##            #maybe update plot here?
        #tcplots:
        if self.triColorWindow.exist==1:
            self.triColorWindow.killtcplot()
            ps = TriColorWindowClass.TriColorWindowParams(self.savetcdisplayasjpg, self.viewtricolormap, self.savetricolormap, self.maindisp, self.CMYKOn, self.tcrangedict, self.status, self.dodt, self.deadtimevalue, self.DTICRchanval, self.root, self.xyflip, self.tcrefresh, self.tclegendexist, self.showscalebar, self.showscalebarText, self.tcmarkerupdate)
            self.triColorWindow.create(self.mapdata, ps)
        #maths:
        if self.mathWindow.exist:
            t=1
            try:
                temp=self.mathWindow.mathA.getvalue()[0]
            except:
                t=0
            self.mathWindow.mathA.setlist(self.mapdata.labels)
            if t:
                try:
                    self.mathWindow.mathA.setvalue(temp)
                except:
                    pass
            t=1
            try:
                temp=self.mathWindow.mathB.getvalue()[0]
            except:
                t=0
            mathBlist=self.mapdata.labels
            mathBlist.append('Scalar')        
            self.mathWindow.mathB.setlist(mathBlist)
            self.mapdata.labels.pop()
            if t:
                try:
                    self.mathWindow.mathB.setvalue(temp)
                except:
                    pass
        #xanes fit
        if self.xanesFitWindow.exist:
            #getlist
            temp=self.xanesFitWindow.fitXdata.getvalue()
            #replace list
            self.xanesFitWindow.fitXdata.setlist(self.mapdata.labels)
            #try to replace
            for n in temp:
                try:
                    self.xanesFitWindow.fitXdata.select_set(self.mapdata.labels.index(n))
                except:
                    pass
        if self.histogramWindow.exist:
            #getlist
            temp=self.histogramWindow.fitHdata.getvalue()
            #replace list
            self.histogramWindow.fitHdata.setlist(self.mapdata.labels)
            #try to replace
            for n in temp:
                try:
                    self.histogramWindow.fitHdata.select_set(self.mapdata.labels.index(n))
                except:
                    pass
        if self.momentWindow.exist:
            #getlist
            temp=self.momentWindow.fitMdata.getvalue()
            #replace list
            self.momentWindow.fitMdata.setlist(self.mapdata.labels)
            #try to replace
            for n in temp:
                try:
                    self.momentWindow.fitMdata.select_set(self.mapdata.labels.index(n))
                except:
                    pass
        globalfuncs.setstatus(self.status,"Ready")

    def askrenamechannel(self):
        #rename the currently selected channel
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return
        #make sure
        mt='Enter new name for channel: '+self.datachan.getvalue()[0]
        newval=tkinter.simpledialog.askstring(title='Rename Channel',prompt=mt)
        newval=globalfuncs.fixlabelname(newval)
        if newval=='' or newval is None or (newval in self.mapdata.labels):
            print('Action cancelled')
            globalfuncs.setstatus(self.status,'Action cancelled')
            return
        #rename it
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])
        self.mapdata.labels[datind]=newval
        if datind!=0:
            self.datachan.setvalue(self.datachan.get()[0])
        else:
            self.datachan.setvalue(self.datachan.get()[1])            
        self.updatelists()
        self.changes=1

       
    def askremovechannel(self):
        #remove the currently selected channel
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return
        #make sure
        mt='Are you sure you want\nto delete channel: '+self.datachan.getvalue()[0]
        self.doublecheck=Pmw.MessageDialog(self.imgwin,title='Delete Channel',buttons=('No','Yes'),
                                           defaultbutton='No',message_text=mt,
                                           iconpos='n',icon_bitmap='warning',command=self.removechannel)
        self.changes=1
                                        
    def removechannel(self,result):
        self.doublecheck.withdraw()
        if result=='No':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        self.killremovechannel(datind)
        self.domapimage()
        
    def killremovechannel(self,datind):        
        #remove from data 
        temp=[]
        ##newdat=zeros((self.mapdata.data.shape[0],self.mapdata.data.shape[1],self.mapdata.data.shape[2]-1),Float)
        #iterate over data
        #hitme=0
        self.mapdata.data.removeChannel(datind)
##        for i in range(self.mapdata.data.shape[2]):
##            if i!=datind and not hitme:
##                newdat[:,:,i]=self.mapdata.data[:,:,i]
##            if i!=datind and hitme:
##                newdat[:,:,i-1]=self.mapdata.data[:,:,i]
##            if i==datind:
##                hitme=1
        for i in range(self.mapdata.data.shape[2]-2+1):
            if i!=datind-2:
                temp.append(self.mapdata.labels[i])
##        self.mapdata.data=newdat
        self.mapdata.labels=temp
        self.datachan.setvalue(self.datachan.get()[0])
        self.mapdata.channels=self.mapdata.channels-1
        self.changes=1
        self.updatelists()

    def askremovemultichannel(self):
        #remove multiple channels
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.multiremovedialog=Pmw.SelectionDialog(self.imgwin,title="Remove Multiple Channels",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.removemultichannel)
        self.multiremovedialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        
    def removemultichannel(self,result):
        goner=self.multiremovedialog.getcurselection()
        self.multiremovedialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        for going in goner:
            datind=self.mapdata.labels.index(going)+2
            self.killremovechannel(datind)            
        self.domapimage()
        
    def dothreshold(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return

        if not self.thresholdingWindow.exist:
            ps= ThresholdingClass.ThresholdingWindowParams(self.maindisp, self.status, self.showmap, self.savedeconvcalculation)
            self.thresholdingWindow.create(self.mapdata, ps)
        else:
            self.thresholdingWindow.win.show()


       



    def placethescalebar(self):
        self.maindisp.placescalebardialog()

    def defaultscalebar(self):
        self.maindisp.defaultscalebar()
        
######################## Mapping

    def domapimagefromscaselect(self):
        self.domapimage()
        self.datachan.focus_set()

    def domapimagenewwindow(self,event):
        index=self.datachan.nearest(event.y)
        newdisp=Display.Display(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,main=0,callback=[None,None,self.displaySpectrumGraph,self.dispAddMarker],sf=self.dispScaleFactor)
        self.domapimage(nwflag=newdisp,datind=index+2)
        
    def domapimage(self,nwflag=None,datind=-1):
        if self.datachan.get()==():
            return
        globalfuncs.setstatus(self.status,"DISPLAYING...")
        if datind==-1:
            datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        datlab=self.mapdata.labels[datind-2]
        pic=self.mapdata.data.get(datind)[::-1,:]#[::-1,:,datind]
        mi=self.mapdata.mapindex[::-1,:]
        if len(self.mask.mask)!=0:
            picmsk=transpose(self.mask.mask[::-1,:])
        else:
            picmsk=[]
        nodt=0
        if self.datachan.getvalue()[0] in ['ICR','I0','I1','I2','I0STRM','I1STRM','I2STRM']: nodt=1
        if self.dodt.get()==1 and not nodt:
            #DT: corFF=FF*exp(tau*1e-6*ICR)
            icr=self.mapdata.data.get(self.DTICRchanval)[::-1,:]#[::-1,:,self.DTICRchanval]
            dtcor=np.exp(float(self.deadtimevalue.getvalue())*1e-6*icr)
            pic=pic*dtcor
        #i0 corr?
        if self.doI0c.get()==1:
            #geti0
            iind=self.mapdata.labels.index(self.i0chan.getvalue())+2
            i0dat=self.mapdata.data.get(iind)[::-1,:]#[::-1,:,iind]
            #divide
            (xlen,ylen)=self.mapdata.data.shape[:2]
            newdata=zeros((xlen,ylen),dtype=np.float32)
            for i in range(xlen):
                for j in range(ylen):
                    if i0dat[i,j]!=0:
                        newdata[i,j]=float(pic[i,j])/float(i0dat[i,j])
            pic=newdata
        if nwflag is None:
            self.maindisp.placeData(transpose(pic),transpose(mi),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,domask=self.usemaskinimage,mask=picmsk,datlab=datlab)
            self.showmap()
        else:
            nwflag.placeData(transpose(pic),transpose(mi),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,domask=self.usemaskinimage,mask=picmsk,datlab=datlab)

    def doCMYK(self):
        self.maindisp.CMYKOn=self.CMYKOn.get()
        self.domapimage()

    def showmap(self):
        #show map
        self.maindisp.main.show()
        if self.contoursOn.get()==1: self.datacontoursToggle()
        globalfuncs.setstatus(self.status,"Ready")

    def usemask(self):
        self.usemaskinimage=1
        self.domapimage()
        #if self.dataSummaryWindow.exist:  self.dataSummaryWindow.doDataSummary(self.contoursOn)

    def ignoremask(self):
        self.usemaskinimage=0
        self.domapimage()
        #if self.dataSummaryWindow.exist:  self.dataSummaryWindow.doDataSummary(self.contoursOn)

    def changeMIN(self):
        global MINSIZE
        mt='Enter new value for minimum size: '
        MINSIZE=tkinter.simpledialog.askinteger(title='Resize Display',prompt=mt,initialvalue=MINSIZE)
        self.domapimage()

    def doscaleup(self):
        self.domapimage()
        if self.triColorWindow.exist:
            self.triColorWindow.dotcdisplay()

    def startMultiFader(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.multiFader is not None and self.multiFader.mfimageexists:
            #kill it
            self.multiFader.killmfimwin()
            self.multiFader=None
        ps=Display.DisplayParams(self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,self.dispScaleFactor,self.status,self.maindisp.zmxyi)
        self.multiFader=FaderClass.MultiFader(self.imgwin,self.mapdata,self.dataFileBuffer,self.activeFileBuffer,ps, self.filedir)

    def startCrossFader(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.crossFader is not None and self.crossFader.cfimageexists:
            #kill it
            self.crossFader.killcfimwin()
            self.crossFader=None  
        ps=Display.DisplayParams(self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,self.dispScaleFactor,self.status,self.maindisp.zmxyi)
        self.crossFader=FaderClass.CrossFader(self.imgwin,self.mapdata,self.dataFileBuffer,self.activeFileBuffer,ps,self.filedir)

            
########################  Correlation Plot routines
    #WORK HERE
    
    def startmegaCrossPlot(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.megaX is not None:
            self.megaX=None
        #some parameters will get set up here
        ps=self.maindisp.zmxyi
        self.megaX=CorrelationPlotClass.MegaXPlot(self.imgwin,self.mapdata,ps)
        
    def startcorplot(self):
        #show corplot window if needed
        globalfuncs.setstatus(self.status,"Ready")
        if self.correlationPlot.exist:
            self.correlationPlot.win.show()
        elif self.hasdata:
            #self.corplotexist=1
            ps=CorrelationPlotClass.CorrParams(self.exportcorplot, self.useMaskforCorPlotData, self.corplotSQRT, self.maindisp.zmxyi, self.dataFileBuffer,self.activeFileBuffer, self.dodt, self.mask, self.datachanCallBack, self.colormapCallBack, self.deadtimevalue, self.deadtimecorrection, self.DTICRchanval)
            
            #if self.maindisp.zmxyi 
            self.correlationPlot.create(self.mapdata,ps)
        else:
            print('No Data')            

            globalfuncs.setstatus(self.status,'No Data')

    def datachanCallBack(self,ind):
        return self.datachan.getvalue()[ind]
    
    def colormapCallBack(self,asText=True):
        return self.maindisp.getcolormap(asText=asText)
    
    #COME BACK HERE AND RENAME IT JOY
    def clearcpmask(self,replot=True):
        #clear mask
        self.mask.clear()
        
        self.maindisp.masktype=None
        #remove from graph if it exists
        if not self.correlationPlot.exist:
            return
        if not replot:
            return
        self.correlationPlot.clearcpmask()
        
    def makeMaskFromZoom(self):
         globalfuncs.setstatus(self.status,"Ready")
         if not self.hasdata:
             print('No Data')
             globalfuncs.setstatus(self.status,'No Data')
             return
         if self.datachan.get()==():
             return   

         if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
             self.mask.mask=np.zeros((self.mapdata.data.shape[0],self.mapdata.data.shape[1]),dtype=np.float32)
             self.mask.mask[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=np.ones(self.mapdata.data.get(0)[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]].shape)

         else:
             self.mask.mask=np.ones((self.mapdata.data.shape[0],self.mapdata.data.shape[1]),dtype=np.float32)
         globalfuncs.setstatus(self.status,"Mask complete") 

    def useImageasMask(self):
         globalfuncs.setstatus(self.status,"Ready")
         if not self.hasdata:
             print('No Data')
             globalfuncs.setstatus(self.status,'No Data')
             return
         if self.datachan.get()==():
             return        
         datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
         data=self.mapdata.data.get(datind)#[:,:,datind]
         #This is not a mistake
         self.mask.mask=np.where(data>0,1,0)
         globalfuncs.setstatus(self.status,"Mask complete") 
         
    def createNewChannelFromMask(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if len(self.mask.mask)==0:
            print('No Mask')
            globalfuncs.setstatus(self.status,'No mask to use.')
            return
        #check name
        name=tkinter.simpledialog.askstring(title='Add Mask as Channel',prompt="Enter new name for channel.")
        newname=globalfuncs.fixlabelname(name)
        if newname in self.mapdata.labels:
            print('Enter unique channel name')
            globalfuncs.setstatus(self.status,'Enter unique channel name')
            return
        self.savedeconvcalculation(self.mask.mask,newname)
        globalfuncs.setstatus(self.status,"Done")       
        
    def addMaskToChannel(self):
         globalfuncs.setstatus(self.status,"Ready")
         if not self.hasdata:
             print('No Data')
             globalfuncs.setstatus(self.status,'No Data')
             return
         if self.datachan.get()==():
             return
         if len(self.mask.mask)==0:
             print('No Mask')
             globalfuncs.setstatus(self.status,'No mask to use.')
             return
         datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
         data=self.mapdata.data.get(datind)#[:,:,datind]
         #find power of 2...
         factor=globalfuncs.powernext(int(max(np.ravel(data))))
         self.mapdata.data.put(datind,data+self.mask.mask*float(factor))
         self.domapimagefromscaselect()
    
    def delMaskToChannel(self):
         globalfuncs.setstatus(self.status,"Ready")
         if not self.hasdata:
             print('No Data')
             globalfuncs.setstatus(self.status,'No Data')
             return
         if self.datachan.get()==():
             return
         if len(self.mask.mask)==0:
             print('No Mask')
             globalfuncs.setstatus(self.status,'No mask to use.')
             return

         datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
         data=self.mapdata.data.get(datind)#[:,:,datind]
         data=np.where(self.mask.mask>0,0,data)
         self.mapdata.data.put(datind,data)
         self.domapimagefromscaselect()

    
    def invertMaskSelection(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.mask.mask==[]:
            print('No selection in mask')
            globalfuncs.setstatus(self.status, 'No selection in mask')
            return
        newmask=1-self.mask.mask
        self.mask.mask=newmask
        self.domapimagefromscaselect()
        
        
########################## TriColor routines

    def starttcplot(self):
        #show tricolor window if needed
        globalfuncs.setstatus(self.status,"Ready")
        if self.triColorWindow.exist:
            self.triColorWindow.win.show()
        elif self.hasdata:
            ps=TriColorWindowClass.TriColorWindowParams(self.savetcdisplayasjpg, self.viewtricolormap, self.savetricolormap, self.maindisp, self.CMYKOn, self.tcrangedict, self.status, self.dodt, self.deadtimevalue, self.DTICRchanval, self.root, self.xyflip, self.tcrefresh, self.tclegendexist, self.showscalebar,self.showscalebarText, self.tcmarkerupdate)
            self.triColorWindow.create(self.mapdata, ps)
        else:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')


   
    def tcrefresh(self):
        print("in tc refresh")
        try:
            self.dataFileBuffer[self.activeFileBuffer]['zoom']=self.maindisp.zmxyi
        except:
            pass
        #if self.tcimageexists: 
        if self.triColorWindow.exist:
            self.triColorWindow.dotcdisplay()
        if self.correlationPlot.exist: 
            self.correlationPlot.updatezoom(self.maindisp.zmxyi)
            self.correlationPlot.checkcorplot()
        if self.dataSummaryWindow.exist:  self.doDataSummary(self.contoursOn)
        if self.movieView is not None:
            self.movieView.zmxyi=self.maindisp.zmxyi
        if self.contoursOn.get()==1: self.datacontoursToggle()    

    
    def tcmarkerupdate(self,obj,add=1): #self.markerlist{}
        #see if obj in list
        if obj in list(self.tcmarkerlist.keys()):
            #delete
            if self.tcmarkerlist[obj] is not None: self.triColorWindow.tcimframe.delete(self.tcmarkerlist[obj])
            ##self.markerlist[obj]=None
        if not add: return
        #find pos (xpos,ypos)
        (xp,yp)=self.maindisp.datainvcoords(obj.xpos.getvalue(),obj.ypos.getvalue())
        mref=None
        #add new obj #color
            #markertypes=['sm circle','big circle','sm square','big square','sm triangle','big triangle','text'] marker.getvalue()[0]
        if obj.marker.getvalue()[0]=='text':
            #text placement
            mref=self.triColorWindow.tcimframe.create_text(xp,yp,anchor=tkinter.W,fill=obj.color,text=obj.textfield.getvalue())
        else:
            ms=obj.marker.getvalue()[0].split()[0]
            mt=obj.marker.getvalue()[0].split()[1]
            sz=4
            if ms=='big': sz=8
            if mt=='emptycircle':
                mref=self.triColorWindow.tcimframe.create_oval(xp-sz,yp-sz,xp+sz,yp+sz,outline=obj.color)
            if mt=='circle':
                mref=self.triColorWindow.tcimframe.create_oval(xp-sz,yp-sz,xp+sz,yp+sz,fill=obj.color)
            if mt=='emptysquare':
                mref=self.triColorWindow.tcimframe.create_rectangle(xp-sz,yp-sz,xp+sz,yp+sz,outline=obj.color)
            if mt=='square':
                mref=self.triColorWindow.tcimframe.create_rectangle(xp-sz,yp-sz,xp+sz,yp+sz,fill=obj.color)
            if mt=='triangle':
                mref=self.triColorWindow.tcimframe.create_polygon(xp-sz,yp-sz,xp+sz,yp-sz,xp,yp+sz,fill=obj.color,outline='black')
        if mref is not None: self.tcmarkerlist[obj]=mref


    



       


#################################  Math routines

    def startmathwin(self):
        #show maths window if needed
        globalfuncs.setstatus(self.status,"Ready")
        if self.mathWindow.exist:
            self.mathWindow.win.show()
        elif self.hasdata:
            # self.mathwindowexist=1
            # self.createmathwin()
            ps=MathWindowClass.MathWindowParams(Display.DisplayParams(self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,self.dispScaleFactor,self.status,self.maindisp.zmxyi), self.docalcimage, self.addchannel, self.mask)
            self.mathWindow.create(self.mapdata, ps)
        else:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
    
   
    
    def docalcimage(self,data,sc=True):
        globalfuncs.setstatus(self.status,"DISPLAYING...")
        self.maindisp.placeData(transpose(data[::-1,:]),transpose(self.mapdata.mapindex[::-1,:]),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,sc=sc)
        self.showmap()


    def multisum(self):
        #sum over multiple channels
        if not self.hasdata:
            print('No data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #will assume the names are SIMILAR!
        #get root name
        rnam=tkinter.simpledialog.askstring(title='Element Sum',prompt='Enter channel name base: ')
        if rnam=='' or rnam is None:
            print('Cancelled')
            globalfuncs.setstatus(self.status,'Cancelled')
            return
        elems=[]
        for c in self.mapdata.labels:
            if c.rfind(rnam)==0:
                elems.append(c)
        #do calculation
        (xlen,ylen)=self.mapdata.data.shape[:2]
        newdata=zeros((xlen,ylen),dtype=np.float32)
        for c in elems:
            datind=self.mapdata.labels.index(c)+2
            newdata=newdata+self.mapdata.data.get(datind)#[:,:,datind]
        #display
        self.docalcimage(newdata)
        #get new name
        sumnam=tkinter.simpledialog.askstring(title='Element Sum',prompt='Enter new channel name: ')
        #make sure name unique
        sumnam=globalfuncs.fixlabelname(sumnam)
        if sumnam in self.mapdata.labels:
            print('Enter unique channel name')
            globalfuncs.setstatus(self.status,'Enter unique channel name')
            return
        
        #save new channel        
        self.addchannel(newdata,sumnam)        

    #PROCESS
    def startTimeTrans(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.multidoTNdialog=Pmw.SelectionDialog(self.imgwin,title="Perform Time Normalization",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.doTNTrans)
        self.multidoTNdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        
    def doTNTrans(self,result):
        chans=self.multidoTNdialog.getcurselection()
        self.multidoTNdialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        #ask for time channel
        Tind=self.mapdata.labels.index(self.TIMEchan.getvalue())+2
        Tdat=self.mapdata.data.get(Tind)#[:,:,iind]        
        for c in chans:
            datind=self.mapdata.labels.index(c)+2
            ad=self.mapdata.data.get(datind)#[:,:,Aind]
            newdata=ad/Tdat*80000000
            newname=c+"TN"
            if newname in self.mapdata.labels:
                i=1
                while 1:
                    newchname=newname+str(i)
                    if newchname not in self.mapdata.labels:
                        break
                    i+=1
                newname=newchname
            self.addchannel(newdata, newname)
        self.domapimage()


    def startvertTrans(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.transdir='Vert Flip'

        self.multidoflipdialog=Pmw.SelectionDialog(self.imgwin,title="Perform Vertial Flips",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.doFlipTrans)
        self.multidoflipdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)

    
    def starthorzTrans(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.transdir='Horz Flip'

        self.multidoflipdialog=Pmw.SelectionDialog(self.imgwin,title="Perform Vertial Flips",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.doFlipTrans)
        self.multidoflipdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
    
    
    def doFlipTrans(self,result):
        chans=self.multidoflipdialog.getcurselection()
        self.multidoflipdialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        if self.transdir not in ['Vert Flip','Horz Flip']:
            globalfuncs.setstatus(self.status,'No action taken - no direction provided')
            self.transdir = None
            return
        for c in chans:
            datind=self.mapdata.labels.index(c)+2
            ad=self.mapdata.data.get(datind)#[:,:,Aind]
            
            newdata=MathWindowClass.MathOp(self.transdir, ad, None)
            if self.transdir == 'Vert Flip':
                newname=c+'VF'
            else:
                newname=c+'HF'
            if newname in self.mapdata.labels:
                i=1
                while 1:
                    newchname=newname+str(i)
                    if newchname not in self.mapdata.labels:
                        break
                    i+=1
                newname=newchname
            self.addchannel(newdata, newname)
        self.domapimage()

        self.transdir=None


    def startlogTrans(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.multidologdialog=Pmw.SelectionDialog(self.imgwin,title="Perform Log Transforms",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.dologTrans)
        self.multidologdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)

    def dologTrans(self,result):
        chans=self.multidologdialog.getcurselection()
        self.multidologdialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return
        for c in chans:
            datind=self.mapdata.labels.index(c)+2
            ad=self.mapdata.data.get(datind)#[:,:,Aind]
            newdata=MathWindowClass.MathOp('Log', ad, None)
            newname='log'+c
            if newname in self.mapdata.labels:
                i=1
                while 1:
                    newchname=newname+str(i)
                    if newchname not in self.mapdata.labels:
                        break
                    i+=1
                newname=newchname
            self.addchannel(newdata, newname)
        self.domapimage()

################################# Channel Stitching
        
    def openchanstitch(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.stitchprev is not None:
            #kill it
            self.stitchprev.main.destroy()
            self.stitchprev=None
        self.channelStitchdialog=Pmw.Dialog(self.imgwin,title="Channel Stitch",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                            command=self.doneChanStitch)
        
        dh=self.channelStitchdialog.interior()
        dh.configure(background='#d4d0c8')
                
        wmf=Pmw.ScrolledFrame(dh,hscrollmode='none',vscrollmode='static',hull_background='#d4d0c8',usehullsize=1,hull_width=450,hull_height=500,horizflex='expand',vertflex='fixed')
        wmf.pack(side=tkinter.TOP,fill='both')
        h=wmf.interior()
        h.configure(background='#d4d0c8')
        #h=self.channelStitchdialog.interior()
        #h.configure(background='#d4d0c8')
        #new channel name
        self.stitchchanname=Pmw.EntryField(h,labelpos='w',label_text='New Channel Name',entry_width=20,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.stitchchanname.pack(side=tkinter.TOP,padx=5,pady=5)
        #need channel selection and offsets...
        
        self.stitchgroup=[]        
        self.stitchIndexMax=2
        self.stitchMaster=tkinter.Frame(h)
        self.stitchMaster.configure(background='#d4d0c8')
        self.stitchMaster.pack(side=tkinter.TOP,padx=1,pady=1,expand='yes',fill='both')
        self.stitchgroup.append(FaderClass.stitchChannelGroup(self.stitchMaster,self.mapdata.labels,1))
        self.stitchgroup.append(FaderClass.stitchChannelGroup(self.stitchMaster,self.mapdata.labels,2))
        
        #preview button
        b=PmwTtkButtonBox.PmwTtkButtonBox(h,hull_background='#d4d0c8')
        b.add('Preview',command=self.dostitchPreview,style='GREEN.TButton',width=10)
        b.pack(side=tkinter.LEFT,padx=5,pady=5)        
        b.add('Add Channel',command=self.addStitchChan,style='SBLUE.TButton',width=10)
        b.pack(side=tkinter.LEFT,padx=5,pady=5)  
        
        self.channelStitchdialog.show()

    def addStitchChan(self):
        self.stitchIndexMax+=1
        self.stitchgroup.append(FaderClass.stitchChannelGroup(self.stitchMaster,self.mapdata.labels,self.stitchIndexMax))

    def dostitchPreview(self):
        
        for i in self.stitchgroup:
            if i.stitchch.getvalue()==():
                print('Choose data channels to stitch: Channel '+str(i.index)+' is undefined.')
                globalfuncs.setstatus(self.status,'Choose data channels to stitch: Channel '+str(i.index)+' is undefined.')
                return             
        preview=self.doStitch()
        if self.stitchprev is None:
            self.stitchprev=Display.Display(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,main=0,sf=self.dispScaleFactor)
        self.stitchprev.placeData(transpose(preview[::-1,:]),transpose(self.mapdata.mapindex[::-1,:]),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,domask=0,mask=[],datlab='Stitching Preview')
        self.stitchprev.main.lift()

    def doStitch(self):
        #get data, shift, and stitch!
        #get data
        
        (xlen,ylen)=self.mapdata.data.shape[:2]
        newdata=zeros((xlen,ylen),dtype=np.float32)        
        normdata=zeros((xlen,ylen),dtype=np.float32) 
        
        for i in self.stitchgroup:
            i.dataindex=self.mapdata.labels.index(i.stitchch.getvalue()[0])+2
            i.data=self.mapdata.data.get(i.dataindex)
            try: i.v=int(i.vertch.getvalue())
            except: i.v=0
            try: i.h=int(i.horzch.getvalue())
            except: i.h=0 
            i.data=shiftvert(i.data,i.v)
            i.data=shifthorz(i.data,i.h)
            newdata=newdata + i.data
            t=np.where(i.data>0,1,0)
            normdata=normdata + t
        
        divn=np.where(normdata==0,1,normdata)
        
        return newdata/divn
    
    def doneChanStitch(self,result):
        if result=='OK':
            for i in self.stitchgroup:
                if i.stitchch.getvalue()==():
                    print('Choose data channels to stitch: Channel '+str(i.index)+' is undefined.')
                    globalfuncs.setstatus(self.status,'Choose data channels to stitch: Channel '+str(i.index)+' is undefined.')
                    return   
            #validate name
            new=globalfuncs.fixlabelname(self.stitchchanname.getvalue())
            if new in self.mapdata.labels or new=='':
                print('Enter unique channel name')
                globalfuncs.setstatus(self.status,'Enter unique channel name')
                return            
            newchan=self.doStitch()
            #add channel
            self.addchannel(newchan,new)            
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'Stitch cancelled')
        #cleanup
        self.channelStitchdialog.withdraw()
        if self.stitchprev is not None: self.stitchprev.main.destroy()
        self.stitchprev=None                


################################# 3D Image Maker from Stereo Plots
        
    def open3Dmaker(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.stereoprev is not None:
            #kill it
            self.stereoprev.destroy()
            self.stereoprev=None
        self.stereoMakerdialog=Pmw.Dialog(self.imgwin,title="Stereo Image Maker",buttons=('OK','Cancel'),defaultbutton='OK',
                                            command=self.done3Dstereo)
        h=self.stereoMakerdialog.interior()
        h.configure(background='#d4d0c8')
        ###new channel name
        ##self.stereochanname=Pmw.EntryField(h,labelpos='w',label_text='New Channel Name',entry_width=20)
        ##self.stereochanname.pack(side=tkinter.TOP,padx=5,pady=5)
        #parameter types (just color?)
        gp=Pmw.Group(h,tag_text='Stereo Parameters',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        gp.interior().configure(background='#d4d0c8')
        self.stereocolor=Pmw.RadioSelect(gp.interior(),labelpos=tkinter.W,buttontype='radiobutton',label_text='Color Selection: ',label_background='#d4d0c8',hull_background='#d4d0c8',frame_background='#d4d0c8')
        self.stereocolor.add('Red-Cyan',background='#d4d0c8')
        self.stereocolor.add('Red-Blue',background='#d4d0c8')
        self.stereocolor.add('Orange-Blue',background='#d4d0c8')
        self.stereocolor.setvalue('Red-Cyan')
        self.stereocolor.pack(side=tkinter.TOP,padx=2,pady=2)
        gp.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')        
        #need channel selection and offsets...
        g1=Pmw.Group(h,tag_text='Left Channel',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        self.stereoch1=Pmw.ScrolledListBox(g1.interior(),labelpos='n',label_text='Select Channel',items=self.mapdata.labels,listbox_selectmode=tkinter.SINGLE,
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=tkinter.DISABLED,listbox_height=5,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.stereoch1.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')
        self.stereovertch1=Pmw.Counter(g1.interior(),labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',label_text='Vertical Shift',datatype='numeric',entryfield_value=0,entry_width=10)
        self.stereohorzch1=Pmw.Counter(g1.interior(),labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',label_text='Horizontal Shift',datatype='numeric',entryfield_value=0,entry_width=10)
        self.stereointench1=Pmw.Counter(g1.interior(),labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',label_text='Intensity',entryfield_value=100,entry_width=5,entryfield_validate={'validator' : 'integer','min' : 0, 'max' : 1000})
        self.stereovertch1.pack(side=tkinter.LEFT,padx=4,pady=5)
        self.stereohorzch1.pack(side=tkinter.LEFT,padx=4,pady=5)
        self.stereointench1.pack(side=tkinter.LEFT,padx=4,pady=5)
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')

        g2=Pmw.Group(h,tag_text='Right Channel',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g2.interior().configure(background='#d4d0c8')
        self.stereoch2=Pmw.ScrolledListBox(g2.interior(),labelpos='n',label_text='Select Channel',items=self.mapdata.labels,listbox_selectmode=tkinter.SINGLE,
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=tkinter.DISABLED,listbox_height=5,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.stereoch2.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')          
        self.stereovertch2=Pmw.Counter(g2.interior(),labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',label_text='Vertical Shift',datatype='numeric',entryfield_value=0,entry_width=10)
        self.stereohorzch2=Pmw.Counter(g2.interior(),labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',label_text='Horizontal Shift',datatype='numeric',entryfield_value=0,entry_width=10)
        self.stereointench2=Pmw.Counter(g2.interior(),labelpos='n',hull_background='#d4d0c8',label_background='#d4d0c8',label_text='Intensity',entryfield_value=100,entry_width=5,entryfield_validate={'validator' : 'integer','min' : 0, 'max' : 1000})
        self.stereovertch2.pack(side=tkinter.LEFT,padx=4,pady=5)
        self.stereohorzch2.pack(side=tkinter.LEFT,padx=4,pady=5)
        self.stereointench2.pack(side=tkinter.LEFT,padx=4,pady=5)
        g2.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')

        #preview button
        b=PmwTtkButtonBox.PmwTtkButtonBox(h,hull_background='#d4d0c8')
        b.add('Preview',command=self.dostereoPreview,style='GREEN.TButton',width=10)
        b.add('Save',command=self.doStereoSave,style='ORANGE.TButton',width=10)
        b.pack(side=tkinter.TOP,padx=5,pady=5)        

        self.stereoMakerdialog.show()

    def doStereoSave(self):
        self.dostereoPreview()
        if not self.stereoprev is None:
            print('No stereo plot')
            globalfuncs.setstatus(self.status,'No stereo plot')
            return
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_3D.jpg'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving stereo image display...")
        self.stereoprev.lift()
        self.imgwin.update()
        #save image
        rx=int(self.stereoimframe.winfo_rootx())
        ry=int(self.stereoimframe.winfo_rooty())
        rw=int(self.stereoimframe.winfo_width())
        rh=int(self.stereoimframe.winfo_height())
        screencapture.capture(rx,ry,rw,rh,fn)
        #im=ImageGrab.grab((rx,ry,rx+rw,ry+rh))
        #im.save(fn)            
        globalfuncs.setstatus(self.status,"Stereo image display saved in: "+fn)


    def dostereoPreview(self):
        if self.stereoch1.getvalue()==() or self.stereoch2.getvalue()==():
            #not enough info
            print('Choose data channels to stereo-ize')
            globalfuncs.setstatus(self.status,'Choose data channels to stereo-ize')
            return            
        if self.stereoprev is None:
            self.stereocreate()
        self.doStereo()
        #apply zoom issues??
##        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
##            pass
##            len_x, len_y=self.mapdata.data[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2],:].shape[:2]
##            tdata=self.mapdata.data[::-1,:,:]
##            tdata=tdata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2],:]
##            tdata=tdata[::-1,:,:]
##            self.stereoprev=Display.Display(self.imgwin,self.viewMCAplottoggle,self.showMCApix,self.showscalebar,self.showscalebarText,self.xyflip,main=0)
        #add data
            
##        self.stereoprev.placeData(transpose(preview[::-1,:]),transpose(self.mapdata.mapindex[::-1,:]),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,domask=0,mask=[],datlab='Stitching Preview')
##        self.stereoprev.main.lift()

    def stereocreate(self):
        self.stereoprev=Pmw.MegaToplevel(self.imgwin)
        self.stereoprev.title('3D Stereo Preview')
        hf=self.stereoprev.interior()
        self.stereoimframe=tkinter.Canvas(hf,bg='black',borderwidth=2, height=250, width=250, cursor='crosshair')
        self.stereoimframe.pack(side=tkinter.LEFT,fill=tkinter.X)
        self.stereoitems=[]
    
    def doStereo(self):
        #get data, shift, and stereo!
        len_x, len_y=self.mapdata.data.shape[:2]
        #get data
        Aind=self.mapdata.labels.index(self.stereoch1.getvalue()[0])+2
        Bind=self.mapdata.labels.index(self.stereoch2.getvalue()[0])+2
        ch1=self.mapdata.data.get(Aind)#[:,:,Aind]
        ch2=self.mapdata.data.get(Bind)#[:,:,Bind]
        #apply deadtimes
        if self.dodt.get()==1:
            icr=self.mapdata.data.get(self.DTICRchanval)#[:,:,self.DTICRchanval]
            dtcor=np.exp(float(self.deadtimevalue.getvalue())*1e-6*icr)
        else:
            dtcor=1.
        ch1=ch1*dtcor
        ch2=ch2*dtcor
        try: v1=int(self.stereovertch1.getvalue())
        except: v1=0
        try: h1=int(self.stereohorzch1.getvalue())
        except: h1=0        
        try: v2=int(self.stereovertch2.getvalue())
        except: v2=0
        try: h2=int(self.stereohorzch2.getvalue())
        except: h2=0
        try: i1=int(self.stereointench1.getvalue())
        except: i1=100
        try: i2=int(self.stereointench2.getvalue())
        except: i2=100
        #shift verts
        ch1=shiftvert(ch1,v1)
        ch2=shiftvert(ch2,v2)
        #shift horzs
        ch1=shifthorz(ch1,h1)
        ch2=shifthorz(ch2,h2)

        #take zoom
        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
            len_x, len_y=self.mapdata.data.get(0)[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]].shape[:2]
            ch1=ch1[::-1,:]
            ch1=ch1[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            ch1=ch1[::-1,:]
            ch2=ch2[::-1,:]
            ch2=ch2[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            ch2=ch2[::-1,:]
        #blank matrix for rgb-ing
        newdata=zeros((len_y,len_x,3),dtype=np.float32)
        #combine in proper rgb format
        dmod=2
        if self.stereocolor.getvalue()=='Red-Cyan':
            reddata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch1[::-1,:]),(None,None),100/float(self.stereointench1.getvalue()),0.0)
            greendata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch2[::-1,:]),(None,None),100/float(self.stereointench2.getvalue())*dmod,0.0)
            bluedata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch2[::-1,:]),(None,None),100/float(self.stereointench2.getvalue())*dmod,0.0)
        if self.stereocolor.getvalue()=='Red-Blue':
            reddata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch1[::-1,:]),(None,None),100/float(self.stereointench1.getvalue()),0.0)
            bluedata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch2[::-1,:]),(None,None),100/float(self.stereointench2.getvalue()),0.0)
        if self.stereocolor.getvalue()=='Orange-Blue':
            reddata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch1[::-1,:]),(None,None),100/float(self.stereointench1.getvalue())*dmod,0.0)
            greendata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch1[::-1,:]),(None,None),100/float(self.stereointench1.getvalue())*dmod,0.0)
            bluedata,(scalex,scaley)=Display.preprocess(self.root,transpose(ch2[::-1,:]),(None,None),100/float(self.stereointench2.getvalue()),0.0)
        newdata[:,:,0]=reddata
        if self.stereocolor.getvalue()!='Red-Blue': newdata[:,:,1]=greendata
        newdata[:,:,2]=bluedata

##        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:     
##            len_x, len_y=self.mapdata.data[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2],:].shape[:2]
##            newdata=newdata[::-1,:,:]
##            newdata=newdata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2],:]
##            newdata=newdata[::-1,:,:]

        #match display order
        newdata=newdata[::self.maindisp.xdir,::self.maindisp.ydir]
        #convert to bin
        newdata=newdata.astype('b')
        #check for flip
        if self.xyflip.get(): newdata=transpose(newdata)        
        if 1:
            pilim=ImRadon.toimage(transpose(newdata),cmin=0,skip=1)
        (w,h)=pilim.size        
        pilim=pilim.resize((int(w*scalex),int(h*scaley)))
        self.stereoimage=ImageTk.PhotoImage(pilim)
        #clear        
        if self.stereoitems !=[] : self.stereoimframe.delete(self.stereoitems.pop())
        #rescale canvas
        self.stereoimframe.config(height=int(h*scaley),width=int(w*scalex))
        self.stereoitems.append(self.stereoimframe.create_image((int(w*scalex+scalex))/2,(int(h*scaley+scaley))/2,anchor='center', image=self.stereoimage))
    
    def done3Dstereo(self,result):
        if result=='OK':
            self.doStereo()
            return
##            if self.stereoch1.getvalue()==() or self.stereoch2.getvalue()==():
##                #not enough info
##                print 'Choose data channels to stereo-ize'
##                globalfuncs.setstatus(self.status,'Choose data channels to stereo-ize')
##                return    
##            #validate name
##            new=globalfuncs.fixlabelname(self.stereochanname.getvalue())
##            if new in self.mapdata.labels or new=='':
##                print 'Enter unique channel name'
##                globalfuncs.setstatus(self.status,'Enter unique channel name')
##                return            
##            newchan=self.doStereo()
##            #add channel
##            self.addchannel(newchan,new)            
        #cleanup
        self.stereoMakerdialog.withdraw()
        if self.stereoprev is not None: self.killstereowin()

    def killstereowin(self):
        self.stereoprev.destroy()
        self.stereoitems=[]
        self.stereoprev=None

#################################  MCA viewing routines

    def getMCAfile(self,pex=False,retval=False):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            if pex:
                print('EXAFS MCA data ok')
                globalfuncs.setstatus(self.status,'EXAFS MCA data ok')
            else:
                return 0
        #get file name
        fty=[("HDF5 data files","*.hdf5"),("MCA data files","*.mca"),("Binary MCA data files","*.bmd"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())
        if t!='':
            self.MCAfilename=t
        else:
            globalfuncs.setstatus(self.status,"No MCA data defined")
            return 0
        globalfuncs.setstatus(self.status,"MCA data stored in: "+self.MCAfilename)
        if retval:
            return 1,self.MCAfilename
        else:
            return 1


    def setMCAslopeValue(self):
        mt='Enter number of ev per MCA bin'
        nv=tkinter.simpledialog.askfloat(title='MCA properties',prompt=mt,initialvalue=self.currentMCAXraySlope)
        if nv==None or nv=='': return
        self.currentMCAXraySlope=float(nv)
        if self.MCAfilename=='': 
            return
        else:
            self.currentMCAXvalues = np.arange(self.mcamaxno)*self.currentMCAXraySlope
        self.MCAreplotBuffers()

    def MCAlinestep(self):
        mt='Enter total number of energy sweeps in MCA file: '
        self.MCAnofiles=tkinter.simpledialog.askinteger(title='MCA file properties',prompt=mt,initialvalue=self.MCAnofiles)
        mt='Enter desired sweep number in MCA file: '
        if self.MCAnofiles<1: self.MCAnofiles=1
        if self.MCAoffset>self.MCAnofiles:self.MCAoffset=self.MCAnofiles
        ans=tkinter.simpledialog.askinteger(title='MCA file properties',prompt=mt,initialvalue=self.MCAoffset)
        if ans>self.MCAnofiles:
            self.MCAoffset=self.MCAnofiles
        else:
            self.MCAoffset=ans
        if self.MCAoffset<1: self.MCAoffset=1            

    def MCAsetpixofs(self):
        mt='Enter pixel offset per MCA line: '
        self.MCApixoffs=tkinter.simpledialog.askinteger(title='MCA file properties',prompt=mt,initialvalue=self.MCApixoffs)
        #if self.MCApixoffs<0: self.MCApixoffs=0

    def MCAset1stpixofs(self):
        mt='Enter pixel offset on first MCA line: '
        self.MCA1stpixoffs=tkinter.simpledialog.askinteger(title='MCA file properties',prompt=mt,initialvalue=self.MCA1stpixoffs)
        
    def showMCApix(self,pixno,multi=0):
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        startt=time.process_time()
        (fn,ext)=os.path.splitext(self.MCAfilename)
        if ext==".bmd":
            (self.MCArawdata,xtext)=self.getBinaryMCAfromFile(pixno,multi)
        elif ext==".hdf5":
            (self.MCArawdata,xtext)=self.getHDF5MCAfromFile(pixno,multi)
        else:
            if not multi:
                pixno=int(pixno+1)
                globalfuncs.setstatus(self.status,'Getting MCA at pixel: '+str(pixno))
                #fix pixno for multiple MCA files...
                #print (pixno/self.mapdata.nxpts)
                row=(pixno/self.mapdata.nxpts)
                pixno=pixno+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs
                #print pixno
                #open MCA file and get line pixno:
                fid=open(self.MCAfilename,"rU")
                if self.mapdata.type=='BL62TXM':
                    self.mcamaxno=int(fid.readline().split()[2])
                    fid.readline()
                else:
                    self.mcamaxno=2048
                if pixno>25:
                    for i in range(pixno-25):
                        line=fid.readline()
                linenum=pixno-25
                self.MCArawdata=[]
                early=0
                if pixno<0:
                    early=1
                    linenum=pixno
                while linenum !=pixno:
                    line=fid.readline()
                    try:
                        linenum=int(line.split()[0])
                    except:
                        globalfuncs.setstatus(self.status,'EARLY End of MCA file')
                        early=1
                        break                    
                if not early:
                    i=0
                    for p in line.split():
                        if i!=0:self.MCArawdata.append(int(p))
                        i=i+1
                else:
                    self.MCArawdata=zeros(self.mcamaxno,dtype=np.float32)
                xtext="MCA Spectrum at pixel "+str(pixno)
                fid.close()
            else:
                pixno=array(pixno,'int')+1
                pixno=np.sort(pixno)
                newpixno=[]
                for p in pixno:
                    row=(p/self.mapdata.nxpts)
                    p=p+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs
                    newpixno.append(p)
                pixno=array(newpixno,'int')
                globalfuncs.setstatus(self.status,'Getting MCA at mutliple pixels... ')
                #open MCA file and get line pixno:
                fid=open(self.MCAfilename,"rU")
                if self.mapdata.type=='BL62TXM':
                    self.mcamaxno=int(fid.readline().split()[2])
                    fid.readline()
                else:
                    self.mcamaxno=2048
                self.MCArawdata=zeros(self.mcamaxno,dtype=np.float32)
                linenum=0
                for pix in pixno:
                    if pix<0: continue
                    if pix>linenum+25:
                        for i in range(pix-25-linenum):
                            line=fid.readline()
                    while linenum !=pix:
                        line=fid.readline()
                        try:
                            linenum=int(line.split()[0])
                        except:
                            globalfuncs.setstatus(self.status,'EARLY End of MCA file in average')
                            break
                    i=0
                    curmca=[]
                    for p in line.split():
                        try:
                            if i!=0:curmca.append(int(p))
                            i=i+1
                        except:
                            break
                    if curmca==[]: curmca=zeros(self.mcamaxno,dtype=np.float32)
                    self.MCArawdata=self.MCArawdata+array(curmca)
                xtext="MCA Spectrum Area Average"
                fid.close()
        #account for average
        if self.MCASUMOPT.get()=='Average':
            if multi:
                self.MCArawdata=self.MCArawdata/len(pixno)
        print(time.process_time()-startt)
        self.setMCABuffer()
        self.MCAgraphtitle=xtext
        if self.MCAviewexist:
            self.MCAgraph.hasUnitsEnergy=False
        self.updateMCAgraph()
        
    def updateMCAgraph(self,plot=True):
        #define new window if needed
        if not self.MCAviewexist:
            self.MCAgraphBarMarker=None
            self.MCAviewexist=1
            self.newMCAplot=Pmw.MegaToplevel(self.imgwin)
            self.newMCAplot.title('MCA Spectrum View')
            self.newMCAplot.userdeletefunc(func=self.killMCAplot)           
            h=self.newMCAplot.interior()
            h.configure(background='#d4d0c8')
            #menubar for MCA...
            menubar=PmwTtkMenuBar.PmwTtkMenuBar(h)
            if os.sys.platform=='win32': menubar.component('hull').configure(bg='#d4d0c8')
            menubar.addmenu('MCA','')
            menubar.addmenuitem('MCA','command',label='Define MCA file',command=self.getMCAfile)
            menubar.addmenuitem('MCA','command',label='Save MCA Spectrum',command=self.saveMCAdata)
            menubar.addmenuitem('MCA','command',label='Export MCA Spectrum to Clipboard',command=self.exportMCAdata)
            menubar.addmenuitem('MCA','separator')
            menubar.addmenuitem('MCA','command',label='Correct MCA for bad lines',command=self.MCAdeglitch)
            menubar.addmenuitem('MCA','separator')
            menubar.addmenuitem('MCA','command',label='Rebin MCA to Data',command=self.MCArebindata)
            menubar.addmenuitem('MCA','command',label='Rebin MCA Using Value-Axis to Data',command=self.MCArebindataValue)
            menubar.addmenuitem('MCA','command',label='Rebin MCA Ranges to Data',command=self.MCArebindataRange)


            #menubar.addmenuitem('MCA','command',label='Fit MCA to Data',command=self.MCAfitdata)
            menubar.addmenuitem('MCA','separator')
            menubar.addcascademenu('MCA','XRF Fit Type')
            self.MCAXRFfitType=tkinter.StringVar()
            menubar.addmenuitem('XRF Fit Type','radiobutton',label='Fast-32',variable=self.MCAXRFfitType)
            menubar.addmenuitem('XRF Fit Type','radiobutton',label='Fast-64',variable=self.MCAXRFfitType)
            menubar.addmenuitem('XRF Fit Type','radiobutton',label='Individual Pixel',variable=self.MCAXRFfitType)
            self.MCAXRFfitType.set('Fast-64')
            
            menubar.addmenu('View','')
            menubar.addcascademenu('View','Trace Scale')
            self.MCAtracescalevar=tkinter.StringVar()
            self.MCAtracescalevar.set('Linear')
            menubar.addmenuitem('Trace Scale','radiobutton',label='Linear',command=self.updateMCAgraphscale,variable=self.MCAtracescalevar)
            menubar.addmenuitem('Trace Scale','radiobutton',label='Log',command=self.updateMCAgraphscale,variable=self.MCAtracescalevar)
            menubar.addmenuitem('View','separator')
            menubar.addcascademenu('View','MCA View Options')
            menubar.addmenuitem('MCA View Options','radiobutton',label='Bins',command=tkinter.DISABLED,variable=self.MCAXAXIS)
            menubar.addmenuitem('MCA View Options','radiobutton',label='Spectrum',command=self.setMCAslopeValue,variable=self.MCAXAXIS)
            menubar.addmenuitem('View','separator')
            menubar.addmenuitem('View','command',label='Replot',command=self.MCAreplotBuffers)
            menubar.addmenuitem('View','separator')
            menubar.addcascademenu('View','Spectrum Buffers')
            self.MCAplotBufferSwitch1=tkinter.IntVar()
            self.MCAplotBufferSwitch1.set(1)
            self.MCAplotBufferSwitch2=tkinter.IntVar()
            self.MCAplotBufferSwitch2.set(0)
            self.MCAplotBufferSwitch3=tkinter.IntVar()
            self.MCAplotBufferSwitch3.set(0)
            self.MCAplotBufferSwitch4=tkinter.IntVar()
            self.MCAplotBufferSwitch4.set(0)
            self.MCAplotBufferSwitch5=tkinter.IntVar()
            self.MCAplotBufferSwitch5.set(0)
            self.MCAplotBufferSwitch6=tkinter.IntVar()
            self.MCAplotBufferSwitch6.set(0)
            self.MCAplotBufferSwitch7=tkinter.IntVar()
            self.MCAplotBufferSwitch7.set(0)
            self.MCAplotBufferSwitch8=tkinter.IntVar()
            self.MCAplotBufferSwitch8.set(0)
            self.MCAplotBufferSwitch9=tkinter.IntVar()
            self.MCAplotBufferSwitch9.set(0)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 1',command=lambda: self.setActiveMCABuffer(0),variable=self.MCAplotBufferSwitch1)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 2',command=lambda: self.setActiveMCABuffer(1),variable=self.MCAplotBufferSwitch2)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 3',command=lambda: self.setActiveMCABuffer(2),variable=self.MCAplotBufferSwitch3)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 4',command=lambda: self.setActiveMCABuffer(3),variable=self.MCAplotBufferSwitch4)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 5',command=lambda: self.setActiveMCABuffer(4),variable=self.MCAplotBufferSwitch5)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 6',command=lambda: self.setActiveMCABuffer(5),variable=self.MCAplotBufferSwitch6)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 7',command=lambda: self.setActiveMCABuffer(6),variable=self.MCAplotBufferSwitch7)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 8',command=lambda: self.setActiveMCABuffer(7),variable=self.MCAplotBufferSwitch8)
            menubar.addmenuitem('Spectrum Buffers','checkbutton',label='Buffer 9',command=lambda: self.setActiveMCABuffer(8),variable=self.MCAplotBufferSwitch9)
            menubar.addmenuitem('View','command',label='Clear Buffers',command=self.clearMCABuffers)
            menubar.pack(side=tkinter.TOP,fill=tkinter.X) 
            self.MCABufferMenu=menubar
            self.setActiveMCABuffer(0,up=False)
            ls=tkinter.Frame(h, background='#d4d0c8')
            ls.pack(side=tkinter.TOP,fill='both')
            self.MCAgraph=MyGraph.MyGraph(ls,whsize=(6.5,3),tool=1,graphpos=[[.15,.1],[.9,.9]],side=tkinter.LEFT)
            self.MCAgraph.fluolines=[]
            self.MCAgraph.hasUnitsEnergy=False
            #self.MCAgraph.legend_configure(hide=1)
            #self.MCAgraph.pack(side=tkinter.TOP,expand=1,fill='both',padx=2)
            #self.MCAgraph.bind(sequence="<ButtonPress>",   func=self.MCAmouseDown)
            #self.MCAgraph.bind(sequence="<ButtonRelease>", func=self.MCAmouseUp  )
            #self.MCAgraph.bind(sequence="<Motion>", func=self.MCAcoordreport)
            
            g2=tkinter.Frame(ls, relief=tkinter.GROOVE,bd=2, background='#d4d0c8')
            g2.pack(side=tkinter.LEFT,padx=2,pady=2, fill=tkinter.BOTH, expand=1)
            l=tkinter.Label(g2,text="Fitting",relief=tkinter.RAISED,bd=2, background='#d4d0c8')
            l.pack(side=tkinter.TOP,fill=tkinter.X,padx=2,pady=4)
            w=15
            bb=PmwTtkButtonBox.PmwTtkButtonBox(g2,labelpos='n',label_text='Fit Actions:',orient='vertical',pady=3,padx=5,frame_background='#d4d0c8',hull_background='#d4d0c8',label_background='#d4d0c8')
            bb.add('View Config',command=self.openMCAfitConfig,style='SBLUE.TButton',width=w)            
            bb.add('Load Config',command=self.loadMCAfitConfig,style='NAVY.TButton',width=w)
            bb.add('Save Config',command=self.saveMCAfitConfig,style='BROWN.TButton',width=w)
            bb.add('Fit Current',command=self.testMCAfitdata,style='LGREEN.TButton',width=w)
            bb.add('Fit Zoom MCA',command=self.MCAfitZoomdata,style='OGREEN.TButton',width=w)
            bb.add('Fit All MCA',command=self.MCAfitAlldata,style='GREEN.TButton',width=w)
            bb.pack(side=tkinter.TOP,fill='both',padx=2,pady=5)
    
            
            
            xyf=tkinter.Frame(h,background='#d4d0c8')
            xyf.pack(side=tkinter.TOP,fill='both')
##            self.MCAxcoord=tkinter.Label(xyf,text="X=     ",width=15,bd=2,relief=tkinter.RIDGE,anchor=tkinter.W,fg='red')
##            self.MCAycoord=tkinter.Label(xyf,text="Y=     ",width=15,bd=2,relief=tkinter.RIDGE,anchor=tkinter.W,fg='red')
##            self.MCAycoord.pack(side=tkinter.RIGHT,fill=tkinter.X)
##            self.MCAxcoord.pack(side=tkinter.RIGHT,fill=tkinter.X)
        if plot: self.MCAreplotBuffers()

    def clearMCABuffers(self):
        self.MCArawdataBuffer=[[],[],[],[],[],[],[],[],[]]
        self.MCArawdata=[]
        self.setActiveMCABuffer(0,up=False)
        
    def setActiveMCABuffer(self,buf,up=True):
        if not self.MCAviewexist:
            self.MCAactiveBuffer=buf
            self.MCArawdata=self.MCArawdataBuffer[self.MCAactiveBuffer]
            return
        l=[self.MCAplotBufferSwitch1,self.MCAplotBufferSwitch2,self.MCAplotBufferSwitch3,self.MCAplotBufferSwitch4,
           self.MCAplotBufferSwitch5,self.MCAplotBufferSwitch6,self.MCAplotBufferSwitch7,self.MCAplotBufferSwitch8,
           self.MCAplotBufferSwitch9]
        if l[buf].get()==1:
            active=buf
        else:
            active=1
            for i in range(9):
                if l[i].get()==1:
                    active=i
        self.MCAactiveBuffer=active
        menu=self.MCABufferMenu.component('Spectrum Buffers' + '-menu')
        for j in range(9):
            if j==active:
                menu.entryconfig(j,foreground='red')
            else:                
                menu.entryconfig(j,foreground='black')
        self.MCArawdata=self.MCArawdataBuffer[self.MCAactiveBuffer]
        if up: self.MCAreplotBuffers()

    def setMCABuffer(self):
        self.MCArawdataBuffer[self.MCAactiveBuffer]=self.MCArawdata
        
    def MCAreplotBuffers(self):
        l=[self.MCAplotBufferSwitch1,self.MCAplotBufferSwitch2,self.MCAplotBufferSwitch3,self.MCAplotBufferSwitch4,
           self.MCAplotBufferSwitch5,self.MCAplotBufferSwitch6,self.MCAplotBufferSwitch7,self.MCAplotBufferSwitch8,
           self.MCAplotBufferSwitch9]
        palette=sblite.color_palette(n_colors=8)
        c=['#ffff9d']#,'#488f31','#51a676','#88c580','#c2e38c','#fdd172','#f7a258','#ea714e','#de425b']
        c.extend(palette.as_hex())
            #clear old
        self.MCAgraph.cleargraphs()
##            glist=self.MCAgraph.element_names()
##            if glist !=():
##                for g in glist:
##                    self.MCAgraph.element_delete(g)            
        #make graphs
        if self.MCAXAXIS.get()==('Bins'):
            MCAx=list(range(self.mcamaxno))
        else:
            if len(self.currentMCAXvalues) != len(self.MCArawdataBuffer[0]):
                MCAx=list(range(self.mcamaxno))
            else:
                MCAx=self.currentMCAXvalues
        self.MCAgraph.setTitle(self.MCAgraphtitle)
        #self.MCAgraph.configure(title=xtext)
        #self.MCAgraph.line_create('MCA',xdata=tuple(MCAx),ydata=tuple(self.MCArawdata),symbol='',color='yellow')
        if self.MCAtracescalevar.get()=='Log':
            log='semilogy'
        else:
            log=None
        for i in range(9):
            if l[i].get()==1 and self.MCArawdataBuffer[i]!=[]:
                self.MCAgraph.plot(tuple(MCAx),tuple(self.MCArawdataBuffer[i]),text='MCA',color=c[i],log=log)        
        self.MCAgraph.draw()
        self.newMCAplot.show()
        globalfuncs.setstatus(self.status,'Ready')

    def openMCAfitConfig(self):
        ##need chanDict
        cdt={}
        i=0
        for i in range(len(self.mapdata.labels)):
            u=1
            m=max(ravel(self.mapdata.data.get(i+2)))+1
            b=np.mod(ravel(self.mapdata.data.get(i+2)),1)
            if m>31: u=0
            if sum(abs(b))>0: u=0
            if len(np.where(self.mapdata.data.get(i+2)<0)[0])>1: u=0
            cdt[self.mapdata.labels[i]]=[int(m),u]
        if self.pyMCAparamDialog is None:
            self.pyMCAparamDialog=pyMcaParamGUI.PyMcaParameterDialog(self.root,graphwid=self.MCAgraph,closeCallBack=self.closeMCAfitConfig,energy=self.mapdata.energy,chanDict=cdt)
            
    def closeMCAfitConfig(self):
        self.pyMCAparamDialog=None

    def loadMCAfitConfig(self):
        if getattr(sys, 'frozen', False):
            apppath=os.path.dirname(sys.executable)+os.sep
            if sys.platform=='darwin':
                if 'MacOS' in apppath: apppath=apppath.replace('MacOS','Resources')
        elif __file__:
            apppath=os.path.dirname(__file__)
        
        fn=globalfuncs.ask_for_file([("PyMCA Config","*.cfg"),("SUPER files","*.*G"),("all files","*")],apppath+"pyMcaConfigs"+os.sep)
        if fn=='': return
        if self.pyMCAparamDialog is None: 
            print("Need to open dialog first")
            self.openMCAfitConfig()
        self.pyMCAparamDialog.clearAllPeakLine()
        self.pyMCAparamDialog.readConfiguration(file=fn)     
        self.pyMCAparamDialog.populateParameters()
        self.pyMCAlastconfig=fn
    
    def saveMCAfitConfig(self):  
        if getattr(sys, 'frozen', False):
            apppath=os.path.dirname(sys.executable)+os.sep
            if sys.platform=='darwin':
                if 'MacOS' in apppath: apppath=apppath.replace('MacOS','Resources')
        elif __file__:
            apppath=os.path.dirname(__file__)        
        
        if self.pyMCAparamDialog is None: 
            print("Need to open dialog first")
            return
        self.pyMCAparamDialog.getGUIParameters()
        fn=globalfuncs.ask_save_file('config.cfg',apppath+'pyMcaConfigs')
        if fn=='': return
        self.pyMCAparamDialog.saveConfiguration(fn)
        self.pyMCAlastconfig=fn
    
    def testMCAfitdata(self):
        
        if getattr(sys, 'frozen', False):
            apppath=os.path.dirname(sys.executable)+os.sep
            if sys.platform=='darwin':
                if 'MacOS' in apppath: apppath=apppath.replace('MacOS','Resources')
        elif __file__:
            apppath=os.path.dirname(__file__)
        
        if self.pyMCAparamDialog is None and self.pyMCAlastconfig is None:
            cf=apppath+"pyMcaConfigs"+os.sep+"defaultPyMCAConfig.cfg"
            self.openMCAfitConfig()
            self.pyMCAparamDialog.clearAllPeakLine()
            self.pyMCAparamDialog.readConfiguration(file=cf)     
            self.pyMCAparamDialog.populateParameters()

        elif self.pyMCAparamDialog is None and self.pyMCAlastconfig is not None:
            cf=self.pyMCAlastconfig
            self.openMCAfitConfig()
            self.pyMCAparamDialog.clearAllPeakLine()
            self.pyMCAparamDialog.readConfiguration(file=cf)     
            self.pyMCAparamDialog.populateParameters()
            
        else:
            cf=apppath+"pyMcaConfigs"+os.sep+"activePyMCAConfig.cfg"
            self.pyMCAparamDialog.getGUIParameters()
            self.pyMCAparamDialog.saveConfiguration(cf)
            self.pyMCAlastconfig=cf
            
        
        mcaWrap=pyMcaFitWrapper.Wrapper(pkm=cf)

        mcaWrap.setData(self.MCArawdata)
        
        useConc=False
        if self.pyMCAparamDialog.matrixType.getvalue()=="Single":
            useConc=True
        fit,cfit,flist=mcaWrap.doFit(useConc=useConc)            
        [xw,ydata,yfit0,zz]=mcaWrap.getFitData()
        
        fp=mcaWrap.getFittedParameters()
        self.pyMCAparamDialog.populateFittedParameters(fp)
        
        #plot it
        self.MCAgraph.cleargraphs()
        self.MCAgraph.setTitle(self.MCAgraphtitle+" Fit")
        if self.MCAtracescalevar.get()=='Log':
            log='semilogy'
        else:
            log=None

        for group in mcaWrap.mcafitresult['groups']:
            print((group,mcaWrap.mcafitresult[group]['fitarea'],' +/- ', \
                mcaWrap.mcafitresult[group]['sigmaarea'],mcaWrap.mcafitresult[group]['mcaarea']))
        if useConc:
            for ind in range(len(flist)):
                print(flist[ind],cfit[ind])

        self.MCAgraph.hasUnitsEnergy=True
        self.MCAgraph.plot(tuple(xw),tuple(ydata),text='MCA',color='yellow',log=log)        
        self.MCAgraph.plot(tuple(xw),tuple(yfit0),text='MCAf',color='lightgreen',log=log)        
        self.MCAgraph.plot(tuple(xw),tuple(zz),text='MCAb',color='pink',log=log)        

        self.MCAgraph.draw()
        self.newMCAplot.show()
        globalfuncs.setstatus(self.status,'Ready')
        


    def getBinaryMCAfromFile(self,pixno,multi):
        #have binary data...
        data=[]
        fid=open(self.MCAfilename,'rb')
        self.mcamaxno=struct.unpack('i',fid.read(4))[0]
        linenum=0
        early=0
        if not multi:
            pixno=int(pixno+1)
            globalfuncs.setstatus(self.status,'Getting MCA at pixel: '+str(pixno))
            row=(pixno/self.mapdata.nxpts)
            pixno=pixno+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs        
            if pixno<0:
                globalfuncs.setstatus(self.status,'EARLY End of MCA file')
                linenum=pixno
            while linenum !=pixno:
                try:
                    linenum=int(struct.unpack('f',fid.read(4))[0])
                    datline=fid.read(self.mcamaxno*4)
                except:
                    globalfuncs.setstatus(self.status,'EARLY End of MCA file')
                    early=1
                    break
            #print linenum,early
            if not early:
                fmt=str(self.mcamaxno)+'f'
                data=struct.unpack(fmt,datline)
            else:
                data=zeros(self.mcamaxno,dtype=np.float32)
            xtext="MCA Spectrum at pixel "+str(pixno)
            fid.close()
        else:
            pixno=array(pixno,'int')+1
            pixno=np.sort(pixno)
            newpixno=[]
            for p in pixno:
                row=(p/self.mapdata.nxpts)
                p=p+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs
                newpixno.append(p)
            pixno=array(newpixno,'int')
            globalfuncs.setstatus(self.status,'Getting MCA at mutliple pixels... ')
            data=zeros(self.mcamaxno,dtype=np.float32)
            for pix in pixno:
                if pix<0: continue
                while linenum !=pix:
                    try:
                        linenum=int(struct.unpack('f',fid.read(4))[0])
                        datline=fid.read(self.mcamaxno*4)
                    except:
                        globalfuncs.setstatus(self.status,'EARLY End of MCA file')
                        early=1
                        break
                #print pix,early
                fmt=str(self.mcamaxno)+'f'
                try:
                    curmca=struct.unpack(fmt,datline)
                except:
                    curmca=zeros(self.mcamaxno,dtype=np.float32)
                data=data+array(curmca)
            xtext="MCA Spectrum Area Average"
            fid.close()
        return data,xtext

    def getHDF5MCAfromFile(self,pixno,multi):
        #have hdf5 data...
        fid=h5py.File(self.MCAfilename)
        if "/main/mcadata" in fid:
            mcadata=fid['/main/mcadata']
        elif "/main/oodata" in fid:
            mcadata=fid['/main/oodata']
        else:
            print('no mcadata found')
            return

        self.mcamaxno=mcadata.shape[1]
        maxlines=mcadata.shape[0]
        print('hdf',self.mcamaxno,maxlines)

        if "/main/mcadatacv" in fid:
            self.currentMCAXvalues = fid['/main/mcadatacv']
        else:
            self.currentMCAXvalues = np.arange(self.mcamaxno)*self.currentMCAXraySlope

        if not multi:
            pixno=int(pixno)
            #print 'pix:',pixno
            globalfuncs.setstatus(self.status,'Getting MCA at pixel: '+str(pixno))
            row=(pixno/self.mapdata.nxpts)
            pixno=pixno+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs        
            if pixno<0 or pixno>=maxlines:
                globalfuncs.setstatus(self.status,'EARLY End of HDF file')
                data=zeros(self.mcamaxno,dtype=np.float32)
            else:
                data=np.array(mcadata[int(pixno),:])
                data=data.astype(float)
            xtext="MCA Spectrum at pixel "+str(pixno)

        else:
            pixno=array(pixno,'int')
            pixno=np.sort(pixno)
            newpixno=[]
            for p in pixno:
                row=(p/self.mapdata.nxpts)
                p=p+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs
                newpixno.append(p)
            pixno=array(newpixno,'int')
            globalfuncs.setstatus(self.status,'Getting MCA at mutliple pixels... ')
            data=zeros(self.mcamaxno,dtype=np.float32)
            for pix in pixno:
                if pix<0 or pix>=maxlines: continue
                curmca=np.array(mcadata[pix,:])
                curmca=curmca.astype(float)               
                data=data+array(curmca)
            xtext="MCA Spectrum Area Average"
        fid.close()
        return data,xtext
 

    def killMCAplot(self):
        self.MCAviewexist=0
        self.MCAzoomstack=[]
        self.newMCAplot.destroy()        

    def updateMCAgraphscale(self,*args):
        self.updateMCAgraph()
##        if self.MCAtracescalevar.get()=='Log':
##            self.MCAgraph.yaxis_configure(logscale=1)
##        else:
##            self.MCAgraph.yaxis_configure(logscale=0)

##    def MCAcoordreport(self,event):
##        (x,y)=event.widget.invtransform(event.x,event.y)
##        xtext="X="+str(x)
##        ytext="Y="+str(y)
##        xtext=xtext[:12]
##        ytext=ytext[:12]
##        globalfuncs.setstatus(self.MCAxcoord,xtext)
##        globalfuncs.setstatus(self.MCAycoord,ytext)
    
##    def MCAzoom(self, x0, y0, x1, y1):
##        #add last to zoomstack
##        a0=self.MCAgraph.xaxis_cget("min")
##        a1=self.MCAgraph.xaxis_cget("max")
##        b0=self.MCAgraph.yaxis_cget("min")
##        b1=self.MCAgraph.yaxis_cget("max")        
##        self.MCAzoomstack.append((a0,a1,b0,b1))
##        #configure
##        self.MCAgraph.xaxis_configure(min=x0, max=x1)
##        self.MCAgraph.yaxis_configure(min=y0, max=y1)

##    def MCAunzoom(self):
##        #get last off stack
##        if self.MCAzoomstack==[]:
##            return
##        limit=self.MCAzoomstack.pop()
##        self.MCAgraph.xaxis_configure(min=limit[0],max=limit[1])
##        self.MCAgraph.yaxis_configure(min=limit[2],max=limit[3])

##    def MCAmouseDrag(self,event):
##        global x0, y0, x1, y1, druged
##        druged=1
##        (x1, y1)=self.MCAgraph.invtransform(event.x, event.y)             
##        self.MCAgraph.marker_configure("marking rectangle", 
##            coords=(x0, y0, x1, y0, x1, y1, x0, y1, x0, y0))
##        self.MCAcoordreport(event)
    
##    def MCAmouseUp(self,event):
##        global dragging, druged
##        global x0, y0, x1, y1
##        if dragging:
##            self.MCAgraph.unbind(sequence="<Motion>")
##            self.MCAgraph.bind(sequence="<Motion>", func=self.MCAcoordreport)
##            self.MCAgraph.marker_delete("marking rectangle")           
##            if event.num==1 and druged:
##                if x0 <> x1 and y0 <> y1:   
##                    # make sure the coordinates are sorted
##                    if x0 > x1: x0, x1=x1, x0
##                    if y0 > y1: y0, y1=y1, y0         
##                    self.MCAzoom(x0, y0, x1, y1) # zoom in
##            if event.num==3:
##                self.MCAunzoom() # zoom out
##                           
##    def MCAmouseDown(self,event):
##        global dragging, druged, x0, y0
##        dragging=0
##        druged=0
##        if self.MCAgraph.inside(event.x, event.y):
##            dragging=1
##            (x0, y0)=self.MCAgraph.invtransform(event.x, event.y)
##            self.MCAgraph.marker_create("line", name="marking rectangle", outline='white',dashes=(2, 2))
##            self.MCAgraph.bind(sequence="<Motion>",  func=self.MCAmouseDrag)

    def updateMCASCA(self,*args):
        if not self.E1.valid() or not self.E2.valid():
            return
        binmin=int(self.E1.getvalue())
        binmax=int(self.E2.getvalue())
        width=binmax-binmin
        center=(binmax+binmin)/2
        gmax=max(self.MCArawdata)
##        glist=self.MCAgraph.element_names()
##        if 'data' in glist:
##            self.MCAgraph.element_delete('data')
        if self.MCAgraphBarMarker is None:
            self.MCAgraphBarMarker=self.MCAgraph.graphaxes.axvspan(binmin,binmax,ymin=0,ymax=1,color='DarkOliveGreen',alpha=0.5)
            #self.MCAgraph.bar_create('data',xdata=tuple([center]),ydata=tuple([gmax]),barwidth=width,fg='DarkOliveGreen4',bg='black')
        else:
            self.MCAgraphBarMarker.remove() #self.MCAgraph.graphaxes.patches[0].remove()#self.MCAgraphBarMarker)
            self.MCAgraphBarMarker=self.MCAgraph.graphaxes.axvspan(binmin,binmax,ymin=0,ymax=1,color='DarkOliveGreen',alpha=0.5)
            #self.MCAgraphBarMarker.set_xy(((binmin,0),(binmax,1)))
        self.MCAgraph.draw()
        #self.MCAgraph.element_configure('data',label='')

    def MCAfitAlldata(self):
        self.MCAfitdata(False)
        
    def MCAfitZoomdata(self):
        self.MCAfitdata(True)



    def MCAfitdata(self,zoomfit):
        #only for HDF right now
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return

        newdata=[]
        matsize=self.mapdata.data.shape[0]*self.mapdata.data.shape[1]
        
        matindex=np.arange(matsize)
        matindexsq=matindex.reshape((self.mapdata.nypts,self.mapdata.nxpts))
        
        #go thru MCA file and integrate
        startt=time.process_time()
        (fn,ext)=os.path.splitext(self.MCAfilename)

        fid=h5py.File(self.MCAfilename)
        data=fid['/main/mcadata']
        self.mcamaxno=data.shape[1]
        maxlines=data.shape[0]
        print('hdf',self.mcamaxno,maxlines)
        
        binmin=0
        binmax=2048

#        data=np.array(mcadata[self.MCAoffset-1::self.MCAnofiles,binmin:binmax])
        ##Let's assume no offsets and no file skipping
        matindex=matindex[self.MCAoffset-1::self.MCAnofiles]
        if matsize<len(data):
            print("mca file too long? clipping...")
            matindex=matindex[:matsize]

        if getattr(sys, 'frozen', False):
            apppath=os.path.dirname(sys.executable)+os.sep
            if sys.platform=='darwin':
                if 'MacOS' in apppath: apppath=apppath.replace('MacOS','Resources')
        elif __file__:
            apppath=os.path.dirname(__file__)

        if self.pyMCAparamDialog is None and self.pyMCAlastconfig is None:
            cf=apppath+"pyMcaConfigs"+os.sep+"defaultPyMCAConfig.cfg"
        elif self.pyMCAparamDialog is None and self.pyMCAlastconfig is not None:
            cf=self.pyMCAlastconfig
        else:
            cf=apppath+"pyMcaConfigs"+os.sep+"activePyMCAConfig.cfg"
            self.pyMCAparamDialog.getGUIParameters()
            self.pyMCAparamDialog.saveConfiguration(cf)
            self.pyMCAlastconfig=cf


#        #calculate smoothing?
#        #only for "perfect" data sets -- will not be ideal for missing data sets
#        [mnum,ml]=data.shape
#        if self.pyMCAparamDialog is not None:
#            if data.shape[0]==self.mapdata.nxpts*self.mapdata.nypts:
#                print "map data and MCA data are size matched"
#                tempdata=data.reshape((self.mapdata.nypts,self.mapdata.nxpts,ml))
#                for i in range(tempdata.shape[2]):
#                    new=advancedfilters(tempdata[:,:,i],filter='Blur',size=self.pyMCAparamDialog.avgfilterVar.get(),sigma=self.pyMCAparamDialog.avgfilterBlurVar.get())
#                    tempdata[:,:,i]=new
#                data=tempdata.reshape((mnum,ml))
#            else:
#                print "map/MCA data do NOT match size"
#        else:
#            print "no dialog to calculate blurs, skipping..."

        #get the dynamic channel if needed
        dyChan=None
        if self.pyMCAparamDialog is not None and self.pyMCAparamDialog.matrixType.getvalue()=="Dynamic":
            ind=self.mapdata.labels.index(self.pyMCAparamDialog.dynamicChan.getvalue()[0])+2
            dyChan=self.mapdata.data.get(ind)
            print("dyCh rawget",dyChan.shape)
        fitShape=(self.mapdata.nypts,self.mapdata.nxpts)
          

        #auto shrink?
        if len(data)<self.mapdata.nypts*self.mapdata.nxpts and not(zoomfit and self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]):
            cury=0 #self.mapdata.nypts
            while len(data)<self.mapdata.nxpts*(self.mapdata.nypts-cury):
                cury+=1
            zoomfit=True
            self.maindisp.zmxyi[0]=0
            self.maindisp.zmxyi[1]=cury
            self.maindisp.zmxyi[2]=self.mapdata.nxpts
            self.maindisp.zmxyi[3]=self.mapdata.nypts
            self.maindisp.zmxyi[4]=0
            self.maindisp.zmxyi[5]=cury
            self.maindisp.iamzoomed=True
            
            print("shorten...",cury-1,self.maindisp.zmxyi) 
            #update display
            try:
                self.placePPMimage(self.raw)
            except:
                print("no update")
            if self.tcrefresh is not None: self.tcrefresh()     
            
        #worry about zoom
        takezoom=False
        if zoomfit and self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:  
            takezoom=True
            print("pre-zoom",np.max(matindex))
            tempdata=matindex.reshape((self.mapdata.nypts,self.mapdata.nxpts))
            #zerodata=np.zeros((self.mapdata.nypts,self.mapdata.nxpts))
            tempdata=tempdata[::-1,:]            
            tempdata=tempdata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            #zerodata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=tempdata            
            #zerodata=zerodata[::-1,:]
            print("afterzoom",np.max(tempdata))
            tdsy=(self.maindisp.zmxyi[3]-self.maindisp.zmxyi[1])
            tdsx=(self.maindisp.zmxyi[0]-self.maindisp.zmxyi[2]) 
            tds=tdsx*tdsy            
            print(tds, tdsx, tdsy, tempdata.shape)
            print(self.maindisp.zmxyi)
            fitShape=tempdata.shape
            matindex=tempdata.reshape((abs(tds),))  
            matindex=np.sort(matindex)
            print("aftersort",np.max(matindex))
#            dataToFit=np.take(data[:,binmin:binmax],matindex,axis=0)
#        else:
#            dataToFit=data[:,binmin:binmax]
            #if dyChan is not None:
            #    tempdyn=dyChan[::-1,:]            
            #    dyChan=tempdyn[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                
        if dyChan is not None:
            dyChan=ravel(dyChan)
            print("dyCh",dyChan.shape)

        target=10.0
        mindind=0

        if self.MCAXRFfitType.get()[0:4]=='Fast':
    
            #look at the size of the datasets...
            chunk=1
            chunkslice=[[(0,-1),fitShape[0]]]
            if self.MCAXRFfitType.get()[-1]=="2":
                maxsize=52428800 #204800000
                print("cur,max",self.mcamaxno*len(matindex),maxsize)
                print("curfitshape",fitShape)
                if self.mcamaxno*len(matindex)>maxsize:
                    print("need to chunk it")
                    chunkslice=[]
                    sliceline=maxsize/self.mcamaxno/fitShape[1]
                    sliceinc=sliceline*fitShape[1]
                    print("sliceinc",sliceinc)
                    chunk=0
                    sllin=0
                    while sllin<fitShape[0]*fitShape[1]:
                        chunk+=1
                        print(chunk,sllin)
                        if sllin+sliceinc>fitShape[0]*fitShape[1]:
                            chunkslice.append([(sllin-1,-1),(fitShape[0]*fitShape[1]-sllin)/fitShape[1]])
                            sllin=fitShape[0]*fitShape[1]
                        else:
                            chunkslice.append([(sllin,sllin+sliceinc),sliceline])
                            sllin=sllin+sliceinc
            print(chunk,chunkslice) 
            
            mcaWrap=pyMcaFitWrapper.Wrapper(pkm=cf)
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
                            if newname not in self.mapdata.labels:
                                valid=1
                            else:
                                i+=1
                                newname=nameroot+str(i)
                        newDataNames.append(newname)
        
#                    chdata=fresult['parameters'][ch][::-1,:]
#                    chsigma=fresult['uncertainties'][ch][::-1,:]
                    chdata=fresult._buffers['parameters'][ch][::-1,:]
                    chsigma=fresult._buffers['uncertainties'][ch][::-1,:]

                    zerodata=np.zeros((self.mapdata.nypts,self.mapdata.nxpts)) 
                    zerosigma=np.zeros((self.mapdata.nypts,self.mapdata.nxpts))                          
                    if chunk==1 and not takezoom:
                        zerodata=chdata[::-1,:] 
                        zerosigma=chsigma[::-1,:] 
                    elif chunk==1 and takezoom:
                        zerodata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=chdata                
                        zerodata=zerodata[::-1,:] 
                        zerosigma[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=chsigma               
                        zerosigma=zerosigma[::-1,:] 
                    else:
                        if takezoom:
                            areadata=zerodata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                            areasigma=zerosigma[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]                                        
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
                            zerodata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=areadata                 
                            zerosigma[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=areasigma                
                        else:
                            zerodata=areadata
                            zerosigma=areasigma
                        zerodata=zerodata[::-1,:] 
                        zerosigma=zerosigma[::-1,:] 
        
                    if forIter==0:
                        self.addchannel(zerodata,globalfuncs.fixlabelname(newname))
                        self.addchannel(zerosigma,globalfuncs.fixlabelname(newname+"-sigma"))
                    else:
                        datachan=self.mapdata.labels.index(globalfuncs.fixlabelname(newDataNames[ch]))
                        sigmachan=self.mapdata.labels.index(globalfuncs.fixlabelname(newDataNames[ch]+"-sigma"))
                        self.mapdata.data.put(datachan+2,self.mapdata.data.get(datachan+2)+zerodata)
                        self.mapdata.data.put(sigmachan+2,self.mapdata.data.get(sigmachan+2)+zerosigma)
                    
            print('data in',time.process_time()-startt)
            fid.close()
    
            return
        else:
            #this is the old fashioned fitting...
            
            for mind in matindex: #range(data.shape[0]):
    
                ##config dynamic if needed...
                if dyChan is not None:
                    self.pyMCAparamDialog.setMatrixDefinition(dyChan[mind])
                    self.pyMCAparamDialog.saveConfiguration(cf)
                    
    
                #use PyMCA wrapper class
                mcaWrap=pyMcaFitWrapper.Wrapper(pkm=cf)
    
                if mindind*100.0/len(matindex)>target:
                    print(target, end=' ')
                    target+=10.0
    
                if mind>=len(data):
                    #no more data
                    tempz=zeros(len(newdata[-1]))
                    newdata.append(tempz)
                else:
    
                    #averaging?
                    if self.pyMCAparamDialog.avgfilterVar.get()>1:
                        mi=np.where(matindexsq==mind)
                        nb=neighbors(mi[0][0],mi[1][0],self.pyMCAparamDialog.avgfilterVar.get(),self.mapdata.nypts,self.mapdata.nxpts)
                        kg=gkern(self.pyMCAparamDialog.avgfilterVar.get(),self.pyMCAparamDialog.avgfilterBlurVar.get())
                        dataToFit=zeros(data[mind,binmin:binmax].shape)
                        for nc in nb:
                            dataToFit=dataToFit+data[matindexsq[nc[0],nc[1]],binmin:binmax]*kg[nc[0]-nb[0][0],nc[1]-nb[0][1]]
                    else:
                        dataToFit=data[mind,binmin:binmax]
                                
                    mcaWrap.setData(dataToFit)
        
                    useConc=False
                    if self.pyMCAparamDialog is not None and self.pyMCAparamDialog.matrixType.getvalue()!="None":                
                        useConc=True
                    fit,cfit,flist=mcaWrap.doFit(useConc=useConc)   
                    
                    if useConc:
                        newdata.append(cfit)
                    else:
                        newdata.append(fit)
                        
                mindind+=1
                    
            sys.stdout.write('\n') 
    
            dim=len(newdata[0])
            zerosize=[]
            for j in range(dim):
                zerosize.append(0)
    
            fid.close()
    
            print('')
            print('fit',time.process_time()-startt)
    
            #data here from either data format...
            print("new array",len(newdata))
            print('matrix',matsize)
    
            if not zoomfit:
                if matsize>len(newdata): #expand
                    while len(newdata)!=matsize:
                        newdata.append(zerosize)
                        print('add one',len(newdata))
    
            newdata=array(newdata)
            print(newdata.shape)
            print(flist)
            print(newdata[0])
            print(self.mapdata.nypts,self.mapdata.nxpts,self.mapdata.nypts*self.mapdata.nxpts,matsize)
            
            if not zoomfit or self.maindisp.zmxyi[0:4]==[0,0,-1,-1]:
                if matsize<len(newdata):
                    newdata=np.reshape(newdata[:matsize,:],(self.mapdata.nypts,self.mapdata.nxpts,dim))
                else:
                    newdata=np.reshape(newdata,(self.mapdata.nypts,self.mapdata.nxpts,dim))
            else:
                nd=zeros((self.mapdata.nypts,self.mapdata.nxpts,dim),dtype=np.float32)
                temp=np.reshape(newdata,(abs(self.maindisp.zmxyi[1]-self.maindisp.zmxyi[3]),abs(self.maindisp.zmxyi[0]-self.maindisp.zmxyi[2]),dim))
                nd[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2],:]=temp
                newdata=nd[::-1,:]
    
                    
            print(newdata.shape)
            
            #place new data into main data
            for ch in range(dim):
                nameroot="fitMCA_"+str(flist[ch])
                valid=0
                i=0
                newname=nameroot
                while not valid:
                    if newname not in self.mapdata.labels:
                        valid=1
                    else:
                        i+=1
                        newname=nameroot+str(i)
                self.addchannel(newdata[:,:,ch],globalfuncs.fixlabelname(newname))
            print(time.process_time()-startt)

    def MCArebindataRange(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return        

        bottom=tkinter.simpledialog.askinteger(title='MCA Rebin Range',prompt='Enter starting bin',initialvalue=0)
        if bottom is None: return
        top=tkinter.simpledialog.askinteger(title='MCA Rebin Range',prompt='Enter ending bin',initialvalue=self.mcamaxno)
        if top is None: return
        inc=tkinter.simpledialog.askinteger(title='MCA Rebin Range',prompt='Enter bin increment',initialvalue=10)
        if inc is None: return
        bottom=int(bottom)
        top=int(top)
        inc=int(inc)
        intvals = list(range(bottom,top,inc))
        if len(intvals)<2:
            globalfuncs.setstatus(self.status,'Inappropriate range entered')
            return
        bvals=intvals.copy()
        tvals=intvals.copy()
        tvals.pop(0)
        tvals.append(top)
        for b,t in zip(bvals,tvals):
            
            noexit=1
            name=str(b)+"."+str(t)
            cind=''
            while noexit:
                chname=name+str(cind)
                if chname not in self.mapdata.labels: 
                    noexit=0
                if noexit:
                    if cind=='': cind=0
                    cind+=1
            if cind!='':cind='.'+str(cind)
            newname=name+str(cind)                        
            
            self.MCArebinfunction(b,t,newname)

    def MCArebindataValue(self):
        if self.MCAXAXIS.get()=='Bins':
            self.MCAXAXIS.set('Values')
            self.updateMCAgraph()
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        #window with start/stop and sliders and new channel name
        if self.MCAbindialogexist:
            self.MCAbindialog.show()
            self.MCAdialogupdate("Values")
            return        
        self.makeMCAdialog("Values")
        
    def MCArebindata(self):
        if self.MCAXAXIS.get()!='Bins':
            self.MCAXAXIS.set('Bins')
            self.updateMCAgraph()
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        #window with start/stop and sliders and new channel name
        if self.MCAbindialogexist:
            self.MCAbindialog.show()
            self.MCAdialogupdate("Bins")
            return
        self.makeMCAdialog("Bins")
        
    def makeMCAdialog(self,itype="Bins"):
        self.MCAbindialogexist=1
        self.MCAbindialog=Pmw.Dialog(self.imgwin,title='MCA Spectrum Rebin',buttons=('OK','Cancel'),
                                          defaultbutton='OK',command=self.MCAdotherebin)
        h=self.MCAbindialog.interior()
        h.configure(background='#d4d0c8')
        f=tkinter.Frame(h,background='#d4d0c8')
        f.pack(side=tkinter.TOP,fill='both')
        text=itype[0:-1]
        self.E1=Pmw.Counter(f,labelpos='n',label_text='Min '+text,entryfield_validate='numeric',entry_width=7,increment=5,hull_relief=tkinter.FLAT,entryfield_modifiedcommand=self.updateMCASCA,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.E2=Pmw.Counter(f,labelpos='n',label_text='Max '+text,entryfield_validate='numeric',entry_width=7,increment=5,hull_relief=tkinter.FLAT,entryfield_modifiedcommand=self.updateMCASCA,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.E1.component('uparrow').bind(sequence='<Button-1>',func=self.updateMCASCA,add='+')
        self.E1.component('downarrow').bind(sequence='<Button-1>',func=self.updateMCASCA,add='+')
        self.E2.component('uparrow').bind(sequence='<Button-1>',func=self.updateMCASCA,add='+')
        self.E2.component('downarrow').bind(sequence='<Button-1>',func=self.updateMCASCA,add='+')
        self.E1.pack(side=tkinter.LEFT,fill='both',padx=5)
        self.E2.pack(side=tkinter.LEFT,fill='both',padx=5)
        self.E1.setvalue(0)
        self.E2.setvalue(0)
        self.MCAnewchannel=Pmw.EntryField(h,labelpos='w',label_text='New Channel Name: ',entry_width=15,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.MCAnewchannel.pack(side=tkinter.TOP,fill=tkinter.X,pady=10)

    def MCAdialogupdate(self,itype):
        text=itype[0:-1]
        self.E1.component('Label').configure(text='Min '+text)
        self.E2.component('Label').configure(text='Max '+text)
        self.updateMCASCA()

    def MCAdotherebin(self,result):        
        #on enter from the dialog... 
        self.MCAbindialog.withdraw()
        if result=='Cancel' or self.MCAnewchannel.get()=='':
            print('MCA rebin cancelled')
            globalfuncs.setstatus(self.status,'MCA rebin cancelled')
            return
        #check for uniqueness
        newname=globalfuncs.fixlabelname(self.MCAnewchannel.get())
        if newname in self.mapdata.labels:
            print('Enter unique channel name')
            globalfuncs.setstatus(self.status,'Enter unique channel name')
            tkinter.messagebox.showwarning('MCA data','Please enter unique channel name')
            self.MCArebindata()
            return
        globalfuncs.setstatus(self.status,"Integrating MCA data...")
        if self.MCAXAXIS.get()=='Bins':
            binmin=int(self.E1.getvalue())
            binmax=int(self.E2.getvalue())
        else:
            vmin=int(self.E1.getvalue())
            vmax=int(self.E2.getvalue())
            binmin=globalfuncs.find_nearest(self.currentMCAXvalues,vmin)
            binmax=globalfuncs.find_nearest(self.currentMCAXvalues,vmax)
        self.MCArebinfunction(binmin,binmax,newname)
        
    def MCArebinfunction(self,binmin,binmax,newname):            
        #define a zero matrix
        newdata=[]
        matsize=self.mapdata.data.shape[0]*self.mapdata.data.shape[1]
        #go thru MCA file and integrate
        startt=time.process_time()
        (fn,ext)=os.path.splitext(self.MCAfilename)
        if ext==".bmd":
            newdata=self.getBinaryMCArebinData(matsize,binmin,binmax)
        elif ext==".hdf5":
            newdata=self.getHDF5MCArebinData(matsize,binmin,binmax)
        else:
            fid=open(self.MCAfilename,"rU")
            if self.mapdata.type=='BL62TXM':
                self.mcamaxno=int(fid.readline().split()[2])
                fid.readline()
            else:
                self.mcamaxno=2048
    ##        allMCAlines=fid.read().split('\n')
    ##        fid.close()
            linenum=0
            linetot=0
            early=0
            lineindex=0
            while linetot < matsize:
                line=fid.readline()
                try:
                    linenum=int(line.split()[0])
                    ##linenum=int(allMCAlines[lineindex].split()[0])
                except:
                    early=1
                    linenum=linenum+1
                    lineindex=lineindex+1
                    globalfuncs.setstatus(self.status,'Integrating MCA data, EOF...')
                if not early:
                    row=((linetot+1)/self.mapdata.nxpts)
                    need=(linetot)+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs
                    linov=0
                    sprs=0
    ##                try:
    ##                    line=allMCAlines[need]
    ##                except:
    ##                    line=zeros(self.mcamaxno+1,Int)
    ##                    sprs=1
                    if need<0:
                        line=zeros(self.mcamaxno+1,dtype=np.int32)
                        sprs=1
                    linetot=linetot+1
                    lineindex=lineindex+1
                    datain=[]
                    if not sprs:
                        i=0
                        for p in line.split():
                            if i!=0: datain.append(int(p))
                            i=i+1
                    else:
                        i=0
                        for p in line:
                            if i!=0: datain.append(int(p))
                            i+=1
                else:
                    datain=zeros(self.mcamaxno,dtype=np.float32)
                    linetot=linetot+1
                    lineindex=lineindex+1
                #integrate
                intdat=sum(datain[binmin:binmax+1])
                newdata.append(intdat)
            #resize newdata
            fid.close()
        #data here from either data format...
        print("new array",len(newdata),newname)

        if self.mapdata.nypts*self.mapdata.nxpts>len(newdata): #expand
            while len(newdata)!=self.mapdata.nypts*self.mapdata.nxpts:
                newdata=np.append(newdata,0)

        newdata=array(newdata)
        if self.mapdata.nypts*self.mapdata.nxpts<len(newdata):
            newdata=np.reshape(newdata[:self.mapdata.nypts*self.mapdata.nxpts-1],(self.mapdata.nypts,self.mapdata.nxpts))
        else:
            newdata=np.reshape(newdata,(self.mapdata.nypts,self.mapdata.nxpts))
        #place new data into main data
        self.addchannel(newdata,newname)
        print(time.process_time()-startt)

    def getBinaryMCArebinData(self,matsize,binmin,binmax):
        fid=open(self.MCAfilename,"rb")
        self.mcamaxno=struct.unpack('i',fid.read(4))[0]
        linenum=0
        linetot=0
        early=0
        sprs=0
        lineindex=0
        newdata=[]
        while linetot<matsize:
            try:
                linenum=int(struct.unpack("f",fid.read(4))[0])
                line=fid.read(self.mcamaxno*4)
            except:
                early=1
                linenum+=1
                globalfuncs.setstatus(self.status,'Integrating MCA data, EOF...')
            if not early:
                row=((linetot+1)/self.mapdata.nxpts)
                need=(linetot)+row*self.mapdata.nxpts*(self.MCAnofiles-1)+(self.MCAoffset-1)*self.mapdata.nxpts+self.MCApixoffs*row-self.MCA1stpixoffs
                linov=0
                sprs=0
                if need<0:
                    line=zeros(self.mcamaxno,dtype=np.float32)
                    sprs=1
                linetot+=1
                lineindex+=1
                if not sprs:
                    fmt=str(self.mcamaxno)+"f"
                    datain=struct.unpack(fmt,line)
                else:
                    datain=line
            else:
                datain=zeros(self.mcamaxno,dtype=np.float32)
                linetot+=1
                lineindex+=1
            #integrate
            intdat=sum(datain[binmin:binmax+1])
            newdata.append(intdat)
        fid.close()
        return newdata

    def getHDF5MCArebinData(self,matsize,binmin,binmax):
        #have hdf5 data...
        fid=h5py.File(self.MCAfilename)
        if "/main/mcadata" in fid:
            mcadata=fid['/main/mcadata']
        elif "/main/oodata" in fid:
            mcadata=fid['/main/oodata']
        else:
            print('no mcadata found')
            return
        self.mcamaxno=mcadata.shape[1]
        maxlines=mcadata.shape[0]
        print('hdf',self.mcamaxno,maxlines)
        
        linenum=0
        linetot=0
        early=0
        sprs=0
        lineindex=0

        newdata=zeros(self.mapdata.nxpts*self.mapdata.nypts,dtype=np.float32)
        #MCApixoffs    MCA1stpixoffs
        data=np.sum(mcadata[self.MCAoffset-1::self.MCAnofiles,binmin:binmax],axis=1)
        #data=np.sum(data,axis=1)
        data=data.astype(float)
        
        if len(newdata)<len(data):
            newdata=data[:len(newdata)-1]
        elif len(newdata)>len(data):
            newdata[:len(data)]=data
        else:
            newdata=data

        fid.close()
        return newdata


    def exportMCAdata(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for data to exist:
        if not self.MCAviewexist:
            print('No MCA data')
            globalfuncs.setstatus(self.status,'No MCA data')
            return
        globalfuncs.setstatus(self.status,"Saving MCA data to clipboard...")
        #get data for clipboard save
        self.clipboardexport(self.MCAgraph,'MCA Spectrum Data',type='MCA')
        

    def MCAwritebinary(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for datafile to exist:
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        globalfuncs.setstatus(self.status,"Parsing MCA file")
        fid=open(self.MCAfilename,'r')
        if self.mapdata.type=='BL62TXM':
            self.mcamaxno=int(fid.readline().split()[2])
            fid.readline()
        else:
            self.mcamaxno=2048
        (fn,ext)=os.path.splitext(self.MCAfilename)
        fout=open(fn+'_bindata.bmd','wb')
        fout.write(struct.pack('i',self.mcamaxno))
        linesin=fid.readlines()
        for line in linesin:
            l=line.split()
            #write in binary
            for n in l:
                fout.write(struct.pack('f',float(n)))           
        fid.close()
        fout.close()
        globalfuncs.setstatus(self.status,"Ready")                
    
    def HDF5writeHDF5(self):

        globalfuncs.setstatus(self.status,"Converting HDF5s")
        
        (fn,ext)=os.path.splitext(self.MCAfilename)

        #have hdf5 data...
        fid=h5py.File(self.MCAfilename)
        if "/main/mcadata" in fid:
            mcadata=fid['/main/mcadata']
        elif "/main/oodata" in fid:
            mcadata=fid['/main/oodata']
        else:
            print('no mcadata found')
            return
        self.mcamaxno=mcadata.shape[1]
        maxlines=mcadata.shape[0]
        print('hdf',self.mcamaxno,maxlines)

        fnout=fn+"_hdfmca.hdf5"
        fout=h5py.File(fnout,'w')
        groupout=fout.create_group("main")

        if self.HDFCOMPRESS.get()=="GZIP 4":
            outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=4)
        elif self.HDFCOMPRESS.get()=="GZIP 9":
            outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=9)
        elif self.HDFCOMPRESS.get()=="LZF":
            outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="lzf")           
        else:
            outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int')

        #copy data...

        ilines=0
        inc=1000
        done=False
        while not done:
            if ilines+inc<maxlines:
                outmcadata[ilines:ilines+inc,:]=mcadata[ilines:ilines+inc,:]
                ilines+=inc
            elif ilines==maxlines:
                outmcadata[maxlines-1,:]=mcadata[maxlines-1,:]
                done=True
            else:
                outmcadata[ilines:maxlines,:]=mcadata[ilines:maxlines,:]
                done=True
                ilines=maxlines


        fid.close()
        fout.flush()
        fout.close()
        globalfuncs.setstatus(self.status,"Ready")   
    
    
    def MCAwriteHDF5(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for datafile to exist:
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        globalfuncs.setstatus(self.status,"Parsing MCA file")

        (fn,ext)=os.path.splitext(self.MCAfilename)
        if ext==".hdf5":
            self.HDF5writeHDF5()
            return
        
        fnout=fn+"_hdfmca.hdf5"
        if ext=="bmd":
            binfile=True
            fid=open(self.MCAfilename,'rb')
            mcamaxno=struct.unpack('i',fid.read(4))[0]
        else:
            binfile=False
            fid=open(self.MCAfilename,'r')
            mcamaxno=2048


        fout=h5py.File(fnout,'w')
        groupout=fout.create_group("main")

        if self.HDFCOMPRESS.get()=="GZIP 4":
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=4)
        elif self.HDFCOMPRESS.get()=="GZIP 9":
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=9)
        elif self.HDFCOMPRESS.get()=="LZF":
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="lzf")           
        else:
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int')            
        maxhdf=100


        if not binfile:
            i=0
##            for line in linesin:
##                l=line.split()
##                #write in binary
##                d=np.array(l)
##                d=d.astype(int)
##                d=d[1:]
##                mcadata[i,:]=d
##                i+=1
            done=False
            while not done:
                try:
                    line=fid.readline().split()
                    datline=np.array(line[1:])
                except:
                    globalfuncs.setstatus(self.status,'End of MCA file')
                    done=True
                    break
                if len(datline)==0:
                    done=True
                if i==maxhdf:
                    maxhdf+=100
                    mcadata.resize(maxhdf,axis=0)
                if not done:
                    datline=datline.astype(int)
                    mcadata[i,:]=datline
                    i+=1
                    
        else:
            i=0
            done=False
            while not done:
                try:
                    linenum=int(struct.unpack('f',fid.read(4))[0])
                    datline=fid.read(mcamaxno*4)
                except:
                    globalfuncs.setstatus(self.status,'End of MCA file')
                    done=True
                    break
                if i==maxhdf:
                    maxhdf+=100
                    mcadata.resize(maxhdf,axis=0)
                if not done:
                    fmt=str(self.mcamaxno)+'f'
                    data=struct.unpack(fmt,datline)
                    data=np.array(data)
                    data=data.astype(int)
                    mcadata[i,:]=data
                    i+=1

        mcadata.resize(i,axis=0)
        fout.flush() 
        fid.close()
        fout.close()
        globalfuncs.setstatus(self.status,"Ready")     
    
    def MCAlinesplit(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for datafile to exist:
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        if self.MCAnofiles==1:
            globalfuncs.setstatus(self.status,'Only single MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please set MCA properties to multi-file first')
            return
        (fn,ext)=os.path.splitext(self.MCAfilename)
        if ext==".hdf5":
            self.HDF5linesplit()
            return
        globalfuncs.setstatus(self.status,"Parsing MCA file")
        fid=open(self.MCAfilename,'r')
        if self.mapdata.type=='BL62TXM':
            self.mcamaxno=int(fid.readline().split()[2])
            fid.readline()
        else:
            self.mcamaxno=2048
        #(fn,ext)=os.path.splitext(self.MCAfilename)
        foutlist=[]
        for a in range(self.MCAnofiles):
            temp=fn+'_split_'+str(a)+ext
            foutlist.append(open(temp,'w'))
        linenum=0
        pixnum=0
        error=0
        while not error:
            linenum=linenum+pixnum
            for fp in range(self.MCAnofiles):
                pixnum=0
                for xs in range(self.mapdata.nxpts):
                    try:
                        linein=fid.readline()
                        l=linein.split()[0]
                    except:
                        error=1
                    if not error:
                        place=0
                        for p in linein.split():
                            if place:
                                foutlist[fp].write(p+'\t')
                            else:
                                foutlist[fp].write(str(linenum+pixnum)+'\t')
                                place=1
                        foutlist[fp].write('\n')
                    else:                        
                        foutlist[fp].write(str(linenum+pixnum)+'\t')
                        for p in range(self.mcamaxno):
                            foutlist[fp].write(str(0)+'\t')
                        foutlist[fp].write('\n')
                    print(linenum,pixnum)
                    pixnum=pixnum+1
                    
        fid.close()
        for fp in foutlist:
            fp.close()
        globalfuncs.setstatus(self.status,"Ready")                

    def HDF5linesplit(self):

        globalfuncs.setstatus(self.status,"Splitting HDFs")   

        (fn,ext)=os.path.splitext(self.MCAfilename)

        #have hdf5 data...
        fid=h5py.File(self.MCAfilename)
        if "/main/mcadata" in fid:
            mcadata=fid['/main/mcadata']
        elif "/main/oodata" in fid:
            mcadata=fid['/main/oodata']
        else:
            print('no mcadata found')
            return
        self.mcamaxno=mcadata.shape[1]
        maxlines=mcadata.shape[0]
        print('hdf',self.mcamaxno,maxlines)
        

        foutlist=[]
        outmcalist=[]
        findex=[]
        for a in range(self.MCAnofiles):
            temp=fn+'_split_'+str(a)+ext
            fout=h5py.File(temp,'w')
            foutlist.append(fout)
            groupout=fout.create_group("main")
            if self.HDFCOMPRESS.get()=="GZIP 4":
                outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=4)
            elif self.HDFCOMPRESS.get()=="GZIP 9":
                outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=9)
            elif self.HDFCOMPRESS.get()=="LZF":
                outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="lzf")           
            else:
                outmcadata=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int')
            outmcalist.append(outmcadata)
            findex.append(0)

        ilines=0
        fpi=0
        inc=self.mapdata.nxpts  #should be # horizontal points...
        done=False
        while not done:
            if ilines%(10*len(findex))==0: print("line: ",ilines/len(findex))
            if ilines+inc<=maxlines:
                outmcalist[fpi][findex[fpi]:findex[fpi]+inc-1,:]=mcadata[ilines:ilines+inc-1,:]
                ilines+=inc
                findex[fpi]+=inc
                fpi+=1
                if fpi==len(findex): fpi=0
            else:
                done=True

        for j in range(len(findex)):
            outmcalist[j].resize(findex[j],axis=0)

        fid.close()
        for fp in foutlist:
            fp.flush()
            fp.close()
        globalfuncs.setstatus(self.status,"Ready")   

    def MCAdeglitch(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for datafile to exist:
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        #choose monitor channel
        self.MCAdeglitchdialog=Pmw.SelectionDialog(self.imgwin,title='Choose Monitor Channel',buttons=('OK','Cancel'),
                                                   defaultbutton='OK',command=self.MCAdodeglitch,scrolledlist_labelpos='n',
                                                   label_text='Channel',scrolledlist_items=self.mapdata.labels)
        self.MCAdeglitchdialog.show()
        
    def MCAdodeglitch(self,result):
        sels=self.MCAdeglitchdialog.getcurselection()       
        self.MCAdeglitchdialog.withdraw()
        if result=='Cancel':
            print('MCA correction cancelled')
            globalfuncs.setstatus(self.status,'MCA correction cancelled')
            return
        if len(sels)==0:
            print('No monitor channel selected')
            globalfuncs.setstatus(self.status,'No monitor channel selected')
            return
        
        monchan=self.mapdata.labels.index(sels[0])
        mondata=ravel(self.mapdata.data.get(monchan+2))#[:,:,monchan+2])
        if self.mapdata.labels.count('I0STRM'):
            i0chan=self.mapdata.labels.index('I0STRM')
        elif self.mapdata.labels.count('I0'):
            i0chan=self.mapdata.labels.index('I0')
        else:
            i0chan=0
        i0data=ravel(self.mapdata.data.get(i0chan+2))#[:,:,i0chan+2])
        if not i0chan:
            i0data=zeros(i0data.size,dtype=np.int32)

        globalfuncs.setstatus(self.status,"Parsing MCA file")
        fid=open(self.MCAfilename,'r')
        if self.mapdata.type=='BL62TXM':
            self.mcamaxno=int(fid.readline().split()[2])
            fid.readline()
        else:
            self.mcamaxno=2048
        (fn,ext)=os.path.splitext(self.MCAfilename)
        fn=fn+'_deg'+ext
        fout=open(fn,'w')
        total=0
        for i in range(len(mondata)):
            total=total+1
            if int(mondata[i])!=0:
                linein=fid.readline()
                place=0
                for p in linein.split():
                    if place:
                        fout.write(p+'\t')
                    else:
                        fout.write(str(total)+'\t')
                        place=1
                fout.write('\n')
            else:
                if int(i0data[i])>250:
                    fout.write(str(total)+'\t')
                    for p in range(self.mcamaxno):
                        fout.write(str(0)+'\t')
                    fout.write('\n')
                else:
                    linein=fid.readline()
                    place=0
                    for p in linein.split():
                        if place:
                            fout.write(p+'\t')
                        else:
                            fout.write(str(total)+'\t')
                            place=1
                    fout.write('\n')                    
                
        fid.close()
        fout.close()
        globalfuncs.setstatus(self.status,"Ready")


    def HDFtoMCA(self):
        #get filebase
        fn=globalfuncs.ask_for_file([("HDF files","*.hdf"),("NXS data files","*.nxs"),("H5 data files","*.h5"),("SUPER files","*.*G")],self.filedir.get())
        #validate and determine SUPER vs NSLS
        if os.path.splitext(fn)[1][-1]=='G':
            print('SUPER HDF')
            self.SUPERGHDF_MCA(fn)
        else:
            print('HDF')
            self.NSLSHDF_MCA(fn)

    def SUPERGHDF_MCA(self,fn):
        base=os.path.splitext(fn)[0]
        start=int(os.path.splitext(fn)[1][1:-1])
        #open G file
        fid=open(fn,'r')
        inlines=fid.read().split('\n')
        fid.close()
        active=0
        globalfuncs.setstatus(self.status,'Translating HDFs')
        mcaoutfn=base+'_'+start.zfill(3)+'.mca'
        lnum=1
        fidmcaout=open(mcaoutfn,'w')
        fidmcaout.write(str(lnum)+'\t')
        for l in inlines:
            if len(l)<1: continue
            if l.split()[0]=='#L':
                active=1
                continue
            if active:
                #do file
                hdffn=base+start.zfill(3)+'.hdf'
                outfn=os.path.dirname(fn)+'/temp.txt'
                os.system('hdp dumpsds -d -o '+outfn+' '+hdffn)
                #parse
                fidin=open(outfn,'r')
                hdflines=fidin.read().split('\n')
                fidin.close()
                fml=0
                for hl in hdflines:
                    if len(hl)<1 and not fml:
                        lnum=lnum+1
                        fidmcaout.write('\n')
                        fml=1
                        continue
                    if len(hl)>1 and fml:
                        fidmcaout.write(str(lnum)+'\t')
                        fml=0
                        continue
                    if len(hl)>1:
                        fidmcaout.write(hl)
                #increment
                start=start+1
        
        fidmcaout.close()
        os.remove(os.path.dirname(fn)+'/temp.txt')
        globalfuncs.setstatus(self.status,'Ready')

    def NSLSHDF_MCA(self,fn):
        globalfuncs.setstatus(self.status,'Translating HDFs')
        
        outfn=os.path.dirname(fn)+'/temp.txt'
        os.system("""hdp dumpsds -d -n "mca data" -o """+outfn+' '+fn)
        #parse
        fidin=open(outfn,'r')
        hdflines=fidin.read().split('\n')
        fidin.close()
        dat=[]
        for hf in hdflines:
            for d in hf.split():
                dat.append(int(d))
        dat=array(dat)
        nums=dat.shape[0]/2048
        dat=np.reshape(dat,(2048,nums))
        #print dat[:,0]
        #write new mca
        fidout=open(os.path.splitext(fn)[0]+'.mca','w')
        for i in range(nums):
            fidout.write(str(i+1)+'\t')
            for j in dat[:,i]:
                fidout.write(str(j)+'\t')
            fidout.write('\n')
        fidout.close()
        #done
        os.remove(os.path.dirname(fn)+'/temp.txt')
        globalfuncs.setstatus(self.status,'Ready')


################################ MCA Movie:

    def makeMCAmovie(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for datafile to exist:
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('MCA data','Please define file with MCA data first')
            return
        #should ask parameters here...
        bottom=1000
        top=14000
        step=100
        bottom=tkinter.simpledialog.askinteger(title='MCA Movie Parameter',prompt='Enter energy minimum (eV)',initialvalue=bottom)
        if bottom is None: return
        top=tkinter.simpledialog.askinteger(title='MCA Movie Parameter',prompt='Enter energy maximum (eV)',initialvalue=top)
        if top is None: return
        step=tkinter.simpledialog.askinteger(title='MCA Movie Parameter',prompt='Enter energy step (eV)',initialvalue=step)
        if step is None: return
        bottom=int(bottom/10.)
        top=int(top/10.)
        step=int(step/10.)
        print(bottom,top,step)

        if os.path.splitext(self.MCAfilename)[1]==".hdf5":
            self.makeMCAHDFmovie(bottom,top,step)
            return


        if not self.PCAdataLoaded:        
            self.loadMCAforPCA(justload=1)
            self.PCAdataLoaded=1
        globalfuncs.setstatus(self.status,'Processing MCA')
        imgs=[]
        frames=[]
        print(self.PCArawdata.shape)
        for pos in range(bottom,top,step):
            n=np.sum(self.PCArawdata[:,pos:pos+step-1],axis=1)
            n=np.reshape(n,(self.mapdata.data.get(0).shape))
            imgs.append(transpose(n[::-1,:]))
            frames.append(pos)
        imgs=array(imgs)
        mi=self.mapdata.mapindex[::-1,:]
        print(imgs.shape)
        if self.movieView is None:
            self.movieView=Display.Movie(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,main=0)        
        self.movieView.placeMovie(imgs,transpose(mi),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,datlab='MOVIE',pause=0.2,frames=frames)            

    def makeMCAHDFmovie(self,bottom,top,step):

        fid=h5py.File(self.MCAfilename)
        if "/main/mcadata" in fid:
            mcadata=fid['/main/mcadata']
        elif "/main/oodata" in fid:
            mcadata=fid['/main/oodata']
        else:
            print('no mcadata found')
            return
        self.mcamaxno=mcadata.shape[1]
        maxlines=mcadata.shape[0]
        print('hdf',self.mcamaxno,maxlines)

        imgs=[]
        frames=[]

        for pos in range(bottom,top,step):
            n=np.sum(mcadata[:,pos:pos+step-1],axis=1)
            n=np.reshape(n,(self.mapdata.data.get(0).shape))
            imgs.append(transpose(n[::-1,:]))
            frames.append(pos)
        imgs=array(imgs)
        mi=self.mapdata.mapindex[::-1,:]
        print(imgs.shape)
        if self.movieView is None:
            self.movieView=Display.Movie(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,main=0)        
        self.movieView.placeMovie(imgs,transpose(mi),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,datlab='MOVIE',pause=0.2,frames=frames)            

        fid.close()

    def swapdirXspressHDF(self):
        #get file...
        fty=[("HDF5 MCA data files","*.hdf5"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())
        if t!='':
            (fn,ext)=os.path.splitext(t)
            if ext.lower()!=".hdf5":
                globalfuncs.setstatus(self.status,"Improper HDF5 file")
                return 0
            nfn=fn+'_rev'            
        else:
            globalfuncs.setstatus(self.status,"No HDF MCA data defined")
            return 0
        #open hdf as test and get # of channels...
        #data in: /entry/instrument/detector/data
        # data is npoints in row x nchannels x 4096 (2048)
        fid=h5py.File(fn+ext)
        mcadata=fid['/main/mcadata']
        maxlines=mcadata.shape[0]        
        
        fout=h5py.File(nfn+ext,'w')
        groupout=fout.create_group("main")

        if self.HDFCOMPRESS.get()=="GZIP 4":
            mcadataout=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=4)
        elif self.HDFCOMPRESS.get()=="GZIP 9":
            mcadataout=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=9)
        elif self.HDFCOMPRESS.get()=="LZF":
            mcadataout=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="lzf")           
        else:
            mcadataout=groupout.create_dataset("mcadata",(maxlines,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int')            
        
        globalfuncs.setstatus(self.status,"Reading HDF files... ...")
        print("Reading HDF files... ...")

        mcadataout=mcadata.copy()
        mcadataout[1::2,]=mcadataout[1::2,::-1]

        fid.close()
        fout.flush() 
        fout.close()
        globalfuncs.setstatus(self.status,"Ready")          
            
    def constructFullXspressHDF(self):
        #get file...
        fty=[("HDF5 MCA data files","*.hdf5"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())
        if t!='':
            (fn,ext)=os.path.splitext(t)
            if ext.lower()!=".hdf5":
                globalfuncs.setstatus(self.status,"Improper HDF5 file")
                return 0
            fnsplit=fn.split('_')
            fnbase=''
            for i in range(len(fnsplit)-1):
                fnbase+=fnsplit[i]+"_"
            fnbase=fnbase[:-1]
            print(fnbase)
        else:
            globalfuncs.setstatus(self.status,"No HDF MCA data defined")
            return 0
        #open hdf as test and get # of channels...
        #data in: /entry/instrument/detector/data
        # data is npoints in row x nchannels x 4096 (2048)
        fid=h5py.File(fn+ext)
        try: 
            mcadata=fid['/entry/instrument/detector/data']
            typ='x'
        except:
            mcadata=fid['/main/mcadata']
            typ='oo'
        print('hdf',mcadata.shape)
        if typ=='oo': nchan=1
        else: nchan=mcadata.shape[1]
        fid.close()

        if nchan==1:
            hdfInfo=HDFConstruct(fnbase,list(range(nchan)))
            self.createXspressHDF(hdfInfo,typ=typ)
        else:
            self.tempfnbase=fnbase
            self.xspressHDFChanSelect=Pmw.Dialog(self.imgwin,title="Channel Select",buttons=('OK','Cancel'),defaultbutton='OK',
                                                   command=self.checkXspressChan)
            intex=self.xspressHDFChanSelect.interior()
            intex.configure(background='#d4d0c8')
            self.xspressChanSelect=PmwTtkRadioSelect.PmwTtkRadioSelect(intex,buttontype='checkbutton',orient='vertical',selectmode='multiple',hull_background='#d4d0c8')
            l=[]
            for text in range(nchan):
                self.xspressChanSelect.add("Channel "+str(text+1))
                l.append("Channel "+str(text+1))
                self.xspressChanSelect.button(text).config(background='#d4d0c8')
            self.xspressChanSelect.setvalue(l)
            self.xspressChanSelect.pack(side=tkinter.TOP,padx=3,pady=3)
            self.xspressHDFChanSelect.show()
            

    def checkXspressChan(self,result):
        if result=='Cancel':
            print("Xspress import cancelled")
            globalfuncs.setstatus(self.status,"Xspress import cancelled")
            self.xspressHDFChanSelect.withdraw()
            return
        else:
            indicies=[]
            for b in self.xspressChanSelect.getvalue():
                indicies.append(int(b.split()[1])-1)
            self.xspressHDFChanSelect.withdraw()

            hdfInfo=HDFConstruct(self.tempfnbase,indicies)
            self.createXspressHDF(hdfInfo)
            

    def createXspressHDF(self,hdfInfo,typ='x'):

        bidir=tkinter.messagebox.askyesno("Scan Direction", "Was this a BIDIRECTIONAL scan?")
        dir=1
        baseline=0
        #create new HDF
        fout=h5py.File(hdfInfo.fnbase+"_MCA.hdf5",'w')
        groupout=fout.create_group("main")

        if self.HDFCOMPRESS.get()=="GZIP 4":
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=4)
        elif self.HDFCOMPRESS.get()=="GZIP 9":
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="gzip",compression_opts=9)
        elif self.HDFCOMPRESS.get()=="LZF":
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int',compression="lzf")           
        else:
            mcadata=groupout.create_dataset("mcadata",(100,self.mcamaxno),maxshape=(None,self.mcamaxno),dtype='int')            
        maxhdf=100

        globalfuncs.setstatus(self.status,"Reading HDF files... ...")
        print("Reading HDF files... ...")
        
        i=0
        ilines=0
        done=False
        while not done:
            prc=True
            if (i+1)%10==0: print("file: ",i+1)
            if typ=='x': 
                if not os.path.exists(hdfInfo.fnbase+"_"+str(i+1)+".hdf5"):
                    done=True
            elif typ=="oox":
                print(hdfInfo.fnbase+"_"+str(i+1).zfill(3)+"_uv_00001.hdf5")
                if not os.path.exists(hdfInfo.fnbase+"_"+str(i+1).zfill(3)+"_uv_00001.hdf5"):
                    done=True
            else:
                if not os.path.exists(hdfInfo.fnbase+"_"+str(i+1).zfill(5)+".hdf5"):
                    done=True
            try:
                
                if typ=='x':
                    fid=h5py.File(hdfInfo.fnbase+"_"+str(i+1)+".hdf5","r")
                    mcadatain=fid['/entry/instrument/detector/data']
                    dslice=np.array(mcadatain[:,:,:2048])
                elif typ=='oox':
                    fid=h5py.File(hdfInfo.fnbase+"_"+str(i+1).zfill(3)+"_uv_00001.hdf5","r")
                    mcadatain=fid['/main/mcadata']
                    dslice=np.array(mcadatain)                    
                else: 
                    fid=h5py.File(hdfInfo.fnbase+"_"+str(i+1).zfill(5)+".hdf5","r")
                    mcadatain=fid['/main/mcadata']
                    dslice=np.array(mcadatain)
            except:
                print("error in reading data in file: ",i+1)
                if i>0: done=True
                else: prc=False 
            if not done and prc:                
                dslice=dslice.astype("int")
                #sum
                if typ=='x': datline=dslice[:,hdfInfo.chans,:].sum(axis=1)
                else: datline=dslice[:,:]
                nps=datline.shape[0]
                if baseline==0: baseline=nps
                
                fid.close()
                                              
                if ilines+baseline>maxhdf:
                    maxhdf=ilines+baseline
                    mcadata.resize((maxhdf,self.mcamaxno))
                datline=datline.astype(int)

                #print i,dir,len(datline),np.where(np.sum(datline,axis=1)==max(np.sum(datline,axis=1)))

                if nps>baseline: #need to pad
                    print('long',i,baseline,nps,datline.shape)
                    delta=nps-baseline
                    if ilines+nps+delta>maxhdf:
                        maxhdf=ilines+nps+delta
                        mcadata.resize(maxhdf,axis=0)
                    depth=mcadata.shape[1]
                    pad=np.zeros((delta,depth))
                    pad=pad.astype(int)

                    mcadata[ilines:ilines+nps,:]=datline[::dir]
                    ilines+=nps
                    mcadata[delta:,:]=mcadata[0:ilines,:]
                    mcadata[0:delta,:]=pad
                    ilines+=delta
                    ##mcadata=np.insert(mcadata,0,pad).reshape([ilines+nps,depth])                    
                    baseline=nps
                elif nps<baseline: #need to pad
                    delta=baseline - nps
                    depth=mcadata.shape[1]
                    pad=np.zeros((baseline,depth))
                    pad=pad.astype(int)
                    
                    print('short',i,baseline,nps,pad.shape,datline.shape)
                    if 0==0:
                        pad[delta:,:]=datline[:,:]
                    else:
                        pad[0:nps,:]=datline[:,:]

                    mcadata[ilines:ilines+baseline,:]=pad[::dir]
                    ilines+=baseline
                else:
                    mcadata[ilines:ilines+nps,:]=datline[::dir]
                    ilines+=nps                    

                if bidir: dir=dir*-1
                
            i+=1

        mcadata.resize((ilines,self.mcamaxno))#,axis=0)
        fout.flush() 
        fout.close()
        globalfuncs.setstatus(self.status,"Ready")     


    def constructXASXspressHDF(self):
        #get file...
        fty=[("HDF5 MCA data files","*.hdf5"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())
        if t!='':
            (fn,ext)=os.path.splitext(t)
            if ext.lower()!=".hdf5":
                globalfuncs.setstatus(self.status,"Improper HDF5 file")
                return 0
            if fn[-9:]!="_uv_00001":
                globalfuncs.setstatus(self.status,"Improper OOXAS HDF5 file")
                return 0                
            fnsplit=fn.split('_uv_00001')[0].split("_")
            fnbase=''
            for i in range(len(fnsplit)-1):
                fnbase+=fnsplit[i]+"_"
            fnbase=fnbase[:-1]
            print(fnbase)
        else:
            globalfuncs.setstatus(self.status,"No HDF MCA data defined")
            return 0
        #open hdf as test and get # of channels...
        #data in: /entry/instrument/detector/data
        # data is npoints in row x nchannels x 4096 (2048)
        fid=h5py.File(fn+ext)
        try: 
            mcadata=fid['/entry/instrument/detector/data']
            typ='x'
        except:
            mcadata=fid['/main/mcadata']
            typ='oox'
        print('hdf',mcadata.shape)
        if typ=='oox': nchan=1
        else: nchan=mcadata.shape[1]
        fid.close()

        if nchan==1:
            hdfInfo=HDFConstruct(fnbase,list(range(nchan)))
            self.createXspressHDF(hdfInfo,typ=typ)
        else:
            print("something went wrong with XAS UV data")


#################################  FTIR Analysis



    def IRprocessingWindow(self):
        if not self.IRwinexist:

            self.IRwinexist=1
            self.IRwin=Pmw.MegaToplevel(self.imgwin)
            self.IRwin.title('IR Processing View')
            self.IRwin.userdeletefunc(func=self.killIRwin)           
            h=self.IRwin.interior()
            h.configure(background='#d4d0c8')

            ls=tkinter.Frame(h, background='#d4d0c8')
            ls.pack(side=tkinter.TOP,fill='both')
            
            g2=tkinter.Frame(ls, relief=tkinter.GROOVE,bd=2, background='#d4d0c8')
            g2.pack(side=tkinter.LEFT,padx=2,pady=2, fill=tkinter.BOTH, expand=1)
            l=tkinter.Label(g2,text="Processing Params",relief=tkinter.RAISED,bd=2, background='#d4d0c8')
            l.pack(side=tkinter.TOP,fill=tkinter.X,padx=2,pady=4)
            w=15

            bb=PmwTtkButtonBox.PmwTtkButtonBox(g2,labelpos='n',label_text='Fit Actions:',orient='vertical',pady=3,padx=5,frame_background='#d4d0c8',hull_background='#d4d0c8',label_background='#d4d0c8')
            bb.add('View Config',command=self.openMCAfitConfig,style='SBLUE.TButton',width=w)            
            bb.add('Load Config',command=self.loadMCAfitConfig,style='NAVY.TButton',width=w)
            bb.add('Save Config',command=self.saveMCAfitConfig,style='BROWN.TButton',width=w)
            bb.add('Fit Current',command=self.testMCAfitdata,style='LGREEN.TButton',width=w)
            bb.add('Fit Zoom MCA',command=self.MCAfitZoomdata,style='OGREEN.TButton',width=w)
            bb.add('Fit All MCA',command=self.MCAfitAlldata,style='GREEN.TButton',width=w)
            bb.pack(side=tkinter.TOP,fill='both',padx=2,pady=5)
    
            self.MCAgraph=MyGraph.MyGraph(ls,whsize=(6.5,3),tool=1,graphpos=[[.15,.1],[.9,.9]],side=tkinter.LEFT)
            
            





    def updateIRgraph(self):
        pass

    def killIRwin(self):
        self.IRwinexist=0
        self.IRwin.destroy()       

    def IRprocessAll(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for datafile to exist:
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No IR-MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('IR data','Please define "MCA" file with IR data first')
            return



    def saveIRsettings(self):
        if self.FTIRgroup is None:
            print ("No parameters to save...")
            globalfuncs.setstatus(self.status,'No parameters to save...')
        #get file name to save
        fn="IR_processing_params.json"
        fn=globalfuncs.ask_save_file(fn,self.ps.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return   
        if os.path.splitext(fn)[1]!='.json':
            fn=fn+".json"
        
        self.FTIRgroup.save(fn)
        print('save complete')
        globalfuncs.setstatus(self.status,'IR parametere save complete')
    
    def loadIRsettings(self):
    
        infile=globalfuncs.ask_for_file([("JSON IR paramater files","*.json"),("all files","*")],self.ps.filedir.get(),multi=False)
        if infile == '' or infile is None:
            print ('canceled')
            globalfuncs.setstatus(self.status,"No calibration file defined...")
            return        
        #load file
        self.FTIRgroup = IR_MathClass.IRParamClass()
        self.dbFTIRgroup.load(infile)       
        print('load complete')
        globalfuncs.setstatus(self.status,'IR parametere load complete')        
        
#################################  Axis averaging routine

    def datacompressplot(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #ask for axis to average
        ps = DataAveragingClass.DataAveragingParams(self.status, self.datachan, self.mask, self.maindisp, self.usemaskinimage)
        self.dataAxisAveragingDialog = DataAveragingClass.DataAveraging(self.imgwin, self.mapdata, ps)
    
    def datacontoursToggle(self):
        #check to see if dataSummaryWindow exists... if so update.
        #JOY Seems wrong...
        if not self.hasdata:
            return
        if not self.dataSummaryWindow.exist:
            return
        else:
            self.dataSummaryWindow.doDataSummary(self.contoursOn)

    def rowBalanceCalc(self):
        #get current channel
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        data=self.mapdata.data.get(datind)[::-1,:]#[::-1,:,datind]
        pm=ones(self.mapdata.data.get(0)[::-1,:].shape)
        dt=data*pm
        #average data
        print(data.shape)

        pix=np.sum(pm,axis=1)
        tdat=np.sum(dt,axis=1)#/data.shape[1]
        adat=np.where(pix==0,0,tdat/pix)

        nv=sum(adat)/len(adat)
        self.rat=[]

        for i in range(len(adat)):
            if adat[i]!=0:
                self.rat.append(nv/adat[i])
            else:
                self.rat.append(1)

        self.rowBalance()

    def rowBalance(self):
        if self.rat is None:
            print('no ratio channel calculated')
            globalfuncs.setstatus(self.status,"no ratio channel calculated.")
            return
        #get current channel
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        data=self.mapdata.data.get(datind)[::-1,:]#[::-1,:,datind]
        pm=ones(self.mapdata.data.get(0)[::-1,:].shape)
        dt=data*pm

        for i in range(len(self.rat)):
            dt[i,:]=dt[i,:]*self.rat[i]
            
        self.addchannel(dt[::-1,:],self.datachan.getvalue()[0]+"bal")

    def datacompresssummary(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        globalfuncs.setstatus(self.status,"AVERAGING...")
        if not self.dataSummaryWindow.exist:
            ps = DataSummaryClass.DataSummaryParams(self.maindisp, self.mask, self.usemaskinimage, self.contoursOn, self.status)
            self.dataSummaryWindow.create(self.mapdata, ps)
        else:
            self.dataSummaryWindow.doDataSummary(self.contoursOn)


    
#################################  Spectrum Maker

    def startspectrumMaker(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.spectrumdialog=Pmw.SelectionDialog(self.imgwin,title="Select Spectrum Channels",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.spectrumMakernext)
        self.spectrumdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        
    def spectrumMakernext(self,result):
        chans=self.spectrumdialog.getcurselection()
        self.spectrumdialog.withdraw()
        if result=='Cancel':
            self.spectrumSwitch=0
            globalfuncs.setstatus(self.status,'No spectrum plotter OFF')
            return
        self.spectrumSwitch=1
        #set x-axis values
        self.specX=[]
        self.specC=[]
        if tkinter.messagebox.askyesno("Accept Default Values", "e.g. first value="+str(ChannelInterpret.value(chans[0]))):            
            for c in chans:
                self.specC.append(c)
                self.specX.append(ChannelInterpret.value(c))
        elif tkinter.messagebox.askyesno("Enter Range", "Enter a constant energy step range?"):  
            #get start value
            ev=tkinter.simpledialog.askfloat(title='Spectrum Maker',prompt='Enter the value for initial channel on x-axis',initialvalue=0)
            if ev=='':
                self.spectrumSwitch=0
                return
            #get step
            evstep=tkinter.simpledialog.askfloat(title='Spectrum Maker',prompt='Enter the step value of the x-axis',initialvalue=1)
            if evstep=='':
                self.spectrumSwitch=0
                return
            for c in chans:
                self.specC.append(c)
                self.specX.append(ev)
                ev+=evstep            
        else:
            for c in chans:
                self.specC.append(c)
                t=tkinter.simpledialog.askfloat(title='Spectrum Maker',prompt='Enter value for channel on x-axis',initialvalue=ChannelInterpret.value(c))
                if t=='':
                    self.spectrumSwitch=0
                    return
                self.specX.append(t)


    def displaySpectrumGraph(self,pixno):
        if not self.spectrumSwitch: return
        print(pixno)
        yv=[]
        for c in self.specC:
            datind=self.mapdata.labels.index(c)+2
            yv.append(ravel(self.mapdata.data.get(datind))[pixno])
        #do plot
        #define new window if needed
        if not self.maindisp.linegraph2present:
            self.maindisp.linegraph2present=1
            self.maindisp.newlineplot2=Pmw.MegaToplevel(self.maindisp.master)
            self.maindisp.newlineplot2.title('Spectrum Plot View')
            self.maindisp.newlineplot2.userdeletefunc(func=self.maindisp.killlineplot2)           
            h=self.maindisp.newlineplot2.interior()
            self.maindisp.graphx2=MyGraph.MyGraph(h,whsize=(4.5,4),graphpos=[[.15,.1],[.9,.9]])
            #self.graphx2.legend_configure(hide=1)
            #self.graphx2.pack(side=tkinter.LEFT,expand=1,fill='both',padx=2)
        else:
            #clear old
            if self.maindisp.toggleAddPlotVAR.get()==0:
                self.maindisp.newlineplot2.title('Spectrum Plot View')
                self.maindisp.graphx2.cleargraphs()
##                    for gtype in (self.graphx2,):
##                        glist=gtype.element_names()
##                        if glist !=():
##                            for g in glist:
##                                gtype.element_delete(g)            
        #order data
        sord=np.argsort(np.array(self.specX))
        xv=np.array(self.specX)[sord]
        syv=np.array(yv)[sord]
        #make graphs
        palette=sblite.color_palette('hls', n_colors=8)
        colors=palette.as_hex()  #['blue', 'red', 'green', 'white', 'orange','magenta', 'cyan','brown']
        colorplot=colors[self.maindisp.xyplotcolorind%len(colors)]
        #self.graphx2.configure(title=xtext)
        #self.graphx2.line_create('XV',xdata=tuple(xv),ydata=tuple(yv),symbol='',color='green')        
        self.maindisp.graphx2.plot(tuple(xv),tuple(syv),color=colorplot,text='XV'+str(self.maindisp.xyplotcolorind))
        self.maindisp.graphx2.draw()
        self.maindisp.newlineplot2.show()

    def exportFFPCASpectra(self):
        if self.PCAlastprop is None or self.PCAlastevect is None:
            print('Do PCA first')
            globalfuncs.setstatus(self.status,'Do PCA first')
            return
        maxN=tkinter.simpledialog.askinteger(title='Export Spectra ',prompt='Enter number of PCA components to use:',initialvalue=self.PCAcompMAXNO)
        if maxN<1: maxN=1
        if maxN>self.PCAcompMAXNO: maxN=self.PCAcompMAXNO
        self.PCAspectra=np.dot(self.PCAlastprop[:,0:maxN],self.PCAlastevect[0:maxN,:])
        print(self.PCAspectra.shape)
        self.exportFFSpectra(PCA=True)
    
    def exportFFSpectra(self,PCA=False):
        self.exportFFPCAflag=PCA
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.spectrumSwitch: 
            print('No Spectra Data Defined')
            globalfuncs.setstatus(self.status,'No Spectrum Data Defined')
            return
        #stolen from MCA fits...
        self.exportFFSpecCDT={}
        curch=self.datachan.getvalue()[0]
        curind=self.mapdata.labels.index(curch)
        u=1
        m=max(ravel(self.mapdata.data.get(curind+2)))+1
        b=np.mod(ravel(self.mapdata.data.get(curind+2)),1)
        if m>31: u=0
        if sum(abs(b))>0: u=0
        if len(np.where(self.mapdata.data.get(curind+2)<0)[0])>1: u=0
        if u==0:
            if not tkinter.messagebox.askokcancel('Spectrum Export','Current selected data does not appear to be a mask set.  Continue?'):
                globalfuncs.setstatus(self.status,'Spectrum export cancelled')
                return            
        self.exportFFSpecCDT[self.mapdata.labels[curind]]=int(m)
        self.exportFFSpectraNext('Ok',curch)
        #print "inspecting data set"
        #globalfuncs.setstatus(self.status,'inspecting data set')
        #self.exportFFSpecCDT={}
        #i=0
        #for i in range(len(self.mapdata.labels)):
        #    u=1
        #    m=max(ravel(self.mapdata.data.get(i+2)))+1
        #    b=np.mod(ravel(self.mapdata.data.get(i+2)),1)
        #    if m>31: u=0
        #    if sum(abs(b))>0: u=0
        #    if len(np.where(self.mapdata.data.get(i+2)<0)[0])>1: u=0
        #    if u==1: 
        #        self.exportFFSpecCDT[self.mapdata.labels[i]]=int(m)
        
        #self.xportspectrumdialog=Pmw.SelectionDialog(self.imgwin,title="Select Spectrum Export Master",buttons=('OK','Cancel'),defaultbutton='Cancel',
        #                                           scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.exportFFSpecCDT.keys(),
        #                                           command=self.exportFFSpectraNext)
        
    def exportFFSpectraNext(self,result,maskchan):
        #maskchan=self.xportspectrumdialog.getcurselection()[0]
        #self.xportspectrumdialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'Spectrum export cancelled')
            return
        maskvalues=self.exportFFSpecCDT[maskchan]   
        maskind=self.mapdata.labels.index(maskchan)+2     
        maskbase=ravel(self.mapdata.data.get(maskind))
        #order data
        sord=np.argsort(np.array(self.specX))
        xv=np.array(self.specX)[sord]
        print("processing data set")
        globalfuncs.setstatus(self.status,'processing data set')
        syv=[]
        exp=[]
        syv.append(xv)
        for s in range(maskvalues):
            print("calculating ",s)
            yv=[]
            mask=np.where(maskbase==s,1,0)
            nvals=np.sum(mask)
            if nvals==0:
                nvals=1.0
            else:
                nvals=float(nvals)
            if self.exportFFPCAflag:
                for c in range(len(self.specC)):
                    maskedspec=self.PCAspectra[:,c]*mask
                    yv.append(sum(maskedspec)/nvals)               
            else:
                for c in self.specC:
                    datind=self.mapdata.labels.index(c)+2
                    maskedspec=ravel(self.mapdata.data.get(datind))*mask
                    yv.append(sum(maskedspec)/nvals)
            syv.append(np.array(yv)[sord])   
            exp.append(np.array(yv)[sord]) 
        syv=np.array(syv)
        syv=transpose(syv)
        print("prcoessing complete")
        globalfuncs.setstatus(self.status,'processing complete')
        #whew
            
        if self.exportFFPCAflag:
            fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_spectraPCAExport.dat'
        else:
            fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_spectraExport.dat'
        fn=(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving export spectra data...")
        #save data
        np.savetxt(fn,syv)
        for s in range(maskvalues):
            nfn=os.path.splitext(fn)[0]+"_sp"+str(s)+".dat"
            cspecdata=[]
            cspecdata.append(xv)
            cspecdata.append(exp[s])
            cspecdata=np.array(cspecdata)
            cspecdata=transpose(cspecdata)
            np.savetxt(nfn,cspecdata)
        globalfuncs.setstatus(self.status,"Spectrum export saved in: "+fn)
        
        if self.PCAplotVectors.get():
            if not self.maindisp.linegraph2present:
                self.maindisp.linegraph2present=1
                self.maindisp.newlineplot2=Pmw.MegaToplevel(self.maindisp.master)
                self.maindisp.newlineplot2.title('Spectra Plot View')
                self.maindisp.newlineplot2.userdeletefunc(func=self.maindisp.killlineplot2)           
                h=self.maindisp.newlineplot2.interior()
                self.maindisp.graphx2=MyGraph.MyGraph(h,whsize=(4.5,4),graphpos=[[.15,.1],[.9,.9]])
            else:
                #clear old
                self.maindisp.newlineplot2.title('Line Plot View')
                self.maindisp.graphx2.cleargraphs()
            #make graphs
            palette=sblite.color_palette('hls',n_colors=8)
            colors=palette.as_hex() #['blue', 'red', 'green', 'white', 'orange','magenta', 'cyan','brown']
            for i in range(maskvalues):
                yv=exp[i]
                self.maindisp.graphx2.plot(tuple(xv),tuple(yv),color=colors[i%len(colors)],text='EV'+str(i))
            self.maindisp.graphx2.draw()
            self.maindisp.newlineplot2.show()
            
        
        
    def startFFMaker(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #ask for type of I0 normalization
        self.FFreferenceDialog=Pmw.MessageDialog(self.imgwin,title="Full Field Image Processing",buttons=('Masks','Labels','Cancel'),defaultbutton='Cancel',
                                                   message_text='Select the type of i0\nreference to use for\nthe calculation of mu.',
                                                   command=self.FFMakerRefNext)
        
    def FFMakerRefNext(self,result):
        self.FFreferenceDialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'Full field spectrum calculation cancelled')
            return        
        elif result=='Labels':
            self.FFMakerChannelReference()
        elif result=='Masks':
            self.FFMakerInternalReference()
        else:
            print('how did you get here?')

    def FFMakerInternalReference(self):
        #need to select mask area AND channels to process...       
        analysisOptions=['Start','Cancel']
        self.FFMakerIntRefChannelSelectDialog=Pmw.SelectionDialog(self.imgwin,title="Full Field Image Processing",buttons=analysisOptions,defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Data Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.FFMakerInReferenceNextSelect)#,buttonlength=4)
        self.FFMakerIntRefChannelSelectDialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
    
    def FFMakerInReferenceNextSelect(self,result):
        self.FFmakerSelectedDataChans=self.FFMakerIntRefChannelSelectDialog.getcurselection()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No alignment action taken')
            self.FFMakerIntRefChannelSelectDialog.withdraw()
            return
        if len(self.FFmakerSelectedDataChans)==0:
            globalfuncs.setstatus(self.status,'Select channels for alignment first')
            return           
        self.FFMakerIntRefChannelSelectDialog.withdraw()
        #now check theat the current data that corresponds to a mask channel

        #stolen from MCA fits...
        self.FFIntRefCDT={}
        curch=self.datachan.getvalue()[0]
        curind=self.mapdata.labels.index(curch)
        u=1
        m=max(ravel(self.mapdata.data.get(curind+2)))+1
        b=np.mod(ravel(self.mapdata.data.get(curind+2)),1)
        if m>31: u=0
        if sum(abs(b))>0: u=0
        if len(np.where(self.mapdata.data.get(curind+2)<0)[0])>1: u=0
        if u==0:
            if not tkinter.messagebox.askokcancel('Full Field Image Processing','Current selected data does not appear to be a mask set for reference.  Continue?'):
                globalfuncs.setstatus(self.status,'Masked reference normalization cancelled')
                return            
        self.FFIntRefCDT[self.mapdata.labels[curind]]=int(m)
        self.FFMakerInRefNext('Ok',curch)
        #print "inspecting data set"
        #globalfuncs.setstatus(self.status,'inspecting data sets')
        #self.FFIntRefCDT={}
        #i=0
        #for i in range(len(self.mapdata.labels)):
        #    u=1
        #    m=max(ravel(self.mapdata.data.get(i+2)))+1
        #    b=np.mod(ravel(self.mapdata.data.get(i+2)),1)
        #    if m>31: u=0
        #    if sum(abs(b))>0: u=0
        #    if len(np.where(self.mapdata.data.get(i+2)<0)[0])>1: u=0
        #    if u==1: 
        #        self.FFIntRefCDT[self.mapdata.labels[i]]=int(m)
        #
        #self.FFMakerIntRefMaskSelectDialog=Pmw.SelectionDialog(self.imgwin,title="Full Field Image Processing",buttons=('OK','Cancel'),defaultbutton='Cancel',
        #                                           scrolledlist_labelpos='n',label_text='Select channel that is the reference mask',scrolledlist_items=self.FFIntRefCDT.keys(),
        #                                           command=self.FFMakerInRefNext)
        
    def FFMakerInRefNext(self,result,maskchan):
        #maskchan=self.FFMakerIntRefMaskSelectDialog.getcurselection()[0]
        #self.FFMakerIntRefMaskSelectDialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'Full field spectrum calculation cancelled')
            return
        globalfuncs.setstatus(self.status,'processing data set')
        maskvalues=self.FFIntRefCDT[maskchan]    
        maskind=self.mapdata.labels.index(maskchan)+2     
        maskbase=self.mapdata.data.get(maskind)
        print("processing data set")
        mask=np.where(maskbase==1,1,0)
        nvals=np.sum(mask)
        if nvals==0:
            nvals=1.0
        else:
            nvals=float(nvals)
        for c in self.FFmakerSelectedDataChans:
            dataind=self.mapdata.labels.index(c)+2
            dr=self.mapdata.data.get(dataind)
            mr=mask*dr
            ref=sum(ravel(mr))/nvals
            mu=np.log(ref/dr)
            newlabel=c+".mu"
            self.addchannel(mu,newlabel)
            globalfuncs.setstatus(self.status,"Done with "+c)
        globalfuncs.setstatus(self.status,"Ready")

        
    def FFMakerChannelReference(self):        
        #assume CH=data (i1) CH.01 is i0 and CH.02 is offset
        if not tkinter.messagebox.askokcancel('Full Field Image Processing','This will use CH for i1 data,\nCH.01 for i0 reference,\nand CH.02 if present for dark'):
            globalfuncs.setstatus(self.status,'Full field spectrum calculation cancelled')
            return       
        for l in self.mapdata.labels:
            if l[:-2] in self.mapdata.labels: 
                print(l,"is access. data")
                continue
            if l+".1" not in self.mapdata.labels: 
                print(l,"has no accessory data")
                continue
            datindi1=self.mapdata.labels.index(l)+2
            datindi0=self.mapdata.labels.index(l+".1")+2
            if l+".2" in self.mapdata.labels:
                datindoff=self.mapdata.labels.index(l+".2")+2
                dataoff=self.mapdata.data.get(datindoff)
            else:
                dataoff=0
            datai1=self.mapdata.data.get(datindi1)-dataoff
            datai0=self.mapdata.data.get(datindi0)-dataoff
            mu=np.log(datai0/datai1)
            newlabel=l+".mu"
            self.addchannel(mu,newlabel)
            globalfuncs.setstatus(self.status,"Done with "+l)
        globalfuncs.setstatus(self.status,"Ready")
            

#################################  PCA routines

    def startPCAmini(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if sklHasFactorAnalysis:
            analysisOptions=['sPCA','CCIPCA','FA','NMF',"FastICA","SiVM",'Dictionary','LDA','Kmeans','Cancel']
        else:
            analysisOptions=['sPCA','CCIPCA','NMF',"FastICA","SiVM",'Kmeans','Cancel']
        if sklHasAdvancedCluster:
            analysisOptions.pop(analysisOptions.index('Kmeans'))
            analysisOptions.pop(analysisOptions.index('Cancel'))
            manifoldOptions=['Iso','MDS','tSNE']
            clusterOptions=['Kmeans','AffProp','MeanSh','Ward','Birch','AggCluster','Gaussian','Spectral','Cancel'] #'DBSCAN'
            if False:
                analysisOptions.extend(manifoldOptions)
            analysisOptions.extend(clusterOptions)
            analysisOptions=tuple(analysisOptions)
        print("self.mapdata.labels: ",self.mapdata.labels)
        print("done")
        
        self.pcaminidialog=smwPmw.SelectionBonusDialog(self.imgwin,title="Select PCA Channels",buttons=analysisOptions,defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.pcamininext,buttonlength=4)
        self.pcaminidialog.component('radioselect').add('Single File')
        self.pcaminidialog.component('radioselect').add('All Files')
        self.pcaminidialog.component('radioselect').invoke('Single File')
        self.pcaminidialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)

    def checkMultiNames(self,name,iters):
        for nbuf in iters:
            if name in self.dataFileBuffer[nbuf]['data'].labels:
                return False
        return True

    def checkMultiPCA(self,chans,ptype):
        if ptype=='Single File':
            return True
        for buf in list(self.dataFileBuffer.values()):
            for c in chans:
                if c not in buf['data'].labels:
                    print(c,'missing in',buf['name'])
                    return False
        return True

    def pcamininext(self,result):
        chans=self.pcaminidialog.component('scrolledlist').getcurselection()
        self.pcaFileTypes=self.pcaminidialog.component('radioselect').getcurselection()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No PCA action taken')
            self.pcaminidialog.withdraw()
            return
        if len(chans)==0:
            globalfuncs.setstatus(self.status,'Select channels for analysis first')
            return            
        
        self.pcaminidialog.withdraw()
        
        vm=0
        scale=0
        if result=='PCA+Vmx':
            vm=1
        if result=='PCA+ScVmx':
            vm=1
            scale=1
        #assemble PCA data...
        if not self.checkMultiPCA(chans,self.pcaFileTypes):
            print('Missing channels, PCA cancelled')
            return
        pcadata=[]
        self.PCAdatafileInfo={}
        if self.pcaFileTypes=='Single File':
            iters=[self.activeFileBuffer]
            self.dataFileBuffer[self.activeFileBuffer]['zoom']=self.maindisp.zmxyi
        else:
            iters=list(self.dataFileBuffer.keys())
        for c in chans:
            ndfin=[]
            for nbuf in iters:
                buf=self.dataFileBuffer[nbuf]
                dataind=buf['data'].labels.index(c)+2
                #worry about zooms
                dr=buf['data'].data.get(dataind)[::-1,:]#[::-1,:,dataind]
                ##and masks???
                ##if len(self.mask.mask)!=0 and self.usemaskinimage:
                ##    dr=self.mask.mask[::-1,1]*dr
                if buf['zoom'][0:4]!=[0,0,-1,-1]:
                    dr=dr[buf['zoom'][1]:buf['zoom'][3],buf['zoom'][0]:buf['zoom'][2]]
                nd=ravel(dr)
                info={}
                info['len']=len(nd)
                info['zoom']=buf['zoom']
                info['shape']=buf['data'].data.get(0).shape
                self.PCAdatafileInfo[nbuf]=info
                print(c,sum(nd))
                ndfin.extend(nd)
            pcadata.append(ndfin)
        pcarawdata=array(pcadata,dtype=np.float64)
        ##print "PCA data: ",self.PCArawdata.shape
        if not self.PCAcompMAXFixed:
            self.PCAcompMAXNO=pcarawdata.shape[0]
        pcarawdata=transpose(pcarawdata)
        print ('ERRCH: ',self.maindisp.zmxyi)        
        self.PCAdataStruct = PCAAnalysisMathClass.PCADataStructure(pcarawdata,self.PCAcompMAXNO,self.imgwin,
                                                                   pcaft=self.pcaFileTypes,
                                                                   dx = self.mapdata.data.get(0)[::-1,:],
                                                                   dy = self.mapdata.data.get(1)[::-1,:],
                                                                   zmxyi= self.maindisp.zmxyi)
        
        globalfuncs.setstatus(self.status,"WORKING ON PCA")        
        #try new
        ntm=[]
        ntm.append(time.process_time())
        self.PCAdataStruct.donewPCA(pcatype=result)
        ntm.append(time.process_time())
        print('PCA complete in '+str(ntm[1]-ntm[0])+' seconds')
        #compute sum
        if 'PCA' in result: #if result not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','SiVM','Gaussian']:
            esum=sum(self.PCAdataStruct.PCAeval)
            ecsum=[]
            rateind=[]
            evalsq=self.PCAdataStruct.PCAeval*self.PCAdataStruct.PCAeval
            for i in range(len(self.PCAdataStruct.PCAeval)):
                ecsum.append(sum(self.PCAdataStruct.PCAeval[0:i+1]))
                temp=sum(evalsq[i+1:len(self.PCAdataStruct.PCAeval)])
                try:
                    div=(len(self.PCAdataStruct.PCAeval)-(i+1))**5
                except:
                    div=0
                if div!=0:
                    rateind.append(math.sqrt(temp/div))
                else:
                    rateind.append(0)
            ecsum=array(ecsum)
            varcomp=self.PCAdataStruct.PCAeval/esum
            varexp=ecsum/esum
        if 'PCA' in result: print("IND values: ",rateind)
        globalfuncs.setstatus(self.status,"PCA Analysis complete")
        if vm:
            #varimax
            globalfuncs.setstatus(self.status,"Doing Varimax rotation")
            #make new uevect matrix
            if scale==0:
                newevect=self.PCAdataStruct.PCAuevect
            else:
                newevect=self.PCAdataStruct.PCAevect
            (a,b)=varimax.varimax(newevect.copy())
            #make new comp matrix
            rotevect=b
            trevect=transpose(rotevect)
            self.PCAdataStruct.PCAevect=trevect
            print(self.PCAdataStruct.PCAevect.shape)
            #rotate proportions?
            tprop=self.PCAdataStruct.PCAprop.copy()#transpose(self.PCAprop)
            nprop=tprop.copy()
            if scale==0:
                neweval=self.PCAdataStruct.PCAeval.copy()
                newevalmat=np.identity(len(neweval))*neweval
                first=np.dot(transpose(a),newevalmat)
                rtprop=np.dot(first,transpose(nprop))
            if scale==1:
                rtprop=np.dot(transpose(a),transpose(nprop))
            rtprop=transpose(rtprop)
            self.PCAdataStruct.PCAprop=rtprop.copy()     
            globalfuncs.setstatus(self.status,"Varimax rotation of selected scaled components completed")

        if result not in ['AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','LDA','SiVM','Gaussian','Dictionary','Iso','MDS','tSNE',]:
            globalfuncs.setstatus(self.status,"Checking for negative eigenvectors")
            for i in range(self.PCAdataStruct.PCAprop.shape[1]):
                dmax=max(self.PCAdataStruct.PCAevect[i,:])
                dmin=min(self.PCAdataStruct.PCAevect[i,:])
                if abs(dmin)>abs(dmax):
                    #need to inverse
                    self.PCAdataStruct.PCAevect[i,:]=-self.PCAdataStruct.PCAevect[i,:]
                    #need to adjust wt matrix:
                    z=transpose(self.PCAdataStruct.PCAprop)
                    t=z[i,:]
                    z[i,:]=-t
                    self.PCAprop=transpose(z)
            print('sizecheck VDX',self.PCAdataStruct.PCAevect.shape,self.PCAdataStruct.PCAeval.shape,self.PCAdataStruct.PCAprop.shape)
        print('begin export')
        globalfuncs.setstatus(self.status,"Exporting component weights to map dataset...")

        if self.pcaFileTypes=='Single File':
            iters=[self.activeFileBuffer]
            npost='SF'
        else:
            iters=list(self.dataFileBuffer.keys())
            npost='MF'

        self.PCAlastevect=None
        self.PCAlastprop=None
        self.PCAlastchans=None

        if result not in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian'] or self.showClusterVectors.get()==1:           
            self.PCAlastevect=self.PCAdataStruct.PCAevect
            self.PCAlastprop=self.PCAdataStruct.PCAprop
            self.PCAlastchans=chans
            cind=0
            for i in range(self.PCAdataStruct.PCAprop.shape[1]):
                noexit=1
                name=result+npost+'Comp'
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
                        nd=zeros(info['shape'],dtype=np.float32)
                        pm=nd[info['zoom'][1]:info['zoom'][3],info['zoom'][0]:info['zoom'][2]]
                        #data=self.PCArawdata[0,:]
                        data=data[:len(ravel(pm))]
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

        if result in ['Kmeans','AffProp','MeanSh','Spectral','Ward','Birch','DBSCAN','AggCluster','Gaussian']:
            cind=0
            noexit=1
            name=result+npost+'Clusters'
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
                    nd=zeros(info['shape'],dtype=np.float32)
                    pm=nd[info['zoom'][1]:info['zoom'][3],info['zoom'][0]:info['zoom'][2]]
                    #data=self.PCArawdata[0,:]
                    data=data[:len(ravel(pm))]
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

        if result=='SiVM' and self.pcaFileTypes=='Single File':
            #self.PCAdataStruct.PCAeval has the [x,y]
            self.plotmarkermain()
            for i in range(len(self.PCAdataStruct.PCAeval[0])-1):
                self.addmarker(xp=self.PCAdataStruct.PCAeval[0][i],yp=self.PCAdataStruct.PCAeval[1][i])
            self.setPMtext()
            
        #print the vectors
        if len(self.PCAdataStruct.PCAprop.shape) > 1:
            for i in range(self.PCAdataStruct.PCAprop.shape[1]):
                print("EV#"+str(i),self.PCAdataStruct.PCAevect[i,:])
                
        if self.PCAplotVectors.get():
            self.replotPCAplotVectors()        
        globalfuncs.setstatus(self.status,"Done!")
        print('done')
        
    def replotPCAplotVectors(self):  
        print("test in replotPCAplotVectors")
        if self.PCAlastevect is None: return
        if not self.maindisp.linegraph3present:
            self.maindisp.linegraph3present=1
            self.maindisp.newlineplot3=Pmw.MegaToplevel(self.maindisp.master)
            self.maindisp.newlineplot3.title('Vector Plot View')
            self.maindisp.newlineplot3.userdeletefunc(func=self.maindisp.killlineplot3)           
            h=self.maindisp.newlineplot3.interior()
            self.maindisp.graphx3=MyGraph.MyGraph(h,whsize=(4.5,4),graphpos=[[.15,.1],[.9,.9]])
        else:
            #clear old
            self.maindisp.newlineplot3.title('Line Plot View')
            self.maindisp.graphx3.cleargraphs()
        #make graphs
        palette=sblite.color_palette('hls', n_colors=8)
        colors=palette.as_hex()  # ['blue', 'red', 'green', 'white', 'orange','magenta', 'cyan','brown']
        print('PCApropSp',self.PCAlastprop.shape[1])
        print('PCAlastev',self.PCAlastevect.shape)
        for i in range(min(8,self.PCAlastprop.shape[1])):
            yv=self.PCAlastevect[i,:]
            if self.specX is None or len(self.specX)!=len(yv):
                xv=list(range(len(yv)))
            else:
                xv=self.specX
            self.maindisp.graphx3.plot(tuple(xv),tuple(yv),color=colors[i%len(colors)],text='EV'+str(i))
        self.maindisp.graphx3.uselegend(True)
        self.maindisp.graphx3.draw()
        self.maindisp.newlineplot3.show()

    def startPCAVectorAnalysis(self):
        print("     in doPCAVectorAnalysis")
        if self.PCAlastevect is None: return
        print ('PCAlastev',self.PCAlastevect.shape)
        print ('PCAlastchans',self.PCAlastchans)
        labels=[]
        for i in range(self.PCAlastevect.shape[0]):
            labels.append('Comp'+str(i+1))        
        
        if not self.maindisp.linegraph4present:
            self.maindisp.linegraph4present=1
            self.maindisp.newlineplot4=Pmw.MegaToplevel(self.maindisp.master)
            self.maindisp.newlineplot4.title('Vector Analysis Plot View')
            self.maindisp.newlineplot4.userdeletefunc(func=self.maindisp.killlineplot4)           
            h=self.maindisp.newlineplot4.interior()
            g1=Pmw.Group(h,tag_text='Component Selection',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
            g1.interior().configure(background='#d4d0c8')
            g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')            
            self.PCAAnalysisX=Pmw.ScrolledListBox(g1.interior(),labelpos='n',label_text='X Component',items=labels,listbox_selectmode='single',
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=self.plotPCAVectAnalysisPlot,listbox_height=5,
                                              hull_background='#d4d0c8',label_background='#d4d0c8')
            self.PCAAnalysisX.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')            
            self.PCAAnalysisY=Pmw.ScrolledListBox(g1.interior(),labelpos='n',label_text='Y Component',items=labels,listbox_selectmode='single',
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=self.plotPCAVectAnalysisPlot,listbox_height=5,
                                              hull_background='#d4d0c8',label_background='#d4d0c8')
            self.PCAAnalysisY.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')   
            #button box
            self.PCAVectAnalysisPlotLegend=True
            bb=PmwTtkButtonBox.PmwTtkButtonBox(g1.interior(),labelpos='n',label_text='Plot Legend',orient='vertical',pady=3,frame_background='#d4d0c8',hull_background='#d4d0c8',label_background='#d4d0c8')
            w=15
            bb.add('Show',command=self.PCAVectAnalysisShowButton,style='GREEN.TButton',width=w)
            bb.add('Hide',command=self.PCAVectAnalysisHideButton,style='FIREB.TButton',width=w)
            bb.pack(side=tkinter.LEFT,fill='both',pady=5)

            g2=Pmw.Group(h,tag_text='Channel Selection',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
            g2.interior().configure(background='#d4d0c8')
            g2.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')  
            self.PCAVectAnalysisAll=Pmw.ScrolledListBox(g2.interior(),labelpos='n',label_text='All Chans',items=self.PCAlastchans,listbox_selectmode='single',
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=tkinter.DISABLED,listbox_height=5,
                                              hull_background='#d4d0c8',label_background='#d4d0c8')
            self.PCAVectAnalysisAll.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')    
            #button box
            bb=PmwTtkButtonBox.PmwTtkButtonBox(g2.interior(),labelpos='n',label_text='',orient='vertical',pady=3,frame_background='#d4d0c8',hull_background='#d4d0c8',label_background='#d4d0c8')
            w=15
            bb.add('--Add-->',command=self.PCAVectAnalysisAddButton,style='GREEN.TButton',width=w)
            bb.add('Clear',command=self.PCAVectAnalysisClearButton,style='FIREB.TButton',width=w)
            bb.add('<--Remove--',command=self.PCAVectAnalysisRemoveButton,style='BROWN.TButton',width=w)
            bb.pack(side=tkinter.LEFT,fill='both',pady=5)
            self.PCAVectAnalysisPlot=Pmw.ScrolledListBox(g2.interior(),labelpos='n',label_text='Chans to Plot',items=[],listbox_selectmode='single',
                                              listbox_exportselection=tkinter.FALSE,selectioncommand=tkinter.DISABLED,listbox_height=5,
                                              hull_background='#d4d0c8',label_background='#d4d0c8')
            self.PCAVectAnalysisPlot.pack(side=tkinter.LEFT,padx=4,pady=5,fill='both')   
            
            self.maindisp.graphx4=MyGraph.MyGraph(h,whsize=(4.5,4),graphpos=[[.15,.1],[.9,.9]])
            self.maindisp.newlineplot4.title('Channel Vector Analysis')

        else:
            #clear old
            self.maindisp.graphx4.cleargraphs()        
            #set listboxes...
            self.PCAAnalysisX.setlist(labels)
            self.PCAAnalysisY.setlist(labels)
            self.PCAVectAnalysisPlot.clear()
            self.PCAVectAnalysisAll.setlist(self.PCAlastchans)  
            
    def PCAVectAnalysisShowButton(self):
        self.PCAVectAnalysisPlotLegend=True
        self.plotPCAVectAnalysisPlot()
        
    def PCAVectAnalysisHideButton(self):
        self.PCAVectAnalysisPlotLegend=False
        self.plotPCAVectAnalysisPlot()
                
    def PCAVectAnalysisAddButton(self):
        for item in self.PCAVectAnalysisAll.getvalue():
            if item not in self.PCAVectAnalysisPlot.get():
                self.PCAVectAnalysisPlot.insert(tkinter.END,item)
        self.plotPCAVectAnalysisPlot()
            
    def PCAVectAnalysisClearButton(self):
        self.PCAVectAnalysisPlot.clear()
        self.plotPCAVectAnalysisPlot()
    
    def PCAVectAnalysisRemoveButton(self):
        for item in self.PCAVectAnalysisAll.getvalue():
            if item in self.PCAVectAnalysisPlot.get():
                index = self.PCAVectAnalysisPlot.get().index(item)
                self.PCAVectAnalysisPlot.delete(index)
        self.plotPCAVectAnalysisPlot()
                
    def plotPCAVectAnalysisPlot(self,*args):
        clear = False
        if len(self.PCAVectAnalysisPlot.get()) < 1: clear = True
        if len(self.PCAAnalysisX.getvalue()) < 1: clear = True
        if len(self.PCAAnalysisY.getvalue()) < 1: clear = True
        self.maindisp.graphx4.cleargraphs()
        if clear:
            return
        #make graphs
        palette=sblite.color_palette('hls', n_colors=8)
        colors=palette.as_hex() 
        i=0
        for c in self.PCAVectAnalysisPlot.get():
            index = self.PCAlastchans.index(c)
            xindex = int(self.PCAAnalysisX.getvalue()[0].split('Comp')[1])-1
            yindex = int(self.PCAAnalysisY.getvalue()[0].split('Comp')[1])-1
            xv=[0,self.PCAlastevect[xindex,index]]
            yv=[0,self.PCAlastevect[yindex,index]]
            self.maindisp.graphx4.plot(tuple(xv),tuple(yv),color=colors[i%len(colors)],text=c)
            i+=1
        self.maindisp.graphx4.uselegend(self.PCAVectAnalysisPlotLegend)
        self.maindisp.graphx4.draw()
        
    def savePCArecent(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.PCAlastevect is None: 
            print ('No recent PCA data')
            globalfuncs.setstatus(self.status,"No recent PCA data")
            return
        #get file name
        fn=globalfuncs.ask_save_file('PCAexportData.out',self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return        
        #create text
        evtext = self.PCAlastevect.astype(np.float32).tobytes()
        proptext = self.PCAlastprop.astype(np.float32).tobytes()
        filetext = '####@PCA####\n@EV\n'
        filetext+= str(self.PCAlastevect.shape)+'\n'
        filetext+=str(evtext)+'\n'
        filetext+= '@PROP\n'
        filetext+= str(self.PCAlastprop.shape)+'\n'
        filetext+=str(proptext)+'\n'
        filetext+= '@CHAN\n'
        filetext+= str(len(self.PCAlastchans))+'\n'
        filetext+= '\n'.join(map(str,self.PCAlastchans))
        filetext+= '\n####END####\n'
        #write and close
        fid=open(fn,"w")   
        fid.write(filetext)
        fid.close()
        
        print ('PCA data export complete')
        globalfuncs.setstatus(self.status,"PCA data export complete")        

    def loadPCArecent(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #get file
        fty=[("PCA data files","*.out"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())   
        if t=='':
            print('Load cancelled')
            globalfuncs.setstatus(self.status,'Load cancelled')
            return               
        #read/parse file
        fid=open(t,"r")
        d=fid.readlines()
        fid.close()
        if d[0].strip() != '####@PCA####':
            print('Invalid Data File')
            globalfuncs.setstatus(self.status,'Invalid Data File')
            return           
        newev=np.frombuffer(eval(d[3]),dtype=np.float32).reshape(eval(d[2]))
        newprop=np.frombuffer(eval(d[6]),dtype=np.float32).reshape(eval(d[5]))
        chanlen=int(d[8])
        chans=[]
        for i in range(chanlen):
            chans.append(d[9+i].strip())
        #channel check
        missinglist=[]
        for c in chans:
            if c not in self.mapdata.labels:
                missinglist.append(c)
        if len(missinglist)>0:
            mlt=''
            for c in missinglist:
                mlt+=c+'\n'
            if not tkinter.messagebox.askokcancel('PCA Value Load','Following channels not in current data set... OK?\n'+mlt):
                print('Load cancelled')
                globalfuncs.setstatus(self.status,'Load cancelled')
                return                            
        #set variables
        self.PCAlastevect=newev
        self.PCAlastprop=newprop
        self.PCAlastchans=chans
        
        print ('PCA data import complete')
        globalfuncs.setstatus(self.status,"PCA data import complete")        

        
    def startPCA(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return    
        #new window
        if not self.PCAViewWindow.exist: #PCAviewexist:
            ps=PCAAnalysisClass.PCAParams(self.getMCAfile,self.MCAfilename,self.status,self.addchannel,self.dataFileBuffer,self.activeFileBuffer,self.filedir,self.PCAhdf5fout,self.HDFCOMPRESS)
            self.PCAViewWindow.create(self.mapdata,ps)
        else:
            self.PCAViewWindow.win.show()





#this stays here too... but not for PCA MCA data info -- decorrelated now
    def editPCAMAX(self):
        new=tkinter.simpledialog.askinteger('PCA Components','Enter max number of PCA components: ',initialvalue=self.PCAcompMAXNO)
        self.PCAcompMAXNO=new
        if new==0:
            self.PCAcompMAXFixed=False
        else:
            self.PCAcompMAXFixed=True



################################# Linear Combo XANES Fitting

    def startXANESfitting(self):
        #do multiple point linear combo fitting
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.xanesFitWindow.exist:
            ps=XanesFitClass.XanesFitWindowParams(self.status, self.XFITOPT, self.addchannel, self.filedir)
            self.xanesFitWindow.create(self.mapdata, ps)
        else:
            self.xanesFitWindow.win.show()
            
################################# Adv Smooth and Sharpen

    def startadvfilterwin(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.advancedFilterWindow.exist:
            ps=AdvancedFilteringClass.AdvancedFilteringWindowParams(self.status, self.maindisp, self.showmap, self.savedeconvcalculation)
            self.advancedFilterWindow.create(self.mapdata, ps)
        else:
            self.advancedFilterWindow.win.show()


        

       

################################# Custom Kernel Filters

    def startcustomkernels(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.customKernelWindow.exist:
            ps=CustomKernelClass.CustomKernelWindowParams(self.status, self.maindisp,self.showmap, self.savedeconvcalculation, self.filedir)
            self.customKernelWindow.create(self.mapdata, ps)
        else:
            self.customKernelWindow.win.show()



        

################################# Particle Stats?

    def startPartWaterStats(self):
        self.startPartStats(useWater=1)

    def startMaskStats(self):
        self.startPartStats(useMask=1)

    def startPartStats(self,useMask=0,useWater=0):
        # self.partStatuseMask=useMask
        # self.partStatuseWater=0#useWater
        # self.partstatROIflag=0
            
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return

        if self.datachan.get()==():
            return

        globalfuncs.setstatus(self.status,"Ready")

        ps = ParticleStatisticsClass.ParticleStatisticsWindowParams(useMask, useWater, self.status, self.isStatCalcMultiFile, self.activeFileBuffer, self.dataFileBuffer, self.datachan, self.partROIThresh, self.partWaterThresh, self.nullMaskCalc, self.dodt, self.deadtimevalue, self.DTICRchanval, self.root, self.doI0c, self.domapimage, self.maindisp, self.usemaskinimage, self.showmap, self.filedir, self.addchannel)
        if self.particleStatisticsWindow.exist:
            self.particleStatisticsWindow.update(self.mapdata, ps)
        else:
            self.particleStatisticsWindow.create(self.mapdata, ps)



    def setPartWaterThresh(self):
        n=tkinter.simpledialog.askinteger(title='Watershed Thresholding',prompt='Set Minimum Pixel Distance for Watersheding',initialvalue=str(self.partWaterThresh))
        if n is None:
            print('Threshold cancelled')
            globalfuncs.setstatus(self.status,'Threshold cancelled')
            return
        self.partWaterThresh=n

    def setPartROIThresh(self):
        n=tkinter.simpledialog.askinteger(title='ROI Thresholding',prompt='Set Minimum Pixel Area for ROI Thresholds',initialvalue=str(self.partROIThresh))
        if n is None:
            print('Threshold cancelled')
            globalfuncs.setstatus(self.status,'Threshold cancelled')
            return
        self.partROIThresh=n

################################# Multiple Mass Calibration

   

    def startMultiMass(self, useMask=1,useWater=0):
       
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return

        if self.datachan.get()==():
            return

        globalfuncs.setstatus(self.status,"Ready")

        ps = MultiMassCalibrationClass.MultiMassCalibrationWindowParams(useMask, useWater, self.status, self.isStatCalcMultiFile, self.activeFileBuffer, self.dataFileBuffer, self.datachan, self.partROIThresh, self.nullMaskCalc, self.domapimage, self.maindisp, self.usemaskinimage, self.showmap,self.filedir, SAMmasks = self.lastSAMmask)
        if self.multiMassWindow.exist:
            self.multiMassWindow.kill()
        self.multiMassWindow.create(self.mapdata, ps)


    def startStandardEditor(self, useMask=0,useWater=0):
       
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return

        #if self.datachan.get()==():
        #    return

        globalfuncs.setstatus(self.status,"Ready")

        ps=ConcentrationStandardClass.ConcStdParams(self.filedir,self.status)
        if not self.editConcStandardsWindow.exist:
            self.editConcStandardsWindow.create(self.mapdata, ps)



    def doQuantify(self):
        #call dialog starter with function
        self.quantgendialog(self.enterQuantFileLoad)        
        
    def enterQuantFileLoad(self,result):
        self.quantselchans = self.selectquantdialog.getcurselection()
        self.selectquantdialog.withdraw()

        if result=='Cancel' or self.quantselchans==():
            globalfuncs.setstatus(self.status,'No action taken')
            return
        
        #get file?
        fn=globalfuncs.ask_for_file([("QPM Calibration files","*.qpm"),("all files","*")],self.filedir.get(),multi=False)
        if fn == '' or fn is None:
            print ('canceled')
            globalfuncs.setstatus(self.ps.status,"No calibration file defined...")
            return
        
        self.cfilewid = MultiMassCalibrationClass.CalibResultObject()
        self.cfilewid.loadFile(fn)
        
        
        
        #correlate names and channels again...
        self.chanquantdict={}
        self.quantCCSdialog=Pmw.Dialog(self.imgwin,title='Correlate Map Channels to Calibration File',buttons=('OK','Cancel'),defaultbutton='OK',
                                      command=self.qccsordone)
        inter=self.quantCCSdialog.interior()
        self.qfcd={}
        
        #normalize channels?
        self.ncb=None
        if "None" not in self.cfilewid.normalize:
            #select the norm channel...
            self.ncb=Pmw.ComboBox(inter,label_text="Normalize Channel:",labelpos='w',history=0,scrolledlist_items=self.mapdata.labels,dropdown=1)
            self.ncb.pack(side=tkinter.TOP,padx=5,pady=10)            
        
        for q in self.quantselchans:
            cb=Pmw.ComboBox(inter,label_text=q,labelpos='w',history=0,scrolledlist_items=self.cfilewid.ellist,dropdown=1)
            cb.pack(side=tkinter.TOP,padx=5,pady=5)
            for po in self.cfilewid.ellist:
                if q in self.cfilewid.chdict[po].channels:
                    cb.selectitem(po,setentry=1)
            self.qfcd[q]=cb
        alist = list(self.qfcd.values())
        #if self.ncb is not None:
        #    alist.append(self.ncb)
        Pmw.alignlabels(alist)
        self.quantCCSdialog.show()
        
    def qccsordone(self,result):
        if result=='Cancel':
            print('Load cancelled')
            self.quantCCSdialog.withdraw()
            return        
        #check validity
        for n in list(self.qfcd.keys()):
            if self.qfcd[n].get()=='':
                print('Need all channels correlated')
                return
            else:
                self.chanquantdict[n]=self.qfcd[n].get()
        if self.ncb is not None:
            if self.ncb.get()=='':
                print('Need normalize channel defined')
                return
            else:
                self.normCCSchan=self.ncb.get()   
        else:
            self.normCCSchan=None
        self.quantCCSdialog.withdraw()     
        
        #now do a bunch of calibrations...
        globalfuncs.setstatus(self.status,'Doing quantitative analysis...')
        #do
        self.doStQuant()
        globalfuncs.setstatus(self.status,'Quantative analysis complete!')        
        
    def doStQuant(self):
        print ('StQfy')        
        #loop for each channel requests....
        for q in self.quantselchans:
            #get data
            Aind=self.mapdata.labels.index(q)+2     
            (xlen,ylen)=self.mapdata.data.shape[:2]

            if self.normCCSchan is not None:
                iind=self.mapdata.labels.index(self.normCCSchan)+2
                i0dat=self.mapdata.data.get(iind)#[:,:,iind]
            else:
                i0dat=np.ones((xlen,ylen),dtype=np.float32)
            #divide by i0
            adata=self.mapdata.data.get(Aind)
            
            newdata=np.divide(adata,i0dat, out=np.zeros_like(adata),where=i0dat!=0)
            
            
            #fw contains the calibration and units data in fw.slope, fw.intc, and fw.units
            #fw for this is: self.cfilewid.chdict[el]
            # el = self.chanquantdict[chan] where chan is q
            fw = self.cfilewid.chdict[self.chanquantdict[q]]
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
                if newname not in self.mapdata.labels:
                    valid=1
                else:
                    i+=1
            self.addchannel(newdata,newname)
            
            

################################# Segmentation

    def initSAMParams(self):
        self.lastSAMmask=None
        globalfuncs.setstatus(self.status,"Initialize SAM model")
        self.imgwin.update()
        mode,path = globalfuncs.getModePath()
        p = path+os.sep+"sam_vit_h_4b8939.pth"
        print (os.path.exists(p))
        t=time.time()
        self.samModel = sam_model_registry["default"](checkpoint=p)
        et = time.time()-t
        self.samSegmentInitialized = True
        globalfuncs.setstatus(self.status,"Initialized, "+str(et)+" seconds, Ready")
        
    def startSAMAuto(self):
        self.startMainSAM()

    def startSAMDetailed(self):
        self.startMainSAM(detail=True)
    
    def startSAMFlat(self):
        self.startMainSAM(flat=True)

    def startMainSAM(self,flat=False,detail=False):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #check initialization
        if not self.samSegmentInitialized:
            self.initSAMParams()
        #extra params
        #pts per side
        defpps = int(float(min(self.mapdata.nxpts,self.mapdata.nypts))/5.0)
        if detail:
            #ask defpps
            npps=tkinter.simpledialog.askinteger(title='Segmentation',prompt='Enter number of prompt points per side',initialvalue=defpps)
            if npps<=0 or npps is None:
                npps=defpps
            defpps=npps
            #crop layers
            croplayers=tkinter.simpledialog.askinteger(title='Segmentation',prompt='Enter number of crop layers',initialvalue=0)
            if croplayers<0 or croplayers is None:
                croplayers=0
            #crop overlap
            croplap=tkinter.simpledialog.askfloat(title='Segmentation',prompt='Enter fractional overlap',initialvalue=0.10)
            if croplap<0 or croplap is None:
                croplap=0            
            #min mask region area
            minarea=tkinter.simpledialog.askinteger(title='Segmentation',prompt='Enter minumum pixel area',initialvalue=self.partROIThresh)
            if minarea<0 or minarea is None:
                minarea=self.partROIThresh           
            samargs={"points_per_side":defpps,"crop_n_layers":croplayers,"crop_overlap_ratio":croplap,"min_mask_region_area":minarea}
        else:
            samargs={"points_per_side":defpps}
        #use currently selected channel to segment...
        maskgen = SamAutomaticMaskGenerator(model=self.samModel,**samargs)
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        im=self.mapdata.data.get(datind).astype(np.float32)
        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
            im=im[::-1,:]
            im=im[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            im=im[::-1,:]
        #normalize
        im=im/np.max(im)
        im=im*255
        im=im.astype(np.uint8)
        #convert to openCV object...
        imcv = cv.cvtColor(im,cv.COLOR_GRAY2BGR)
        globalfuncs.setstatus(self.status,"Finding masks")
        self.imgwin.update()
        t=time.time()
        masks = maskgen.generate(imcv)
        print (time.time()-t)
        globalfuncs.setstatus(self.status,"Masks complete, forming composite")
        self.imgwin.update()
        print (len(masks))
        #create a masked channel
        allmask = np.zeros(im.shape,np.float32)
        j=1
        toguimasks = []
        for m in masks:
            #allmask = np.zeros(im.shape,np.float32)
            allmask+=j*m['segmentation']
            print (j,len(masks))
            j+=1

            nmc = np.zeros(self.mapdata.data.get(0).shape,np.float32)
            if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
                nmc[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=1*m['segmentation'][::-1,:]
                nmc=nmc[::-1,:]            
            toguimasks.append(nmc)
            
        newmaskchan = np.zeros(self.mapdata.data.get(0).shape,np.float32)
        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
            newmaskchan[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=allmask[::-1,:]
        else:
            newmaskchan=allmask[::-1,:]
        newmaskchan=newmaskchan[::-1,:]
        #add to data
        nameroot=self.datachan.getvalue()[0]+'-seg'
        valid=0
        i=1
        while not valid:
            newname=nameroot+str(j).zfill(3)+'-'+str(i).zfill(3)
            if newname not in self.mapdata.labels:
                valid=1
            else:
                i+=1
        self.addchannel(newmaskchan,newname)    
        j+=1
        
        useMask=0
        useWater=0
        ps = ParticleStatisticsClass.ParticleStatisticsWindowParams(useMask, useWater, self.status, self.isStatCalcMultiFile, self.activeFileBuffer, self.dataFileBuffer, self.datachan, self.partROIThresh, self.partWaterThresh, self.nullMaskCalc, self.dodt, self.deadtimevalue, self.DTICRchanval, self.root, self.doI0c, self.domapimage, self.maindisp, self.usemaskinimage, self.showmap, self.filedir, self.addchannel, SAMmasks=toguimasks)
        if self.particleStatisticsWindow.exist:
            self.particleStatisticsWindow.update(self.mapdata, ps)
        else:
            self.particleStatisticsWindow.create(self.mapdata, ps)
            
        #save mask...
        self.lastSAMmask=toguimasks
        globalfuncs.setstatus(self.status,"Ready...")

    def reuseSAMResult(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.lastSAMmask is None:
            print('No SAM results')
            globalfuncs.setstatus(self.status,'No SAM segmentation results')
            return            
        useMask=0
        useWater=0
        ps = ParticleStatisticsClass.ParticleStatisticsWindowParams(useMask, useWater, self.status, self.isStatCalcMultiFile, self.activeFileBuffer, self.dataFileBuffer, self.datachan, self.partROIThresh, self.partWaterThresh, self.nullMaskCalc, self.dodt, self.deadtimevalue, self.DTICRchanval, self.root, self.doI0c, self.domapimage, self.maindisp, self.usemaskinimage, self.showmap, self.filedir, self.addchannel, SAMmasks=self.lastSAMmask)
        if self.particleStatisticsWindow.exist:
            self.particleStatisticsWindow.update(self.mapdata, ps)
        else:
            self.particleStatisticsWindow.create(self.mapdata, ps)

    def saveSAMResulttoFile(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.lastSAMmask is None:
            print('No SAM results')
            globalfuncs.setstatus(self.status,'No SAM segmentation results')
            return   
        fty=[("SAM param files","*.spm"),("all files","*")]
        fn=globalfuncs.ask_save_file('segments.spm',self.filedir.get(),ext=fty)        
        if fn!='':
            #test extensions
            if os.path.splitext(fn)[1]=='':fn=fn+'.spm'
        else:
            globalfuncs.setstatus(self.status,"cancelling save...")
            return  
        encodedmask = []
        for m in self.lastSAMmask:
            encodedmask.append(mask_utils.encode(np.asfortranarray(m.astype(dtype=np.uint8))))
        filetext = pickle.dumps(encodedmask)
        #write and close
        fid=open(fn,"wb")   
        fid.write(filetext)
        fid.close()        
        globalfuncs.setstatus(self.status,'Segmentation results saved')
        
    
    def loadSAMResultfromFile(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #get file
        fty=[("SAM param files","*.spm"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())   
        if t=='':
            print('Load cancelled')
            globalfuncs.setstatus(self.status,'Load cancelled')
            return  
        #read/parse file
        fid=open(t,"rb")
        d=fid.read()
        fid.close()
        encodedmasks=pickle.loads(d)
        self.lastSAMmask=[]
        for m in encodedmasks:
            self.lastSAMmask.append(mask_utils.decode(m))
        globalfuncs.setstatus(self.status,'Load complete')   
        

    def RFsegment(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #take current data as labels
        curch=self.datachan.getvalue()[0]
        curind=self.mapdata.labels.index(curch)
        u=1
        m=np.max(ravel(self.mapdata.data.get(curind+2)))+1
        b=np.mod(ravel(self.mapdata.data.get(curind+2)),1)
        if m>31: u=0
        if sum(abs(b))>0: u=0
        if len(np.where(self.mapdata.data.get(curind+2)<0)[0])>1: u=0
        if u==0:
            if not tkinter.messagebox.askokcancel('Spectrum Export','Current selected data does not appear to be a mask set.  Continue?'):
                globalfuncs.setstatus(self.status,'Spectrum export cancelled')
                return            
        self.RFtrlab = self.mapdata.data.get(curind+2)
        
        #get channels for analyis:
        self.RFchannelsDialog=Pmw.SelectionDialog(self.imgwin,title="Random Forest Segmentation",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.RFdestination)
        self.RFchannelsDialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        
    def RFdestination(self,result):
        allchans=self.RFchannelsDialog.getcurselection()
        self.RFchannelsDialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            return  
        
        img = np.zeros((self.RFtrlab.shape[0],self.RFtrlab.shape[1],len(allchans)))
        ind=0
        for c in allchans:
            img[:,:,ind]=self.mapdata.data.get(self.mapdata.labels.index(c)+2)
            ind+=1
        
        sigma_min=1
        sigma_max=16
        features_func = functools.partial(skimage.feature.multiscale_basic_features,
                                          intensity=True, edges= False, texture=True,
                                          sigma_min=sigma_min, sigma_max=sigma_max,channel_axis=-1)
        globalfuncs.setstatus(self.status,"Finding features")        
        features = features_func(img)
        globalfuncs.setstatus(self.status,"Classifying")
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,max_depth=10, max_samples=0.05)
        clf = skimage.future.fit_segmenter(self.RFtrlab,features,clf)
        globalfuncs.setstatus(self.status,"Predicting")
        result = skimage.future.predict_segmenter(features,clf)

        basename='RFsegment'
        newname=globalfuncs.fixlabelname(basename)
        ind=1
        ok=False
        while not ok:
            if newname in self.mapdata.labels:
                newname=globalfuncs.fixlabelname(basename+'_'+str(ind))
                ind+=1
            else:
                ok=True
        #add the channel
        self.addchannel(result,newname)     
        globalfuncs.setstatus(self.status,"Ready")        
        
#################################  Radial profile

    def makeradialstart(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        ps = RadialProfileClass.RadialProfileParams(self.maindisp, self.root, self.datachan)
        radialProfileWindow = RadialProfileClass.RadialProfile(self.imgwin, self.mapdata, ps)


#################################  File Transformations

    def tf_rotateFile(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.tf_transformFile(op='Rotate',ext='rot')

    def tf_vertflipFile(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.tf_transformFile(op='Vertical Flip',ext='vf')

    def tf_horzflipFile(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.tf_transformFile(op='Horizontal Flip',ext='hf')


    def tf_transformFile(self,op=None,ext='tf'):
        if op is None:
            print('No Operation')
            globalfuncs.setstatus(self.status,'No Operation')
            return   
        if not tkinter.messagebox.askokcancel('Transform File Export',op+' file orientation to new file?'):
            globalfuncs.setstatus(self.status,op+' file export cancelled')
            return  

        #get new filename       
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_'+ext+'.hdf5'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            return
        
        if op == 'Vertical Flip':
            f = MathWindowClass.VFlipOp
        elif op == 'Horizontal Flip':
            f = MathWindowClass.HFlipOp
        else:
            f = np.transpose

        globalfuncs.setstatus(self.status,"Saving new file with "+op.lower()+"...")
        newfile=ImageGet.EmptyHDF5(fn)
        pdict={}
        pdict['channels']=self.mapdata.channels
        pdict['type']=self.mapdata.type
        pdict['isVert']=self.mapdata.isVert
        pdict['labels']=self.mapdata.labels
        pdict['comments']=self.mapdata.comments
        pdict['energy']=self.mapdata.energy
        newfile.cleanString()

        imy=self.mapdata.data.get(0)
        imx=self.mapdata.data.get(1)
        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
            imx=imx[::-1,:]
            imx=imx[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            imx=imx[::-1,:]
            imy=imy[::-1,:]
            imy=imy[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            imy=imy[::-1,:]

        xv=imy[0,:]
        yv=imx[:,0]
 
        newdatay=f(imy)
        newdatax=f(imx)

        fxv=newdatax[0,:]
        fyv=newdatay[:,0]

        if op == 'Rotate':
            newfile.addParams(yv,xv,pdict)
        else:
            newfile.addParams(fxv,fyv,pdict)

        newfile.data.put(0,newdatay)
        newfile.data.put(1,newdatax)
        for j in range(self.mapdata.channels):
            im=self.mapdata.data.get(j+2)
            if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
                im=im[::-1,:]
                im=im[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                im=im[::-1,:]

            newdata=f(im)
            newfile.data.put(j+2,newdata)
        newfile.close()        
        globalfuncs.setstatus(self.status,"Saving new file with "+op.lower()+" transformation complete.")

        #load new file?
        if tkinter.messagebox.askyesno(title=op+" File Export",message="Load transformed file?"):
            self.fileentry.setvalue(fn)
            self.load_data_file(ignore=1)

#################################  Resolution change  skrescale

    def startResChange(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.reschangeDialog=Pmw.Dialog(self.imgwin,title="Change Resolution",buttons=('Preview','Save','Cancel'),defaultbutton='Cancel',
                                     command=self.enterResChange)        
        h=self.reschangeDialog.interior()
        h.configure(background='#d4d0c8')
        lf=tkinter.Frame(h,background='#d4d0c8')
        lf.pack(side=tkinter.LEFT,fill='both')
        #get cur resolutions...
        curXres=int(abs(self.mapdata.xvals[2]-self.mapdata.xvals[1])*100000)/100000.
        curYres=int(abs(self.mapdata.yvals[2]-self.mapdata.yvals[1])*100000)/100000.
        self.changeResX=Pmw.EntryField(h,label_text='Horizontal Resolution',labelpos='w',validate='real',value=float(curXres),hull_background='#d4d0c8',label_background='#d4d0c8')
        self.changeResX.pack(side=tkinter.TOP,padx=5,pady=5)
        self.changeResY=Pmw.EntryField(h,label_text='Vertical Resolution',labelpos='w',validate='real',value=float(curYres),hull_background='#d4d0c8',label_background='#d4d0c8')
        self.changeResY.pack(side=tkinter.TOP,padx=5,pady=5) 
        Pmw.alignlabels([self.changeResX,self.changeResY])
        
        self.reschangeDialog.show()
        
    def enterResChange(self,result):
        if result=='Cancel':
            self.reschangeDialog.withdraw()
            return
        newXR=float(self.changeResX.getvalue())
        newYR=float(self.changeResY.getvalue())
        curXres=int(abs(self.mapdata.xvals[2]-self.mapdata.xvals[1])*100000)/100000.
        curYres=int(abs(self.mapdata.yvals[2]-self.mapdata.yvals[1])*100000)/100000.       
        xscale=curXres/newXR
        yscale=curYres/newYR
        if result=='Preview':
            i=self.mapdata.labels.index(self.datachan.getvalue()[0])
            im=self.mapdata.data.get(i+2)
            if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
                im=im[::-1,:]
                im=im[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                im=im[::-1,:]
            if xscale<1 or yscale<1:
                aa=True
            else:
                aa=False
            newdata=skrescale(im,(xscale,yscale),mode='edge')
            print(im.shape)
            print(newdata.shape)
            tempzmxyi=self.maindisp.zmxyi.copy()
            globalfuncs.setList(self.maindisp.zmxyi,[0,0,-1,-1,0,0])
            self.docalcimage(newdata,sc=False)
            globalfuncs.setList(self.maindisp.zmxyi,tempzmxyi)
        if result=='Save':
            #get new filename
            fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_res_'+str(newXR)+'_'+str(newYR)+'.hdf5'
            fn=globalfuncs.ask_save_file(fn,self.filedir.get())
            if fn=='':
                print('Save cancelled')
                return
            globalfuncs.setstatus(self.status,"Saving new file with changed resolution...")
            newfile=ImageGet.EmptyHDF5(fn)
            pdict={}
            pdict['channels']=self.mapdata.channels
            pdict['type']=self.mapdata.type
            pdict['isVert']=self.mapdata.isVert
            pdict['labels']=self.mapdata.labels
            pdict['comments']=self.mapdata.comments
            pdict['energy']=self.mapdata.energy
            newfile.cleanString()

            imy=self.mapdata.data.get(0)
            imx=self.mapdata.data.get(1)
            if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
                imx=imx[::-1,:]
                imx=imx[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                imx=imx[::-1,:]
                imy=imy[::-1,:]
                imy=imy[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                imy=imy[::-1,:]
            if xscale<1 or yscale<1:
                aa=True
            else:
                aa=False
            newdatay=skrescale(imy,(xscale,yscale),mode='edge')
            newdatax=skrescale(imx,(xscale,yscale),mode='edge')

            xv=newdatay[0,:]
            yv=newdatax[:,0]
 
            print (newdatay[0,:])   
            print (newdatay[-1,:])   
            print (newdatax[:0])   
            print (newdatax[:-1])   
 
            newfile.addParams(xv,yv,pdict)
            newfile.data.put(0,newdatay)
            newfile.data.put(1,newdatax)
            for j in range(self.mapdata.channels):
                im=self.mapdata.data.get(j+2)
                if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
                    im=im[::-1,:]
                    im=im[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                    im=im[::-1,:]
                if xscale<1 or yscale<1:
                    aa=True
                else:
                    aa=False
                newdata=skrescale(im,(xscale,yscale),mode='edge')
                newfile.data.put(j+2,newdata)
            newfile.close()
            
            self.reschangeDialog.withdraw()
            globalfuncs.setstatus(self.status,"Saving new file with changed resolution complete.")
            #load new file?
            if tkinter.messagebox.askyesno(title="Resolution Change",message="Load file with new resolution?"):
                self.fileentry.setvalue(fn)
                self.load_data_file(ignore=1)
                
            
#################################  Deconvolution

    def startDeconv(self):
        #do beam deconvolution
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return

        if not self.beamDeconvolutionWindow.exist:
            ps=BeamDeconvolutionClass.BeamDeconvolutionWindowParams(self.status, self.maindisp,self.showmap,self.addchannel)

            
            self.beamDeconvolutionWindow.create(self.mapdata, ps)
        else:
            self.beamDeconvolutionWindow.win.show()
        

    def savedeconvcalculation(self,newd,name):
        #make sure name present
        if name=='':
            print('Enter new channel name')
            setstatus(self.status,'Enter new channel name')
            return
        #make sure name unique
        newname=globalfuncs.fixlabelname(name)
        if newname in self.mapdata.labels:
            print('Enter unique channel name')
            globalfuncs.setstatus(self.status,'Enter unique channel name')
            return            
        #save new channel        
        self.addchannel(newd,newname)
    
#################################  Deadtimes

    def deadtimecorrection(self,iv=None):
        #apply deadtime dialog...
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.deadtimedialogexist:
            self.deadtimedialog.show()
            self.DTICRchan.setitems(self.mapdata.labels)         

            return
        self.deadtimedialogexist=1
        self.deadtimedialog=Pmw.Dialog(self.imgwin,title='Deadtime Correction',buttons=('OK','Cancel'),
                                          defaultbutton='OK',command=self.dotheDTC)
        h=self.deadtimedialog.interior()
        f=tkinter.Frame(h)
        f.pack(side=tkinter.TOP,fill='both')
        #dialog for:
        #ask for deadtime tau value in usecs
        if iv is None:
            iv=1
        self.deadtimevalue=Pmw.EntryField(f,labelpos='w',label_text='Deadtime tau (usec)',value=iv,validate='real',entry_width=10)
        self.deadtimevalue.pack(side=tkinter.TOP,padx=2,pady=5)
        #ask for ICR channel
        self.DTICRchan=Pmw.OptionMenu(f,labelpos='w',label_text='ICR Channel',labelmargin=4,items=self.mapdata.labels,menubutton_width=15)
        self.DTICRchan.pack(side=tkinter.TOP,padx=2,pady=5)
        if 'ICR' in self.mapdata.labels:
            self.DTICRchan.invoke('ICR')
        #correction active or not
        cb=tkinter.Checkbutton(f,text="Do DT correction?",variable=self.dodt,anchor=tkinter.W)
        cb.pack(side=tkinter.TOP,padx=2,pady=5)
        return self.deadtimevalue

    def dotheDTC(self,result):
        self.deadtimedialog.withdraw()
        if result=='Cancel':
            print('Deadtime correction cancelled')
            globalfuncs.setstatus(self.status,'Deadtime correction cancelled')
            return
        #validate responses
        if not self.deadtimevalue.valid():
            print('Invalid deatime parameter')
            globalfuncs.setstatus(self.status,'Invalid deadtime parameter')
            return            
        #DT: corFF=FF*exp(tau*1e-6*ICR)
        self.DTICRchanval=self.mapdata.labels.index(self.DTICRchan.getvalue())+2
        self.domapimage()

        if self.correlationPlot.exist:
            self.correlationPlot.checkcorplot(dtval=self.DTICRchanval)
        if self.triColorWindow.exist:
            self.triColorWindow.dotcdisplay()

    def writeSIXPACKdtf(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return        
        #make sure value not zero
        if self.deadtimevalue is None:
            print('No deadtime value')
            globalfuncs.setstatus(self.status,'No deadtime value')
            return                   
        #ask for file name
        fn='deadtime.dat'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Deadtime save cancelled')
            globalfuncs.setstatus(self.status,'Deadtime save cancelled')
            return        
        #ask number of detector channels (all get the same!!!)
        mt='How many detector channels present?'
        dtchans=tkinter.simpledialog.askinteger(title='Save Deadtime File',prompt=mt,initialvalue=1)
        if dtchans is None:
            print('Deadtime save cancelled')
            globalfuncs.setstatus(self.status,'Deadtime save cancelled')
            return
        mt='Data dwell time (sec)?'
        coltime=tkinter.simpledialog.askfloat(title='Save Deadtime File',prompt=mt,initialvalue=0.25)
        if dtchans is None:
            print('Deadtime save cancelled')
            globalfuncs.setstatus(self.status,'Deadtime save cancelled')
            return
        dtw=float(self.deadtimevalue.getvalue())*coltime
        #write file
        fid=open(fn,'w')
        fid.write('FF\ttau\toffset\tkappa\n')
        for i in range(dtchans):
            fid.write(str(i)+'\t'+str(dtw)+'\t0\t0\n')
        fid.close()        

##################################### I0 correction

    def seti0channel(self,wd=0):
        #apply i0channel dialog...
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if wd:
            if self.i0chandialogexist:
                self.i0chan.setitems(self.mapdata.labels)
                if 'I0' in self.mapdata.labels:
                    self.i0chan.invoke('I0')
                if 'I0STRM' in self.mapdata.labels:
                    self.i0chan.invoke('I0STRM')
                self.dotheI0(0)
                return
        if self.i0chandialogexist:
            self.i0chan.setitems(self.mapdata.labels)
            self.i0chandialog.show()
            return
        self.i0chandialogexist=1
        self.i0chandialog=Pmw.Dialog(self.imgwin,title='I0 Channel Selection',buttons=('OK','Cancel'),
                                          defaultbutton='OK',command=self.dotheI0)
        h=self.i0chandialog.interior()
        f=tkinter.Frame(h)
        f.pack(side=tkinter.TOP,fill='both')
        #dialog for:
        #ask for I0 channel
        self.i0chan=Pmw.OptionMenu(f,labelpos='w',label_text='I0 Channel',labelmargin=4,items=self.mapdata.labels,menubutton_width=15)
        self.i0chan.pack(side=tkinter.TOP,padx=2,pady=5)
        if wd:
            if 'I0' in self.mapdata.labels:
                self.i0chan.invoke('I0')
            if 'I0STRM' in self.mapdata.labels:
                self.i0chan.invoke('I0STRM')
            self.dotheI0(0)
        #correction active or not
        cb=tkinter.Checkbutton(f,text="Do I0 correction?",variable=self.doI0c,anchor=tkinter.W)
        cb.pack(side=tkinter.TOP,padx=2,pady=5)

    def dotheI0(self,result):
        if self.i0chandialogexist:
            self.i0chandialog.withdraw()
        if self.TIMEchandialogexist:
            self.TIMEchandialog.withdraw()
        self.domapimage()
        
    def setTIMEchannel(self,wd=0):
        #apply i0channel dialog...
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if wd:
            if self.TIMEchandialogexist:
                self.TIMEchan.setitems(self.mapdata.labels)
                if 'TIME' in self.mapdata.labels:
                    self.TIMEchan.invoke('TIME')
                self.dotheI0(0)
                return
        if self.TIMEchandialogexist:
            self.TIMEchan.setitems(self.mapdata.labels)
            self.TIMEchandialog.show()
            return
        self.TIMEchandialogexist=1
        self.TIMEchandialog=Pmw.Dialog(self.imgwin,title='TIME Channel Selection',buttons=('OK','Cancel'),
                                          defaultbutton='OK',command=self.dotheI0)
        h=self.TIMEchandialog.interior()
        f=tkinter.Frame(h)
        f.pack(side=tkinter.TOP,fill='both')
        #dialog for:
        #ask for I0 channel
        self.TIMEchan=Pmw.OptionMenu(f,labelpos='w',label_text='TIME Channel',labelmargin=4,items=self.mapdata.labels,menubutton_width=15)
        self.TIMEchan.pack(side=tkinter.TOP,padx=2,pady=5)
        if wd:
            if 'TIME' in self.mapdata.labels:
                self.TIMEchan.invoke('TIME')
            self.dotheI0(0)
        #correction active or not

#################################  Tomography routines

    def showCToptions(self):
        #method='BP',acc_values=0,air_values=10,auto_center=0,center=0,fluo=0,rings=0,ring_width=9,filter_name='SHEPP_LOGAN'
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.CTdialogexist:
            self.CTdialog.show()
            return
        self.CTdialogexist=1
        self.CTdialog=Pmw.Dialog(self.imgwin,title='CT calculation Options',buttons=('Done','Compute'),
                                          defaultbutton='Done',command=self.CToptsdone)
        h=self.CTdialog.interior()
        h.configure(background='#d4d0c8')
        g1=Pmw.Group(h,tag_text='Calculation Mode',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')
        #Calc Type
        self.CTcalc=Pmw.RadioSelect(g1.interior(),labelpos=tkinter.W,command=self.CTcalcchange,buttontype='radiobutton',label_text='Calculation Method: ',hull_background='#d4d0c8',label_background='#d4d0c8',frame_background='#d4d0c8')
        self.CTcalc.add('Backprojection',background='#d4d0c8')
        self.CTcalc.add('Fourier',background='#d4d0c8')        
        if self.CTcalctype=='FT': self.CTcalc.setvalue('Fourier')
        else: self.CTcalc.setvalue('Backprojection')
        self.CTcalc.pack(side=tkinter.TOP,padx=2,pady=2)
        #Auto/Fluo/Trans
        self.CTopttype=Pmw.RadioSelect(g1.interior(),labelpos=tkinter.W,command=self.CTopttypechange,buttontype='radiobutton',label_text='Scan Type: ',hull_background='#d4d0c8',label_background='#d4d0c8',frame_background='#d4d0c8')
        self.CTopttype.add('AutoDetect',background='#d4d0c8')
        self.CTopttype.add('Fluorescence',background='#d4d0c8')
        self.CTopttype.add('Transmission',background='#d4d0c8')
        if self.CTtypevar==0: self.CTopttype.setvalue('AutoDetect')
        elif self.CTtypevar==1: self.CTopttype.setvalue('Fluorescence')
        else: self.CTtype.setvalue('Transmission')
        self.CTopttype.pack(side=tkinter.TOP,padx=2,pady=2)
        g1=Pmw.Group(h,tag_text='Pixel Correction',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        #Acc values
        self.CToptacc=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Acceration pixels: ',validate='numeric',entry_width=10,command=self.CTchangeacc,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.CToptacc.pack(side=tkinter.LEFT,padx=2,pady=2,anchor=tkinter.W)
        self.CToptacc.setvalue(self.CTaccvaluevar)
        self.CToptbot=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Bottom pixels: ',validate='numeric',entry_width=10,command=self.CTchangebot,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.CToptbot.pack(side=tkinter.LEFT,padx=2,pady=2,anchor=tkinter.W)
        self.CToptbot.setvalue(self.CTbotvaluevar)
        self.CTopttop=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Top pixels: ',validate='numeric',entry_width=10,command=self.CTchangetop,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.CTopttop.pack(side=tkinter.LEFT,padx=2,pady=2,anchor=tkinter.W)
        self.CTopttop.setvalue(self.CTtopvaluevar)
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')        
        
        g1=Pmw.Group(h,tag_text='Reference Correction',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')        
        #Air values
        self.CToptair=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Air pixels: ',validate='numeric',entry_width=10,command=self.CTchangeair,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.CToptair.pack(side=tkinter.TOP,padx=2,pady=2,anchor=tkinter.W)
        self.CToptair.setvalue(self.CTairvaluevar)
        g1=Pmw.Group(h,tag_text='Artifact Correction',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')             
        #Rings
        self.CTring=Pmw.RadioSelect(g1.interior(),labelpos=tkinter.W,command=self.CTringchange,buttontype='radiobutton',label_text='Remove Rings: ',label_background='#d4d0c8',hull_background='#d4d0c8',frame_background='#d4d0c8')
        self.CTring.add('No',background='#d4d0c8')
        self.CTring.add('Yes',background='#d4d0c8')
        if self.CTringvar==0: self.CTring.setvalue('No')
        else: self.CTringvar.setvalue('Yes')        
        self.CTring.pack(side=tkinter.TOP,padx=2,pady=2)
        #width
        self.CToptrw=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Ring Width: ',validate='numeric',entry_width=10,command=self.CTchangerwid,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.CToptrw.pack(side=tkinter.TOP,padx=2,pady=2,anchor=tkinter.W)
        self.CToptrw.setvalue(self.CTringwidthvar)
        #centering
        g1=Pmw.Group(h,tag_text='Center Correction',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')             
        #Correct?
        self.CTdocent=Pmw.RadioSelect(g1.interior(),labelpos=tkinter.W,command=self.CTcentchange,buttontype='radiobutton',label_text='Correct Center: ',label_background='#d4d0c8',hull_background='#d4d0c8',frame_background='#d4d0c8')
        self.CTdocent.add('No',background='#d4d0c8')
        self.CTdocent.add('Auto',background='#d4d0c8')
        self.CTdocent.add('Manual',background='#d4d0c8')
        if self.CTdocentvar==0: self.CTdocent.setvalue('No')
        elif self.CTdocentvar==1: self.CTdocent.setvalue('Auto')        
        else: self.CTdocent.setvalue('Manual')
        self.CTdocent.pack(side=tkinter.TOP,padx=2,pady=2)
        #width
        self.CToptcp=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Absolute Center: ',validate='real',entry_width=10,command=self.CTchangecentp,label_background='#d4d0c8',hull_background='#d4d0c8')
        self.CToptcp.pack(side=tkinter.TOP,padx=2,pady=2,anchor=tkinter.W)
        self.CToptcp.setvalue(self.CTcentpixvar)        
        #filter
        g1=Pmw.Group(h,tag_text='Filtering',tag_background='#d4d0c8',ring_background='#d4d0c8',hull_background='#d4d0c8')
        g1.interior().configure(background='#d4d0c8')
        g1.pack(side=tkinter.TOP,padx=5,pady=5,expand='yes',fill='both')  
        filters=('Shepp_Logan','Gen_Hamming','LP_Cosine','Ramlak','None')
        self.CTfilters=Pmw.ComboBox(g1.interior(),label_text='CT Filter',hull_background='#d4d0c8',label_background='#d4d0c8',labelpos='n',selectioncommand=self.CTfilterchange,listheight=180,scrolledlist_items=filters,history=0)
        self.CTfilters.setvalue(self.CTfiltervar)
        self.CTfilters.setentry(self.CTfiltervar)      
        self.CTfilters.pack(side=tkinter.LEFT,padx=5,pady=5)
        self.CTfilters.component('entry').configure(state=tkinter.DISABLED)
        #filtwidth
        self.CToptfw=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Filter Width: ',validate='numeric',entry_width=10,command=self.CTchangefwid,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.CToptfw.pack(side=tkinter.LEFT,padx=5,pady=5,anchor=tkinter.W)
        self.CToptfw.setvalue(self.CTfiltwidth)
        #filtd
        self.CToptfd=Pmw.EntryField(g1.interior(),labelpos='w',label_text='Filter D: ',validate='real',entry_width=10,command=self.CTchangefd,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.CToptfd.pack(side=tkinter.LEFT,padx=5,pady=5,anchor=tkinter.W)
        self.CToptfd.setvalue(self.CTfiltd)
        
        Pmw.alignlabels([self.CToptrw,self.CToptair,self.CToptacc])
        self.CTdialog.show()
        
    def CTopttypechange(self,*args):
        if self.CTopttype.getvalue()=='AutoDetect':self.CTtypevar=0
        elif self.CTopttype.getvalue()=='Fluorescence':self.CTtypevar=1
        else: self.CTtypevar=2

    def CTringchange(self,*args):
        if self.CTring.getvalue()=='No':self.CTringvar=0
        else: self.CTringvar=1

    def CTcalcchange(self,*args):
        if self.CTcalc.getvalue()=='Fourier':self.CTcalctype='FT'
        else: self.CTcalctype='BP'

    def CTfilterchange(self,*args):
        self.CTfiltervar=self.CTfilters.get()

    def CTchangeacc(self,*args):
        self.CTaccvaluevar=int(self.CToptacc.getvalue())

    def CTchangetop(self,*args):
        self.CTtopvaluevar=int(self.CTopttop.getvalue())

    def CTchangebot(self,*args):
        self.CTbotvaluevar=int(self.CToptbot.getvalue())        

    def CTchangeair(self,*args):
        self.CTairvaluevar=int(self.CToptair.getvalue())

    def CTchangerwid(self,*args):
        self.CTringwidthvar=int(self.CToptrw.getvalue())

    def CTcentchange(self,*args):
        if self.CTdocent.getvalue()=='No':self.CTdocentvar=0
        elif self.CTdocent.getvalue()=='Auto':self.CTdocentvar=1
        else: self.CTdocentvar=2

    def CTchangecentp(self,*args):
        self.CTcentpixvar=float(self.CToptcp.getvalue())

    def CTchangefwid(self,*args):
        self.CTfiltwidth=int(self.CToptfw.getvalue())

    def CTchangefd(self,*args):
        self.CTfiltd=float(self.CToptfd.getvalue())
        
    def CToptsdone(self,result):
        if result=='Compute':
            self.doCTsection()
        else:
            self.CTdialog.withdraw()
    
    def doCTsection(self,datind=-1):
        print("CT not supported")
        pass
        
        """
        if self.datachan.get()==():
            return
        self.imgwin.update()
        #update vars
        if self.CTdialogexist:
            self.CTchangeacc()
            self.CTchangetop()            
            self.CTchangebot()
            self.CTchangeair()
            self.CTchangerwid()
            self.CTchangefwid()
            self.CTchangefd()
            self.CTcentchange()
            self.CTchangecentp()
        globalfuncs.setstatus(self.status,"CALCULATING...")
        if datind==-1:
            datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        datlab=self.mapdata.labels[datind-2]
        pic=self.mapdata.data.get(datind)[::-1,:]#[::-1,:,datind]
        print(pic.shape)
        pic=pic[self.CTtopvaluevar:pic.shape[0]-self.CTbotvaluevar,:]
        print(pic.shape)
        #mi=self.mapdata.mapindex[::-1,:]
        pic=transpose(pic)
        #mi=transpose(mi)
        ang=self.mapdata.yvals[self.CTtopvaluevar:len(self.mapdata.yvals)-self.CTbotvaluevar]
        print(len(ang))
##        if len(self.mask.mask)!=0:
##            picmsk=transpose(self.mask.mask[::-1,:])
##        else:
##            picmsk=[]
##        if self.dodt.get()==1:
##            #DT: corFF=FF*exp(tau*1e-6*ICR)
##            icr=self.mapdata.data[::-1,:,self.DTICRchanval]
##            dtcor=exp(float(self.deadtimevalue.getvalue())*1e-6*icr)
##            pic=pic*dtcor
##        self.maindisp.placeData(transpose(pic),transpose(mi),self.status,xax=self.mapdata.xvals,yax=self.mapdata.yvals,domask=self.usemaskinimage,mask=picmsk,datlab=datlab)
##        self.showmap()        
        
        if self.CTtypevar==0:
            fluo=1
            if datlab in ['CH2','CH3','I1','I2']: fluo=0
        elif self.CTtypevar==1:
            fluo=1
        else:
            fluo=0

        if self.CTdocentvar==0:
            ac=0
            cnt=0
        elif self.CTdocentvar==1:
            ac=1
            cnt=0
        else:
            ac=0
            cnt=self.CTcentpixvar
        ct=TomoRecon.do_recon(pic,ang,fluo=fluo,method=self.CTcalctype,auto_center=ac,center=cnt,acc_values=self.CTaccvaluevar,air_values=self.CTairvaluevar,rings=self.CTringvar,ring_width=self.CTringwidthvar,filter_name=self.CTfiltervar,filter_width=self.CTfiltwidth,filter_d=self.CTfiltd)
        if self.CTcalctype=='BP' and self.CTfiltervar not in ['Shepp_Logan','None']: ct=-ct
        self.CTdispdict[datlab]=ct
        xst=abs(self.mapdata.xvals[0]-self.mapdata.xvals[1])
        xr=np.arange(ct.shape[0],dtype=np.float32)*xst
        newdisp=Display.Display(self.imgwin,self.viewMCAplottoggle,self.startimgROI,self.showscalebar,self.showscalebarText,self.xyflip,main=0,proc=preprocess,callback=[self.closeCTdisp,datlab],sf=self.dispScaleFactor)
        newdisp.placeData(ct,ct,self.status,xax=xr,yax=xr,domask=0,mask=[],datlab='CT '+datlab)
        #ImRadon.imshow(c)        
        globalfuncs.setstatus(self.status,"Ready")
        """
    def closeCTdisp(self,name):
        try:
            del(self.CTdispdict[name])
        except:
            pass
    
    def saveCTsections(self):
        if len(self.CTdispdict)==0:
            print('No sections to save')
            return
        #assemble data
##        (ctxpts,ctypts)=self.CTdispdict[self.CTdispdict.keys()[0]].shape
##        print ctxpts,ctypts
##        for i in self.CTdispdict.keys():
##            if self.CTdispdict[i].shape!=(ctxpts,ctypts):
##                print 'CT sections not all same size!',self.CTdispdict[i].shape
##                #think about padding data to largest...
##                return
        #find max sizes
        (ctxpts,ctypts)=(0,0)
        for i in list(self.CTdispdict.keys()):
            (a,b)=self.CTdispdict[i].shape
            if a>ctxpts: ctxpts=a
            if b>ctypts: ctypts=b
        print('max:',ctxpts,ctypts)
        #buffer if needed
        for i in list(self.CTdispdict.keys()):
            if self.CTdispdict[i].shape!=(ctxpts,ctypts):
                #pad
                padx=ctxpts-self.CTdispdict[i].shape[0]
                pady=ctypts-self.CTdispdict[i].shape[1]
                zx=zeros((padx,self.CTdispdict[i].shape[1]),dtype=np.float32)
                zy=zeros((ctxpts,pady),dtype=np.float32)
                old=self.CTdispdict[i]
                new=np.concatenate((zx,old),axis=0)
                new=np.concatenate((new,zy),axis=1)
                self.CTdispdict[i]=new
        ctchans=len(list(self.CTdispdict.keys()))
        xst=abs(self.mapdata.xvals[0]-self.mapdata.xvals[1])
        ctvals=np.arange(ctxpts,dtype=np.float32)*xst
        ctdata=zeros((ctxpts,ctypts,ctchans+2),dtype=np.float32)
        ctlabs=[]
        for i in range(ctxpts):
            for j in range(ctypts):
                ctdata[i,j,0]=ctvals[i]
                ctdata[i,j,1]=ctvals[j]
        di=2
        for n in list(self.CTdispdict.keys()):
            ctlabs.append(n)
            ctdata[:,:,di]=transpose(self.CTdispdict[n][:,::-1])
            di+=1
        #get file name
        if self.dataFileBuffer[self.activeFileBuffer]['fname'].rfind('_CT.dat')==-1:
            fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_CT.dat'
        else:
            fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'.dat'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        #save data in SCANE ascii format
        globalfuncs.setstatus(self.status,'Saving CT data...')
        fid=open(fn,"w")
        fid.write('* Abscissa points :   '+str(ctxpts)+'\n')
        fid.write('* Ordinate points :   '+str(ctypts)+'\n')
        globalfuncs.writeblankline(fid,1)
        fid.write('* Data Channels :   '+str(ctchans)+'\n')
        labtemp='# Data Labels : '
        for i in range(len(ctlabs)):
            labtemp=labtemp+ctlabs[i]+'\t'
        fid.write(labtemp+'\n')
        fid.write('* Comments: \n')
        #limit to 3 lines of data:
        temp=self.mapdata.comments.split('\n')
        if len(temp)>3: temp=temp[0:3]
        if len(temp)<3: temp.append('')
        if len(temp)<3: temp.append('')
        if len(temp)<3: temp.append('')
        new=''
        for t in temp:
            new=new+t+'\n'
        fid.write(new)
        globalfuncs.writeblankline(fid,1)
        fid.write('* Abscissa points requested :\n* ')
        #write xpoints
        for x in ctvals:
            fid.write(str(x)+'\t')
        fid.write('\n')
        globalfuncs.writeblankline(fid,2)        
        fid.write('* Ordinate points requested :\n* ')
        #write ypoints
        for y in ctvals:
            fid.write(str(y)+'\t')
        fid.write('\n')
        globalfuncs.writeblankline(fid,2)
        fid.write("* Energy points requested: \n*   "+str(1)+'\n')
        globalfuncs.writeblankline(fid,1)
        fid.write('* DATA\n')
        #start data block
        (xlen,ylen)=ctdata.shape[:2]
        for i in range(xlen):
            for j in range(ylen):
                for k in range(ctdata.shape[2]):            
                    fid.write(str(ctdata[i,j,k])+'\t')
                fid.write('\n')          
        fid.close()
        globalfuncs.setstatus(self.status,"Processed CT data saved in: "+fn)
        

#################################  ROI callback

    def startimgROI(self,pixels,maskpts,multi=0,mcadir=0):
        self.exportedROImask=maskpts
        self.exportedROIpixels=pixels
        if mcadir:
            self.showMCApix(pixels)
            return
        if multi==2:
            if self.MCAfilename=='':
                result='Mask'
            else:
                result='MCA and Mask'
            self.processNewMaskInfo(result)
            return
        if multi==3:
            self.processNewMaskInfo('Mask')
            return
        #ask for option -- mca, ROI mask, both
        self.ROIselectdialog=Pmw.SelectionDialog(self.imgwin,title='ROI Selection Action',buttons=('OK','Cancel'),defaultbutton='OK',
                                                 scrolledlist_items=('MCA','Mask','MCA and Mask'),
                                                 command=self.ROIselectchoice)
        self.ROIselectdialog.show()

    def ROIselectchoice(self,result):
        if result=='Cancel':
            self.ROIselectdialog.withdraw()
            return
        if self.ROIselectdialog.getcurselection()==(): return
        a=self.ROIselectdialog.getcurselection()[0]
        self.ROIselectdialog.withdraw()
        self.processNewMaskInfo(a)

    def processNewMaskInfo(self,a):
        if a.rfind('Mask')!=-1:
            self.clearcpmask()
            self.maindisp.masktype=self.maindisp.MaskDrawType.get()
            self.mask.mask=zeros((self.mapdata.data.shape[0],self.mapdata.data.shape[1]),dtype=np.float32)
            for pt in self.exportedROImask:
                v=self.mapdata.data.shape[0]-pt[1]
                if v<0: v=0
                if v>(self.mapdata.data.shape[0]-1): v=self.mapdata.data.shape[0]-1
                self.mask.mask[v,pt[0]]=1
            if not self.maindisp.showMaskROI.get():
                self.usemask()
            else:
                self.ignoremask()
            print('mask.maskNEW',np.sum(np.ravel(self.mask.mask)))
            self.tcrefresh()
        if a.rfind('MCA')!=-1:
            #print self.exportedROIpixels
            self.showMCApix(self.exportedROIpixels,multi=1)        

    def getMCAfrommask(self):
        #check mask exists?
        if len(self.mask.mask)==0:
            print("no mask in memory")
            globalfuncs.setstatus(self.status,"no mask to get MCA...")
            return
        ep=np.where(np.ravel(self.mask.mask)==1)[0]
        #print ep
        self.showMCApix(ep,multi=1)
        globalfuncs.setstatus(self.status,"MCA from mask done!")
        
    def getMCAfromcluster(self):
        #check mask exists?
        i=self.mapdata.labels.index(self.datachan.getvalue()[0])
        fd=self.mapdata.data.get(i+2)
        ifd=ravel(fd)
        ifd=np.where(ifd==ifd)[0]
        ifd=np.reshape(ifd,fd.shape)
        if self.maindisp.zmxyi[2]!=-1 and self.maindisp.zmxyi[3]!=-1:
            fd=fd[::-1,:]
            fd=fd[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            fd=fd[::-1,:]
            ifd=ifd[::-1,:]
            ifd=ifd[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            ifd=ifd[::-1,:]
        fd=ravel(fd)
        ifd=ravel(ifd)
        u=1
        m=max(fd)
        b=np.mod(fd,1)
        if m>35: u=0
        if sum(abs(b))>0: u=0
        if len(np.where(self.mapdata.data.get(i+2)<0)[0])>1: u=0
        if u==0:
            print('Not valid cluster')
            globalfuncs.setstatus(self.status,"Not valid cluster...")
            return
        if m>8:
            print('More than 9 clusters, showing first 9...')
            globalfuncs.setstatus(self.status,"More than 9 clusters, showing first 9...")
            m=8
        self.clearMCABuffers()
        if not self.MCAviewexist:
            self.updateMCAgraph(plot=False)            
        l=[self.MCAplotBufferSwitch1,self.MCAplotBufferSwitch2,self.MCAplotBufferSwitch3,self.MCAplotBufferSwitch4,
           self.MCAplotBufferSwitch5,self.MCAplotBufferSwitch6,self.MCAplotBufferSwitch7,self.MCAplotBufferSwitch8,
           self.MCAplotBufferSwitch9]

        for i in range(int(m)+1):   
            l[i].set(1)
            self.setActiveMCABuffer(i,up=False)
            ep=np.where(fd==i)[0]
            epi=ifd[ep]
            print(i,len(epi))
            self.showMCApix(epi,multi=1)
        self.MCAreplotBuffers()
        globalfuncs.setstatus(self.status,"MCA from clusters done!")
        
#################################  Save data routines


    def clipboardexport(self,graph,header,type=None):
        text=''
        datay=[]
        datay1=[]
        datay2=[]
        datax=[]
        if type is None:
            temp=graph.get_xdata('data')
        else:
            temp=graph.get_xdata(type)
        datax.append(temp)
        if type is None:
            temp=graph.get_ydata('data')
        else:
            temp=graph.get_ydata(type)
        datay.append(temp)
        if type=='MCA':
            temp=graph.get_ydata('MCAb')
            if temp is not None:
                datay2.append(temp)
            temp=graph.get_ydata('MCAf')
            if temp is not None:
                datay1.append(temp)
        text=text+header+'\n'
        try:
            datax=datax[0].split()
            datay=datay[0].split()
            if datay1!=[]: datay1=datay1[0].split()
            if datay2!=[]: datay2=datay2[0].split()
        except:
            datax=datax[0]
            datay=datay[0]
            if datay1!=[]: datay1=datay1[0]
            if datay2!=[]: datay2=datay2[0]
        #parse list now
        for i in range(len(datax)):
            #setup text
            if len(datay1)>0: 
                ty1=str(datay1[i])+'\t'
            else:
                ty1=''
            if len(datay2)>0: 
                ty2=str(datay2[i])+'\t'
                ty3=str(float(datay[i])-float(datay2[i]))+'\t'
            else:
                ty2=''
                ty3=''
                    
            text=text+str(datax[i])+'\t'+str(datay[i])+'\t'+ty1+ty2+ty3+'\t\n'
        #export to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        globalfuncs.setstatus(self.status,header+" saved to clipboard")

    def saveprocdata(self):
        #save processed data files
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #get file name
        deffn=self.dataFileBuffer[self.activeFileBuffer]['fname']
        if deffn.rfind('_process.dat')==-1:
            fn=globalfuncs.trimdirext(deffn)+'_process.dat'
        else:
            fn=globalfuncs.trimdirext(deffn)+'.dat'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        svn=''
        if self.defaultSaveType.get()=='ASCII' or self.defaultSaveType.get()=='Both':
            #save data in SCANE ascii format
            globalfuncs.setstatus(self.status,'Saving processed data...')
            fid=open(fn,"w")
            fid.write('* Abscissa points :   '+str(self.mapdata.nxpts)+'\n')
            fid.write('* Ordinate points :   '+str(self.mapdata.nypts)+'\n')
            globalfuncs.writeblankline(fid,1)
            fid.write('* Data Channels :   '+str(self.mapdata.channels)+'\n')
            labtemp='# Data Labels : '
            for i in range(len(self.mapdata.labels)):
                labtemp=labtemp+self.mapdata.labels[i]+'\t'
            fid.write(labtemp+'\n')
            fid.write('* Comments: \n')
            #limit to 3 lines of data:
            temp=self.mapdata.comments.split('\n')
            if len(temp)>3: temp=temp[0:3]
            if len(temp)<3: temp.append('')
            if len(temp)<3: temp.append('')
            if len(temp)<3: temp.append('')
            new=''
            for t in temp:
                new=new+t+'\n'
            fid.write(new)
            globalfuncs.writeblankline(fid,1)
            fid.write('* Abscissa points requested :\n* ')
            #write xpoints
            for x in self.mapdata.xvals:
                fid.write(str(x)+'\t')
            fid.write('\n')
            globalfuncs.writeblankline(fid,2)        
            fid.write('* Ordinate points requested :\n* ')
            #write ypoints
            for y in self.mapdata.yvals:
                fid.write(str(y)+'\t')
            fid.write('\n')
            globalfuncs.writeblankline(fid,2)
            fid.write("* Energy points requested: \n*   "+str(self.mapdata.energy)+'\n')
            globalfuncs.writeblankline(fid,1)
            fid.write('* DATA\n')
            #start data block
            (xlen,ylen)=self.mapdata.data.shape[:2]
            #FIXMAPDATA
            for i in range(xlen):
                for j in range(ylen):
                    dcol=self.mapdata.data.getPix(i,j)
                    for k in range(self.mapdata.data.shape[2]):            
                        fid.write(str(dcol[k])+'\t')
                    fid.write('\n')          
            fid.close()
            svn=fn+" "
        if self.defaultSaveType.get()=='HDF5' or self.defaultSaveType.get()=='Both':
            #save HDF by default too
            if self.mapdata.hasHDF5:
                self.mapdata.hdf5group.attrs.create("channels",self.mapdata.channels)
                self.mapdata.hdf5group.attrs.create("labels",self.mapdata.labels)
                self.mapdata.hdf5.flush()
                hdffn=os.path.splitext(fn)[0]+".hdf5"
                shutil.copy(self.mapdata.hdf5.filename,hdffn)
                print("hdf5 saved")
                svn+=hdffn
            
        self.changes=0
        globalfuncs.setstatus(self.status,"Processed data saved in: "+svn)



    def saveMCAdata(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if len(self.MCArawdata)==0 or not self.MCAviewexist:
            print('No MCA data')
            globalfuncs.setstatus(self.status,'No MCA Data')
        #ask for file
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_MCA.mca'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving MCA spectrum data...")
        #save it
        fid=open(fn,'w')
        fid.write('MCA data from file: '+self.dataFileBuffer[self.activeFileBuffer]['fname']+'\n')
        #i=0
        l=[self.MCAplotBufferSwitch1,self.MCAplotBufferSwitch2,self.MCAplotBufferSwitch3,self.MCAplotBufferSwitch4,
           self.MCAplotBufferSwitch5,self.MCAplotBufferSwitch6,self.MCAplotBufferSwitch7,self.MCAplotBufferSwitch8,
           self.MCAplotBufferSwitch9]
        for i in range(len(self.MCArawdata)):
            fid.write(str(i)+'\t')
            for j in range(9):
                fid.write(self.MCArawdataBuffer[i][j]+'\t')
            fid.write('\n')
        fid.close()        
        globalfuncs.setstatus(self.status,"MCA data saved in: "+fn)        
        
    def savedisplayasjpg(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_'+str(self.datachan.getvalue()[0])+'.jpg'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving image display...")
        self.maindisp.savejpgimage(fn)
        globalfuncs.setstatus(self.status,"Image display saved in: "+fn)

    def savedisplayasHD(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_'+str(self.datachan.getvalue()[0])+'.tiff'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving HR image display...")
        e=self.maindisp.saveHDimage(fn)
        if e:
            globalfuncs.setstatus(self.status,"Image save ERROR")
        else:
            globalfuncs.setstatus(self.status,"Image display saved in: "+fn)

    def savedisplayasNPY(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_'+str(self.datachan.getvalue()[0])+'.npy'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving image display as array...")
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        np.save(fn,self.mapdata.data.get(datind))
        globalfuncs.setstatus(self.status,"Image display array saved in: "+fn)
        

    def asksavemanydisplays(self):
        # save multiple channels
        globalfuncs.setstatus(self.status, "Ready")
        if not self.hasdata:
            print()
            'No Data'
            globalfuncs.setstatus(self.status, 'No Data')
            return
        self.multisavedialog=Pmw.SelectionDialog(self.imgwin, title="Save Multiple Displays",
                                                     buttons=('Standard','HighDef', 'Cancel'), defaultbutton='Cancel',
                                                     scrolledlist_labelpos='n', label_text='Select Channels',
                                                     scrolledlist_items=self.mapdata.labels,
                                                     command=self.savemultichannel)
        self.multisavedialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)

    def savemultichannel(self, result):
        goner=self.multisavedialog.getcurselection()
        self.multisavedialog.withdraw()
        if result=='Cancel':
            globalfuncs.setstatus(self.status, 'No displays saved')
            return
        fb=self.dataFileBuffer[self.activeFileBuffer]['fname']
        for going in goner:
            datind=self.mapdata.labels.index(going) + 2
            self.domapimage(datind=datind)
            if result=='Standard':
                fn=fb + '_MD_' + going + '.jpg'
                self.maindisp.savejpgimage(fn)
                globalfuncs.setstatus(self.status, "Image display saved in: " + fn)
            if result=='HighDef':
                fn=fb + '_MD_' + going + '.tiff'
                e=self.maindisp.saveHDimage(fn)
                if e:
                    globalfuncs.setstatus(self.status,"Image save ERROR")
                else:
                    globalfuncs.setstatus(self.status,"Image display saved in: "+fn)



    def savetcdisplayasjpg(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.triColorWindow.tcimageexists:
            print('No tricolor plot')
            globalfuncs.setstatus(self.status,'No tricolor plot')
            return
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_TC.jpg'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving tricolor image display...")
        #save image
        self.triColorWindow.tcimwin.lift()
        self.imgwin.update()

        if not self.showscalebar.get():
            rx=int(self.triColorWindow.tcimframe.winfo_rootx())
            ry=int(self.triColorWindow.tcimframe.winfo_rooty())
            rw=int(self.triColorWindow.tcimframe.winfo_width())
            rh=int(self.triColorWindow.tcimframe.winfo_height())
            screencapture.capture(rx,ry,rw,rh,fn)
            #im=ImageGrab.grab((rx,ry,rx+rw,ry+rh))
            #im.save(fn)                            
        else:
            rx=int(self.triColorWindow.tcimframe.winfo_rootx())
            ry=int(self.triColorWindow.tcimframe.winfo_rooty())
            rw=int(self.triColorWindow.tcimframe.winfo_width())
            rh=int(self.triColorWindow.tcimframe.winfo_height())
            screencapture.capture(rx,ry,rw,rh,fn)
            #im=ImageGrab.grab((rx,ry,rx+rw,ry+rh))
            #im.save(fn)            
        globalfuncs.setstatus(self.status,"Tricolor image display saved in: "+fn)

    def savetcdisplayasHD(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.triColorWindow.tcimageexists:
            print('No tricolor plot')
            globalfuncs.setstatus(self.status,'No tricolor plot')
            return
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_TC.tiff'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving HD tricolor image display...")
        try:
            ##self.tcppm.save(fn)
            pilim=Image.open(Display.save_ppm(self.tcppm))
            pilim.save(fn)
        except:
            globalfuncs.setstatus(self.status,"Image save ERROR")
            return
        globalfuncs.setstatus(self.status,"Tricolor image display saved in: "+fn)
        
    def savexyplotdata(self):
        globalfuncs.setstatus(self.status,"Ready")
        text=''
        text2=''
        text3=''
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for data to exist:
        if not self.maindisp.linegraphpresent:
            print('No XY plot')
            globalfuncs.setstatus(self.status,'No XY plot')
        else:
            globalfuncs.setstatus(self.status,"saving XY data to clipboard...")
            text=self.maindisp.savexyplot()
        if not self.maindisp.linegraph2present:
            print('No XS plot')
        else:
            globalfuncs.setstatus(self.status,"saving xsection data to clipboard...")
            text2=self.maindisp.savexyplotxs()
        if not self.maindisp.linegraph3present:
            print('No vector plot')
        else:
            globalfuncs.setstatus(self.status,"saving vector data to clipboard...")
            text3=self.maindisp.savexyplotxs3()
        if not self.maindisp.linegraph4present:
            print('No vector analysis plot')
        else:
            globalfuncs.setstatus(self.status,"saving vector analysis data to clipboard...")
            text3=self.maindisp.savexyplotxs4()

            
        ft=text+text2+text3
        if ft=='':
            return
        #export to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(ft)
        globalfuncs.setstatus(self.status,"XY data saved to clipboard")

    def exportcorplot(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #test for data to exist:
        if not self.correlationPlot.exist:
            print('No correlation plot')
            globalfuncs.setstatus(self.status,'No correlation plot')
            return
        globalfuncs.setstatus(self.status,"saving correlation data to clipboard...")
        text=self.assemblecordata()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        globalfuncs.setstatus(self.status,"Correlation data saved to clipboard")

    def exporttextdata(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        globalfuncs.setstatus(self.status,"saving data to clipboard...")
        text='Xcoord\tYcoord\t'
        for n in self.mapdata.labels:
            text=text+n+'\t'
        text=text+'\n'
        #worry about zoom and mask...
        if len(self.mask.mask)!=0 and self.usemaskinimage:
            pm=self.mask.mask[::-1,:]
        else:
            pm=ones(self.mapdata.data.get(0)[::-1,:].shape)
        if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:
            pm=pm[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
        
        for p in range(len(ravel(pm))):
            if ravel(pm)[p]==0: continue
            xdata=self.mapdata.data.get(0)[::-1,:]
            ydata=self.mapdata.data.get(1)[::-1,:]
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:
                xdata=xdata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                ydata=ydata[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            xdp=ravel(xdata)[p]
            ydp=ravel(ydata)[p]
            text=text+str(ydp)+'\t'+str(xdp)+'\t'
            for n in self.mapdata.labels:
                datind=self.mapdata.labels.index(n)+2
                data=self.mapdata.data.get(datind)[::-1,:]
                if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:
                    data=data[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                dp=ravel(data)[p]
                text=text+str(dp)+'\t'
            text=text+'\n'


        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        globalfuncs.setstatus(self.status,"Data saved to clipboard")

#################################  Open New data from MCA

    def createDataFromMCA(self):
        #if self.changes:
        #    if not tkinter.messagebox.askyesno("Changes Made", "Unsaved changes made to data! Discard?"):
        #        return
        #ask for file
        fty=[("MCA data files","*.mca"),("all files","*")]
        t=globalfuncs.ask_for_file(fty,self.filedir.get())
        if t!='':
            globalfuncs.setstatus(self.status,"LOADING data from MCAs...")
            inst=MCAImageGet.importDataRead(self.root,t)
            if inst is None:
                globalfuncs.setstatus(self.status,"Data import cancelled...")
                return
            self.mapdata=inst.impdata
        else:
            return        
        #setup init data
        self.cleanup_load_data()

#################################  Open New data from LA-MS data

    def createDataFromLAMS(self):

        wfn=1
        while True:
            if wfn in self.workingFileBufferNames:
                wfn+=1
            else:
                break
        self.workingdir.wfn=wfn
        self.workingFileBufferNames.append(wfn)        

        #ask for files 
        newmap=LAMSImageGet.GetData(self.root,self.createDataFromLAMSCallback)
        newmap.load_data_file(workdir=self.workingdir,filedir=self.filedir)
        #newmap.cleanString()
        globalfuncs.setstatus(self.status,"LOADING...")


    def createDataFromLAMSCallback(self,newmap):
        if newmap.complete is False: 
            print("load failed")
            globalfuncs.setstatus(self.status,"LOADING... failed")
            return
        
        self.mapdata=newmap.mapdata
        self.filedir.set(os.path.dirname(newmap.fn))
        shortfn=os.path.splitext(os.path.basename(newmap.fn))[0]
        shortfn=shortfn.replace('_','-')
        if shortfn in self.dataFileBuffer:
            i=1
            while True:
                if shortfn+'-'+str(i) in self.dataFileBuffer:
                    i+=1
                else:
                    shortfn=shortfn+'-'+str(i)
                    break                
        self.activeFileBuffer=shortfn

        #setup init data
        self.cleanup_load_data()

#################################  Import data from XRD data files

    def importXRD(self):
        globalfuncs.setstatus(self.status,"Ready")
        #check for data
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        #ask for files

        infile=globalfuncs.ask_for_file([("dat files","*.dat"),("all files","*")],os.getcwd(),multi=True)
        multfn=self.root.tk.splitlist(infile)

        if multfn==():
            print('XRD load cancelled')
            globalfuncs.setstatus(self.status,'XRD load cancelled')
            return
        multfn=list(multfn)
        multfn.sort()
        numpts=len(multfn)
        #assemble file list
        self.xrdfileimp=[]
        for sel in multfn:
            #curfil=multdlist[int(sel)]
            #fulpath=curdir+os.sep+curfil
            self.xrdfileimp.append(sel)#(fulpath)
        #get pixels per xrd
        pxpts=''
        pypts=''
        ok=0
        while not ok:
            pxpts=tkinter.simpledialog.askinteger(title='Import XRD',prompt='Enter number of x pixels in XRD map: ',initialvalue=pxpts)
            if pxpts=='' or pxpts==0:
                print('XRD load cancelled')
                globalfuncs.setstatus(self.status,'XRD load cancelled')
                return
            pypts=tkinter.simpledialog.askinteger(title='Import XRD',prompt='Enter number of y pixels in XRD map: ',initialvalue=pypts)
            if pypts=='' or pypts==0:
                print('XRD load cancelled')
                globalfuncs.setstatus(self.status,'XRD load cancelled')
                return
            #check validity
            okvalid=1
            if self.mapdata.nxpts%pxpts!=0:
                if not tkinter.messagebox.askokcancel('XRD Scale','Number X XRD pts not\n integral number of data points?'):
                    okvalid=0
            if self.mapdata.nypts%pypts!=0:
                if not tkinter.messagebox.askokcancel('XRD Scale','Number Y XRD pts not\n integral number of data points?'):
                    okvalid=0
            if okvalid and pxpts*pypts!=numpts:
                if not tkinter.messagebox.askokcancel('XRD Scale','Number X and Y XRD pts does\n not equal number of XRD data points?'):
                    okvalid=0                
              
            if okvalid: ok=1
        self.xrdscale=[numpts,pxpts,pypts]
        print(self.xrdscale,self.mapdata.nxpts,self.mapdata.nypts)
        #setup dialog for integrations
        self.xrdintegdialog=Pmw.Dialog(self.root,title='Set XRD Integration Limits',buttons=('OK','Cancel','Add'),
                                       command=self.evalXRDints,defaultbutton=None)
        self.xrdwidlist=[]
        self.xrdwidlist.append(xrdentryobj(self.xrdintegdialog.interior()))      
        self.xrdintegdialog.show()

    def evalXRDints(self,result):
        if result=='Cancel':
            self.xrdintegdialog.withdraw()
            return                    
        if result=='Add':
            self.xrdwidlist.append(xrdentryobj(self.xrdintegdialog.interior()))
            return
        if result=='OK':
            #check for valid
            allok=1
            for w in self.xrdwidlist:
                if not (w.low.valid() and w.hi.valid()): allok=0
            if not allok:
                print('Invalid XRD integration limits')
                return
            #check names
            for w in self.xrdwidlist:
                if w.name.getvalue() in self.mapdata.labels:
                    print('Invalid name for data')
                    return
            #save limits and close dialog
            self.xrdintdict={}
            for w in self.xrdwidlist:
                v=(float(w.low.getvalue()),float(w.hi.getvalue()))
                self.xrdintdict[w.name.getvalue()]=[min(v),max(v)]
            self.xrdintegdialog.withdraw()
            self.getXRDintegrations()
        
    def getXRDintegrations(self):
        #do it!
        datadict={}
        for k in list(self.xrdintdict.keys()):
            datadict[k]=[]
        for f in self.xrdfileimp:
            temp={}
            for k in list(self.xrdintdict.keys()):
                temp[k]=0
            fid=open(f,'rU')
            lines=fid.readlines()
            fid.close()
            for l in lines:
                if l=='': continue
                if l[0]=='#': continue
                qp=float(l.split()[0])
                dp=float(l.split()[1])
                for k in list(self.xrdintdict.keys()):
                    if self.xrdintdict[k][0]<qp and qp<self.xrdintdict[k][1]:
                        temp[k]=temp[k]+dp
            for k in list(self.xrdintdict.keys()):
                datadict[k].append(temp[k])
        #now have datadict with set of lists of data...

        xax=np.arange(self.xrdscale[1])/float(self.xrdscale[1]-1)
        yax=np.arange(self.xrdscale[2])/float(self.xrdscale[2]-1)
        for k in list(self.xrdintdict.keys()):
            newdat=zeros((self.mapdata.nxpts,self.mapdata.nypts),dtype=np.float32)
            datline=datadict[k]
            if self.xrdscale[0]<self.xrdscale[1]*self.xrdscale[2]:
                temp=zeros(self.xrdscale[1]*self.xrdscale[2],dtype=np.float32)
                temp[0:self.xrdscale[0]]=datline
                fdat=np.resize(temp,(self.xrdscale[1],self.xrdscale[2]))
            elif self.xrdscale[0]>self.xrdscale[1]*self.xrdscale[2]:
                temp=datline[0:self.xrdscale[1]*self.xrdscale[2]]
                fdat=np.resize(temp,(self.xrdscale[1],self.xrdscale[2]))
            else:
                fdat=np.resize(datline,(self.xrdscale[1],self.xrdscale[2]))
            ifxn=InterpolatingFunction((xax,yax),fdat)
            for i in np.arange(self.mapdata.nxpts):
                for j in np.arange(self.mapdata.nypts):
                    x=i/float(self.mapdata.nxpts-1)
                    y=j/float(self.mapdata.nypts-1)
                    newdat[i,j]=ifxn(x,y)
            #add channel
            self.addchannel(transpose(newdat),k)
        globalfuncs.setstatus(self.status,'XRD Import complete')



################################# Quantitative Analysis

    def quantoptions(self):
        #call dialog starter with function
        self.quantgendialog(self.enterQuantSelect)

    def quantgendialog(self,function):        
        #open a new dialog (sim to xanes fit - chanel selection and then fill in for element type and cnts
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.selectquantdialog=Pmw.SelectionDialog(self.imgwin,title="Select Channels to Quantify",buttons=('OK','Cancel'),defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=function)
        self.selectquantdialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)        

    def enterQuantSelect(self,result):
        goner=self.selectquantdialog.getcurselection()
        self.selectquantdialog.withdraw()
        if result=='Cancel' or goner==():
            globalfuncs.setstatus(self.status,'No action taken')
            return
        #now make main dialog...
        self.quantentrydialog=Pmw.Dialog(self.imgwin,title="Quantitative Analysis",buttons=('Quantify','Validate','Cancel'),
                                         command=self.startQuant)
        insd=self.quantentrydialog.interior()
        insd.configure(background='#d4d0c8')
        menubar=PmwTtkMenuBar.PmwTtkMenuBar(insd)
        if os.sys.platform=='win32': menubar.component('hull').configure(bg='#d4d0c8')
        menubar.addmenu('File','')
        menubar.addmenuitem('File','command',label='Load Parameters',command=self.loadquantstdfile)
        ##menubar.addmenuitem('File','command',label='Save Parameters',command=tkinter.DISABLED)
        menubar.pack(side=tkinter.TOP,fill=tkinter.X)          
        inter=tkinter.Frame(insd,bd=2,relief=tkinter.SUNKEN,background='#d4d0c8')
        inter.pack(side=tkinter.TOP)
        #make labels on top
        f=tkinter.Frame(inter,background='#d4d0c8')
        w=12
        px=3
        for t in ['Chan Name','Element','Std Formula','Std Conc','Cts/I0','Samp I0 Gain','Std I0 Gain']:
            if len(t.split())>2: w=25
            l=tkinter.Label(f,text=t,width=w,background='#d4d0c8')
            l.pack(side=tkinter.LEFT,padx=px,fill=tkinter.X,expand=1)
        
        f.pack(side=tkinter.TOP,padx=2,pady=2)
        self.quantlist=[]
        for g in goner:
            self.quantlist.append(quantfield(g,inter))
        self.quantentrydialog.show()

    def loadquantstdfile(self):
        #get fn
        fty=[("QPM param files","*.qpm"),("all files","*")]
        fn=globalfuncs.ask_for_file(fty,self.filedir.get())
        if fn=='':
            print('Load cancelled')
            return 
        #read first line
        fid=open(fn,'r')
        l=fid.readline()
        if l!='SMAK QUANT\n':
            print('Invalid parameter file!')
            fid.close()
            return
        #read data
        lines=fid.readlines()
        fid.close()
        quantdict={}
        quantvals=[]
        for line in lines:
            if len(line)<2: continue
            if line[0]=='#': continue
            l=line.split()
            if len(l)!=6: continue
            quantvals.append(l[1])
            quantdict[l[1]]=[l[0],l[2],l[3],l[4],l[5]]
        #correlate defnames to channels
        self.chanquantdict={}
        self.quantloadcorrel(quantvals)
        #place values
        if self.chanquantdict=={}: return
        for q in self.quantlist:
            cur=quantdict[self.chanquantdict[q.name]]
            q.formula.setvalue(cur[0])
            q.element.setvalue(cur[1])
            q.conc.setvalue(float(cur[2]))
            q.cts.setvalue(float(cur[3]))
            if cur[4] in q.gainlist:
                q.stdi0gain.selectitem(cur[4],setentry=1)
                q.i0gain.selectitem(cur[4],setentry=1)

    def quantloadcorrel(self,qvs):
        self.quantCCdialog=Pmw.Dialog(self.imgwin,title='Correlate Channels',buttons=('OK','Cancel'),defaultbutton='OK',
                                      command=self.qcordone)
        inter=self.quantCCdialog.interior()
        self.qfcd={}
        for q in self.quantlist:
            cb=Pmw.ComboBox(inter,label_text=q.name,labelpos='w',history=0,scrolledlist_items=qvs,dropdown=1)
            cb.pack(side=tkinter.TOP,padx=5,pady=5)
            if q.name in qvs:
                cb.selectitem(q.name,setentry=1)
            self.qfcd[q.name]=cb
        Pmw.alignlabels(list(self.qfcd.values()))
        self.quantCCdialog.show()

    def qcordone(self,result):
        if result=='Cancel':
            print('Load cancelled')
            self.quantCCdialog.withdraw()
            return
        #check validity
        for n in list(self.qfcd.keys()):
            if self.qfcd[n].get()=='':
                print('Need all channels correlated')
                return
            else:
                self.chanquantdict[n]=self.qfcd[n].get()
        self.quantCCdialog.withdraw()        
       
    def startQuant(self,result):
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            self.quantentrydialog.withdraw()
            return
        if result=='Validate':
            rt=self.quantvalidate(1)
            if rt:
                globalfuncs.setstatus(self.status,'Quantification data valid')
        if result=='Quantify':
            rt=self.quantvalidate(0)
            if not rt:
                return
            globalfuncs.setstatus(self.status,'Doing quantitative analysis...')
            #do
            self.doQuant()
            self.quantentrydialog.withdraw()
            globalfuncs.setstatus(self.status,'Quantative analysis complete!')

    def quantvalidate(self,display):
        #check entries for validness...
        globalfuncs.setstatus(self.status,'Checking validity...')
        valid=1
        for q in self.quantlist:
            if q.formula.getvalue()=='': valid=0
            if q.element.getvalue()=='': valid=0
            if q.element.getvalue() not in list(parseFormula.sym2elt.keys()): valid=0
            if not parseFormula.parseCmd('syms',q.formula.getvalue()): valid=0
            if not q.conc.valid(): valid=0
            if not q.cts.valid(): valid=0
            if not q.std:
                if q.i0gain.get() not in q.gainlist: valid=0
            if q.stdi0gain.get() not in q.gainlist: valid=0
        if not valid:
            globalfuncs.setstatus(self.status,'Quantification data entry invalid')
            if display:
                tkinter.messagebox.showwarning("Quantification Validation","Invalid entry")
        return valid
        
    def doQuant(self):
        print('Qfy')
        #loop for each value
        for q in self.quantlist:
            #get data
            Aind=self.mapdata.labels.index(q.name)+2
            #get i0!!!!!
            iind=self.mapdata.labels.index(self.i0chan.getvalue())+2
            i0dat=self.mapdata.data.get(iind)#[:,:,iind]
            #scale if needed
            scale=float(q.stdi0gain.get())/float(q.i0gain.get())
            i0dat=i0dat*scale
            #divide by i0
            (xlen,ylen)=self.mapdata.data.shape[:2]
            newdata=zeros((xlen,ylen),dtype=np.float32)

            adata=self.mapdata.data.get(Aind)
            for i in range(xlen):
                for j in range(ylen):
                    if i0dat[i,j]!=0:
                        newdata[i,j]=adata[i,j]/i0dat[i,j]            
            #find formula pctages
            pct=parseFormula.returnWt(q.element.getvalue(),q.formula.getvalue())
            if pct==-1: #invalid formula (should never get here...)
                print('invalid formula, skipping ',q.name)
                continue                        
            #calculate
            conc=pct*float(q.conc.getvalue())
            concperct=conc/float(q.cts.getvalue())
            print(concperct)
            newdata=newdata*concperct
            #add data!
            nameroot=q.name+'-conc'
            valid=0
            i=1
            while not valid:
                newname=nameroot+str(i)
                if newname not in self.mapdata.labels:
                    valid=1
                else:
                    i+=1
            self.addchannel(newdata,newname)

    def addquantcalc(self):
        #display warning
        if not tkinter.messagebox.askyesno('Quantification','This procedure will add data to\na standard quantification file.\nProceed?'):
            return
        #ask for file to add to
        fty=[("QPM param files","*.qpm"),("all files","*")]
        fn=globalfuncs.ask_save_file('standard.qpm',self.filedir.get(),ext=fty)        
        if fn!='':
            #test extensions
            if os.path.splitext(fn)[1]=='':fn=fn+'.qpm'
            self.quantsavefn=fn
            if os.path.exists(fn):
            #if file exists:
                globalfuncs.setstatus(self.status,"LOADING intial data from file...")
                fid=open(fn,'r')
                self.quantfilestarttext=fid.read()
                fid.close()
            else:
            #else make anew...
                self.quantfilestarttext='SMAK QUANT\n#Formula\tDefaultChan\tElement\t\tConc\t\tCts/I0\t\tGain\n'
        else:
            globalfuncs.setstatus(self.status,"cancelling procedure...")
            return
        #ask for channel
        self.quantgendialog(self.enterQuantMakerSelect)
      
    def enterQuantMakerSelect(self,result):
        goner=self.selectquantdialog.getcurselection()
        self.selectquantdialog.withdraw()
        if result=='Cancel' or goner==():
            globalfuncs.setstatus(self.status,'No action taken')
            return       
        #now make main dialog...
        self.quantentrydialog=Pmw.Dialog(self.imgwin,title="Quantitative Analysis",buttons=('Save','Cancel'),
                                         command=self.saveQuant)
        insd=self.quantentrydialog.interior()
        insd.configure(background='#d4d0c8')
        inter=tkinter.Frame(insd,bd=2,relief=tkinter.SUNKEN,background='#d4d0c8')
        inter.pack(side=tkinter.TOP)
        #make labels on top
        f=tkinter.Frame(inter,background='#d4d0c8')
        w=12
        px=3
        for t in ['Chan Name','Element','Std Formula','Std Conc','Cts/I0','Std I0 Gain']:
            if len(t.split())>2: w=25
            l=tkinter.Label(f,text=t,width=w,background='#d4d0c8')
            l.pack(side=tkinter.LEFT,padx=px,fill=tkinter.X,expand=1)
        
        f.pack(side=tkinter.TOP,padx=2,pady=2)
        #consider masks and zoom...
        if len(self.mask.mask)!=0 and self.usemaskinimage:
            pm=self.mask.mask[::-1,:]
        else:
            pm=ones(self.mapdata.data.get(0)[::-1,:].shape)
        if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:
            pm=pm[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
        #do an I0 calc first - only need to do this one once...
        #get i0!!!!!
        iind=self.mapdata.labels.index(self.i0chan.getvalue())+2
        i0dat=self.mapdata.data.get(iind)[::-1,:]
        ##index change above
        if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:            
            i0dat=i0dat[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
        i0dat=i0dat*np.where(i0dat>0,1,0)
        i0dat=i0dat*pm
        i0sum=sum(sum(i0dat))
        if i0sum==0:
            tkinter.messagebox.showwarning('Quantification','WARNING: I0 sum is ZERO')
            i0sum=1
        #go thru list
        self.quantlist=[]
        for g in goner:
            q=quantfield(g,inter,std=1)
            self.quantlist.append(q)
            #calculate...
            #get data
            datind=self.mapdata.labels.index(q.name)+2
            data=self.mapdata.data.get(datind)[::-1,:]
            #worry about zoom
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:            
                data=data[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            data=data*np.where(data>0,1,0)
            data=data*pm
            s=sum(sum(data))
            ctr=s/i0sum
            #place values
            if ctr>0: q.cts.setvalue(globalfuncs.chop(ctr,.4))
            else: q.cts.setvalue(ctr)
        self.quantentrydialog.show()

    def saveQuant(self,result):
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            self.quantentrydialog.withdraw()
            return
        if result=='Save':
            rt=self.quantvalidate(0)
            if not rt:
                return
            globalfuncs.setstatus(self.status,'Saving quantitative analysis parameters...')
            #save file
            fid=open(self.quantsavefn,'w')
            fid.write(self.quantfilestarttext)
            if self.quantfilestarttext[-1]!='\n': fid.write('\n')
            for q in self.quantlist:
                line=q.formula.getvalue()+'\t'+q.name+'\t'+q.element.getvalue()+'\t'+str(q.conc.getvalue())+'\t'+str(q.cts.getvalue())+'\t'+q.stdi0gain.get()+'\n'
                fid.write(line)
            fid.close()
            self.quantentrydialog.withdraw()
            globalfuncs.setstatus(self.status,'Save complete!')        

    def quantThickness(self):
        self.quantgendialog(self.enterQuantThickness)

    def enterQuantThickness(self,result):
        goner=self.selectquantdialog.getcurselection()
        self.selectquantdialog.withdraw()
        if result=='Cancel' or goner==():
            globalfuncs.setstatus(self.status,'No action taken')
            return

        cdt={}
        i=0
        for i in range(len(self.mapdata.labels)):
            u=1
            m=max(ravel(self.mapdata.data.get(i+2)))+1
            b=np.mod(ravel(self.mapdata.data.get(i+2)),1)
            if m>31: u=0
            if sum(abs(b))>0: u=0
            if len(np.where(self.mapdata.data.get(i+2)<0)[0])>1: u=0
            cdt[self.mapdata.labels[i]]=[int(m),u]

        #main thickness dialog
        self.quantthickdialog=QuantThicknessDialog.QuantThickDialog(self.imgwin,self.startQThick,goner,self.mapdata.energy,chanDict=cdt)
        self.quantthickdialog.show()

    def startQThick(self,result):
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No action taken')
            self.quantthickdialog.withdraw()
            return
        if result=='Apply':
            rt=self.quantthickvalidate(0)
            if not rt:
                return
            globalfuncs.setstatus(self.status,'Doing quantitative thickness correction...')
            #do
            self.doThickQuant()
            self.quantthickdialog.withdraw()
            globalfuncs.setstatus(self.status,'Quantative thickness correction complete!')

    def quantthickvalidate(self,display):
        #check entries for validness...
        globalfuncs.setstatus(self.status,'Checking validity...')
        valid=1
        for q in self.quantthickdialog.quantThicklist:
            #if q.element.getvalue()=='': valid=0
            #if q.element.getvalue() not in parseFormula.sym2elt.keys(): valid=0
            #if not parseFormula.parseCmd('syms',q.formula.getvalue()): valid=0
            if not q.energy.valid(): valid=0
        if not self.quantthickdialog.qtcorEnergy.valid() or float(self.quantthickdialog.qtcorEnergy.getvalue())<=0: valid=0
        for (a,w) in list(self.quantthickdialog.matrixAttenuators.items()):
            if not w.valid(): valid=0
        if not valid:
            globalfuncs.setstatus(self.status,'Quantification data entry invalid')
            if display:
                tkinter.messagebox.showwarning("Quantification Validation","Invalid entry")
        return valid        

    def doThickQuant(self):
        print("Qthick")

        if self.quantthickdialog.corType is None: return

        for q in self.quantthickdialog.quantThicklist:

            #get data
            Aind=self.mapdata.labels.index(q.name)+2
            #get index if needed:
            if self.quantthickdialog.corType=='Dynamic':
                Dind=self.mapdata.labels.index(self.quantthickdialog.dynamicChan.getvalue()[0])+2            
            
            corM=np.ones(self.mapdata.data.get(Aind).shape, dtype=np.float32)
            
            for att in list(self.quantthickdialog.matrixAttenuators.values()):  
                
                cf=self.quantthickdialog.materialDict.materials[att.material.getvalue()[0]].get()['CompoundFraction']
                cl=self.quantthickdialog.materialDict.materials[att.material.getvalue()[0]].get()['CompoundList']
                matcorI=absorptionCalc.calculateXS(cl,cf,float(att.density.getvalue()),float(self.quantthickdialog.qtcorEnergy.getvalue()))
                matcorF=absorptionCalc.calculateXS(cl,cf,float(att.density.getvalue()),float(q.energy.getvalue()))

                #calulate correction factor:
                mui=1.0/float(matcorI.absLength)
                muf=1.0/float(matcorF.absLength)
                d=float(self.quantthickdialog.materialDict.materials[att.material.getvalue()[0]].get()['Thickness'])
                adjd=d*10000
                corf=((1-math.exp(-mui*adjd-muf*adjd))/(mui+muf))/adjd
                
                #change from micron to cm
                if self.quantthickdialog.qtfinalthick.getvalue()==(): thickCM=1
                else: thickCM=d/10000

                print(q.name,corf,1/corf/thickCM)
         
                if self.quantthickdialog.corType=='Single':
                    corM=corM/corf/thickCM
                
                else: #dynamic
                    dval=int(att.label.split()[1])
                    corM[self.mapdata.data.get(Dind)==dval]=1/corf/thickCM
                        
            #apply
            newdata=self.mapdata.data.get(Aind)*corM
            #add data!
            nameroot=q.name+'-dcor'
            valid=0
            i=1
            while not valid:
                newname=nameroot+str(i)
                if newname not in self.mapdata.labels:
                    valid=1
                else:
                    i+=1
            self.addchannel(newdata,newname)
        print("Qthick done")

#################################  Plot Markers

    def plotmarkermain(self,opt=True):
        globalfuncs.setstatus(self.status,"Ready")
        if self.plotmarkerexist:
            if opt: self.plotmarkerwin.show()
            return
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.plotmarkerexist=1
        #create window
        self.plotmarkerwin=Pmw.MegaToplevel(self.imgwin)
        self.plotmarkerwin.title('Plot Marker Window')
        self.plotmarkerwin.userdeletefunc(func=self.killplotmarkerwin)
        h=self.plotmarkerwin.interior()
        h.configure(background='#d4d0c8')
        menubar=PmwTtkMenuBar.PmwTtkMenuBar(h)
        if os.sys.platform=='win32': menubar.component('hull').configure(bg='#d4d0c8')
        menubar.addmenu('File','')
        menubar.addmenuitem('File','command',label='Load Parameters',command=self.loadPMlist)
        menubar.addmenuitem('File','command',label='Save Parameters',command=self.savePMlist)
        menubar.addmenuitem('File','command',label='Save As Queue',command=self.savePMQlist)
        menubar.addmenu('Markers','')
        menubar.addmenuitem('Markers','command',label='Update All',command=self.updatePMall)
        menubar.addmenuitem('Markers','separator')
        menubar.addmenuitem('Markers','command',label='Set All Text',command=self.setPMtext)
        menubar.addmenuitem('Markers','command',label='Set All Default',command=self.setPMdefault)
        menubar.addmenuitem('Markers','separator')
        menubar.addmenuitem('Markers','command',label='Set All Black',command=self.setPMblack)
        menubar.addmenuitem('Markers','command',label='Set All White',command=self.setPMwhite)        
        
        menubar.pack(side=tkinter.TOP,fill=tkinter.X)
        if sys.platform=='darwin': hwv=700
        else: hwv=575
        j=Pmw.ScrolledFrame(h,hull_width=hwv,hull_height=500,usehullsize=1,vertflex='expand',horizflex='expand')
        j.interior().configure(background='#d4d0c8')
        j.pack(side=tkinter.TOP)
        k=tkinter.Frame(j.interior(),width=hwv,background='#d4d0c8')
        k.pack(side=tkinter.TOP)
        self.PMint=tkinter.Frame(j.interior(),bd=2,relief=tkinter.SUNKEN,background='#d4d0c8')
        self.PMint.pack(side=tkinter.TOP,fill='both',expand='y')
        #button for add new marker...
        b=PmwTtkButtonBox.PmwTtkButtonBox(self.PMint,orient='horizontal',hull_width=hwv-25,hull_background='#d4d0c8')
        b.add('Add Marker',command=self.addmarker,style='SBLUE.TButton',width=10)
        b.add('Clear All',command=self.clearallmarker,style='FIREB.TButton',width=10)
        b.pack(side=tkinter.TOP,padx=5,pady=10)
        if not opt:
            #minimize
            self.plotmarkerwin.iconify()

    def dispAddMarker(self,xp=None,yp=None,color=None):
        self.plotmarkermain(opt=False)
        new=self.addmarker(xp=xp,yp=yp)
        new.marker.selectitem('sm circle') 
        if color  is not None:
            new.color=color
            new.colchange()
    

    def addmarker(self,xp=None,yp=None):
        new=plotMarkerField(self.PMint,self.PMgetpos,self.PMupdate,delcb=self.PMdelcbline)
        if xp is not None:
            new.xpos.setvalue(xp)
            
        if yp is not None:
            new.ypos.setvalue(yp)
        self.plotmarkerlist.append(new)
        return new

    def clearallmarker(self,wipe=True):
        for pm in self.plotmarkerlist:
            pm.deleteline(cb=0)
            self.maindisp.markerupdate(pm,add=0)
            self.tcmarkerupdate(pm,add=0)
        if wipe: self.plotmarkerlist=[]

    def outputPMlistParam(self):
        out=[]
        for m in self.plotmarkerlist:
            out.append(m.output())
        return out

    def addmarkersback(self,pmlist):
        self.plotmarkerlist=[]
        for m in pmlist:
            new=plotMarkerField(self.PMint, self.PMgetpos, self.PMupdate, delcb=self.PMdelcbline)
            new.xpos.setvalue(m['xpos'])
            new.ypos.setvalue(m['ypos'])
            new.marker.selectitem(m['mark'], setentry=1)
            if m['mark']=='text':
                new.checktype()
                new.textfield.setvalue(m['text'])
            new.color=m['color']
            new.colchange()
            self.plotmarkerlist.append(new)

    def PMgetpos(self,obj):
        self.maindisp.PMlock.acquire()
        self.maindisp.startPMgetpos()
        self.PMputpos(obj)
        
    def PMputpos(self,obj):
        if self.maindisp.PMlock.locked():
            self.root.after(250,self.PMputpos,obj)
        else:
            #getcoords
            obj.xpos.setvalue(self.maindisp.markerexport[0])
            obj.ypos.setvalue(self.maindisp.markerexport[1])
            obj.validate()
            if obj.valid: self.PMupdate(obj)

    def updatePMall(self):
        for pm in self.plotmarkerlist:
            self.maindisp.markerupdate(pm)
            if self.triColorWindow.exist:
                self.tcmarkerupdate(pm)

    def setPMtext(self):
        i=1
        for pm in self.plotmarkerlist:
           pm.marker.selectitem('text')
           pm.checktype()
           pm.textfield.setvalue(str(i))
           i+=1
           pm.update()

    def setPMdefault(self):
        for pm in self.plotmarkerlist:
            pm.marker.selectitem('sm circle')
            pm.checktype()
            pm.update()

    def setPMwhite(self):
        for pm in self.plotmarkerlist:
            pm.color="#FFFFFF"
            pm.colchange()
           
    def setPMblack(self):
        for pm in self.plotmarkerlist:
            pm.color="#000000"
            pm.colchange()

    def PMupdate(self,obj):
        self.maindisp.markerupdate(obj)
        if self.triColorWindow.exist:
            self.tcmarkerupdate(obj)
            
    def killplotmarkerwin(self):
        self.plotmarkerexist=0
        self.clearallmarker()
        self.plotmarkerwin.destroy()
        
    def PMdelcbline(self,obj):
        self.maindisp.markerupdate(obj,add=0)
        self.tcmarkerupdate(obj,add=0)
        self.plotmarkerlist.remove(obj)

    def savePMQlist(self):
        #make sure all valid
        globalfuncs.setstatus(self.status,'Ready')
        cont=1
        for pm in self.plotmarkerlist:
            pm.validate()
            if not pm.valid: cont=0
        if not cont:
            globalfuncs.setstatus(self.status,'Marker fields not valid')
            tkinter.messagebox.showwarning("Marker Validation","Invalid entry")
            return
        #get filename
        fty=[("Queue param files","*.qsd"),("all files","*")]
        fn=globalfuncs.ask_save_file('dataqueue.qsd',self.filedir.get(),ext=fty)
        if fn!='':
            #test extensions
            if os.path.splitext(fn)[1]=='':fn=fn+'.qsd'
        else:
            globalfuncs.setstatus(self.status,"cancelling save...")
            return        
        #save
        fid=open(fn,'w')
        fid.write('#XASSCAN QUEUE\n')
        for pm in self.plotmarkerlist:
            line=pm.xpos.getvalue()+'\t'+pm.ypos.getvalue()+'\tXXX\tXXX\t1\tNone\t0\tI1\tNone\tNone\tNone'
            fid.write(line+'\n')
        fid.close()
        globalfuncs.setstatus(self.status,"Dataqueue values saved.")
    

    def savePMlist(self):
        #make sure all valid
        globalfuncs.setstatus(self.status,'Ready')
        cont=1
        for pm in self.plotmarkerlist:
            pm.validate()
            if not pm.valid: cont=0
        if not cont:
            globalfuncs.setstatus(self.status,'Marker fields not valid')
            tkinter.messagebox.showwarning("Marker Validation","Invalid entry")
            return
        #get filename
        fty=[("MPM param files","*.mpm"),("all files","*")]
        fn=globalfuncs.ask_save_file('markerlist.mpm',self.filedir.get(),ext=fty)
        if fn!='':
            #test extensions
            if os.path.splitext(fn)[1]=='':fn=fn+'.mpm'
        else:
            globalfuncs.setstatus(self.status,"cancelling save...")
            return        
        #save
        fid=open(fn,'w')
        fid.write('SMAK MARKERS\n')
        for pm in self.plotmarkerlist:
            line=pm.xpos.getvalue()+'\t'+pm.ypos.getvalue()+'\t'+pm.marker.getvalue()[0]+'\t'+pm.color+'\t'
            if pm.textpresent: line=line+pm.textfield.getvalue()
            fid.write(line+'\n')
        fid.close()
        globalfuncs.setstatus(self.status,"Marker values saved.")
    
    def loadPMlist(self):
        #get filename
        fty=[("MPM param files","*.mpm"),("all files","*")]
        fn=globalfuncs.ask_for_file(fty,self.filedir.get())
        if fn=='':
            print('Load cancelled')
            return 
        #read first line
        fid=open(fn,'r')
        l=fid.readline()
        if l!='SMAK MARKERS\n':
            print('Invalid parameter file!')
            fid.close()
            return
        #read data
        lines=fid.readlines()
        fid.close()
        for line in lines:
            #ensure line valid
            if len(line)<2: continue
            if line[0]=='#': continue
            l=line.rstrip()
            l=l.split('\t')
            invalid=0
            try:
                xp=float(l[0])
                yp=float(l[1])
            except:
                invalid=1
            if l[2] not in ['sm circle','big circle','sm square','big square','sm emptycircle','big emptycircle','sm emptysquare','big emptysquare','sm triangle','big triangle','text']: invalid=2
            else:
                if l[2]=='text':
                    try: tf=l[4]
                    except: invalid=3
            if len(l[3])!=7 or l[3][0]!='#': invalid=4
            if not invalid:
                #add line   
                new=plotMarkerField(self.PMint,self.PMgetpos,self.PMupdate,delcb=self.PMdelcbline)
                new.xpos.setvalue(xp)
                new.ypos.setvalue(yp)
                new.marker.selectitem(l[2],setentry=1)
                if l[2]=='text':
                    new.checktype()
                    new.textfield.setvalue(l[4])
                new.color=l[3]
                new.colchange()
                self.plotmarkerlist.append(new)
            else:
                print('line not valid',invalid)
        #update all makers
        for pm in self.plotmarkerlist:
            self.PMupdate(pm)

#################################  Histograms

    def starthistogram(self):
        globalfuncs.setstatus(self.status,"Ready")
        if self.histogramWindow.exist:
            self.histodialog.show()
            return
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        else:
            ps = MakeHistogramClass.MakeHistogramWindowParams(self.DTICRchanval, self.maindisp, self.dodt, self.activeFileBuffer, self.dataFileBuffer, self.filenb, self.root, self.status)
            self.histogramWindow.create(self.mapdata, ps)
            
            
#################################  Row shift

    def doRowInterpolation(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return
        globalfuncs.setstatus(self.status,"Ready")
        ind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        dchan=self.mapdata.data.get(ind)
        edit=False
        for j in range(1,dchan.shape[0]-1):
            if j%10==0: print(j)
            if sum(dchan[j,:].astype(np.int32))==0:
                edit=True
                print(j,"empty")
                dchan[j,:]=(dchan[j-1,:]+dchan[j+1,:])/2.0
        if edit:
            self.mapdata.data.put(ind,dchan)  
            #display
            self.domapimage()            
        globalfuncs.setstatus(self.status,"Done!")        

    def doRowInterpolationAll(self):
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.datachan.get()==():
            return
        globalfuncs.setstatus(self.status,"Ready")
        for ind in range(len(self.mapdata.labels)):
            dchan=self.mapdata.data.get(ind+2)
            edit=False
            for j in range(1,dchan.shape[0]-1):
                if j%10==0: print(j)
                if sum(dchan[j,:].astype(np.int32))==0:
                    edit=True
                    print(j,"empty")
                    dchan[j,:]=(dchan[j-1,:]+dchan[j+1,:])/2.0
            if edit:
                self.mapdata.data.put(ind+2,dchan)  
        #display
        self.domapimage()            
        globalfuncs.setstatus(self.status,"Done!")  
        
    def startPixRowShift(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not tkinter.messagebox.askokcancel('Row Shift','Click on desired row to perform pixel shift'):
            globalfuncs.setstatus(self.status,"Pixel shift cancelled")
            return
        #get row/cancel
        #create binding
        self.maindisp.editBindingB1(False)
        self.maindisp.imframe.bind(sequence="<Button-1>",func=self.maindisp.getcolumn)
    
    def returnPRS(self,row,forward=0):
        if forward:
            self.returnCD(row)
            return
        #highlight row?
        #get num to shift
        shift=tkinter.simpledialog.askinteger(title='Row Shift',prompt='Enter number of pixels to shift',initialvalue=0)
        if shift==0 or shift is None:
            globalfuncs.setstatus(self.status,"Pixel shift cancelled")
            return            
        #get data
        for i in range(len(self.mapdata.labels)):
            tmp=self.mapdata.data.get(i+2)[row,:].copy()
            #replace
            if shift>=0:
                for j in range(len(tmp)):
                    if j<shift:
                        self.mapdata.data.putPixel([row,j,i+2],0.0)
                    else:
                        self.mapdata.data.putPixel([row,j,i+2],tmp[j-shift])
            else:
                for j in range(len(tmp)):
                    if j<len(tmp)+shift:
                        self.mapdata.data.putPixel([row,j,i+2],tmp[j-shift])
                    else:
                        self.mapdata.data.putPixel([row,j,i+2],0.0)
        #display
        self.domapimage()

    def startLowSignalShift(self):
        globalfuncs.setstatus(self.status,"Ready")
        if "TIME" not in self.mapdata.labels:
            tkinter.messagebox.askquestion(title="Low Signal PixImage Fix", message="No TIME channel - CANCELING")
            return
        chan="TIME"
        if not tkinter.messagebox.askyesno("Low Signal PixImage Fix", "Use TIME channel to calculate fix?"):
            return
            chan="INTENS"
            if not tkinter.messagebox.askyesno("Low Signal PixImage Fix", "Use INTENS channel to calculate fix?"):
                return        
        ind=self.mapdata.labels.index(chan)+2
        shftind=[]
        for s in self.mapdata.labels:
            if chan=="TIME" and s!="TIME" and s[0:2]!="I0" and s[0:2]!="I1":
                shftind.append(self.mapdata.labels.index(s)+2)
            if chan=="INTENS" and s in ["RED","GREEN","BLUE"]:
                shftind.append(self.mapdata.labels.index(s)+2)                
        timechan=self.mapdata.data.get(ind)[:,:]
        tv=np.mean(timechan)
        if chan=="TIME":
            thresh=tv*0.005
        else: 
            thresh=tv*0.5
        print(thresh)
        for j in range(timechan.shape[0]):
            if j%10==0: print(j)
            slindlen=1
            mod=False
            res=np.copy(timechan[j,:])
            while slindlen!=0:
                slind=np.where(res<thresh)[0]
                slindlen=len(slind)
                if slindlen==0: 
                    continue
                if not mod:
                    dres={}
                    for m in shftind:
                        dres[m]=self.mapdata.data.getRow(m,j)
                mod=True
                add=np.append(res[slind[0]+1:],tv)
                res[slind[0]:]=add
                for m in shftind:
                    dadd=np.append(dres[m][slind[0]+1:],0)
                    dres[m][slind[0]:]=dadd                    
                
            if mod:
                timechan[j]=res
                for m in shftind:
                    self.mapdata.data.putRow(m,j,dres[m])
        self.mapdata.data.put(ind,timechan)    
        
    def startAutoPixRowShift(self):
        globalfuncs.setstatus(self.status,"Ready")
        if "TIME" not in self.mapdata.labels:
            tkinter.messagebox.askquestion(title="Automatic PixImage Fix", message="No TIME channel - CANCELING")
            return
        if not tkinter.messagebox.askyesno("Automatic PixImage Fix", "Use TIME channel to calculate fix?"):
            return
        ind=self.mapdata.labels.index("TIME")+2
        timechan=self.mapdata.data.get(ind)[:,:]
        shift=[]
        for j in range(timechan.shape[0]):
            sft=0
            maxtest=min(10,timechan.shape[1])
            for k in range(maxtest):
                if timechan[j,k]<0.1*timechan[j,maxtest]:
                    sft=k+1
            shift.append(-sft)
        print(shift)
        globalfuncs.setstatus(self.status,"Applying correction...")
        
        row=0
        for m in shift:
            if m==0: 
                row+=1
                continue
            for i in range(len(self.mapdata.labels)):
                tmp=self.mapdata.data.get(i+2)[row,:].copy()
                #replace
                for j in range(len(tmp)):


                    if j<len(tmp)+m:
                        self.mapdata.data.putPixel([row,j,i+2],tmp[j-m])
                    else:
                        self.mapdata.data.putPixel([row,j,i+2],0.0)
            row+=1
        #display
        globalfuncs.setstatus(self.status,"AutoPix Shift Complete")            
        self.domapimage()        
        

    def startPixInterlace(self):
        globalfuncs.setstatus(self.status,"Ready")
        shift=tkinter.simpledialog.askinteger(title='Interlace Correction',prompt='Enter pixel offset magntitude',initialvalue=0)
        if shift==0 or shift is None:
            globalfuncs.setstatus(self.status,"Interlace shift cancelled")
            return
        shift=-shift
        for i in range(len(self.mapdata.labels)):
            tmp=self.mapdata.data.get(i+2)
            if shift>0:
                nr=zeros((tmp.shape[0],shift),dtype=np.float32)
                sdata=np.concatenate((tmp,nr),axis=1)
                sdata=sdata[:,shift:]
            else:
                nr=zeros((tmp.shape[0],-shift),dtype=np.float32)
                sdata=np.concatenate((nr,tmp),axis=1)
                sdata=sdata[:,:sdata.shape[1]+shift]
            newdata=np.zeros_like(tmp)
            newdata[::2,:]=tmp[::2,:]
            newdata[1::2,:]=sdata[1::2,:]
            
            #display
            self.docalcimage(newdata)

            #save
            self.mapdata.data.put(i+2,newdata)
            self.changes=1

    def startApplyAlignStackSequence(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.lastAlignStack==[]:
            print("Need alignment, do Stack Alignment first")
            globalfuncs.setstatus(self.status,'Need alignment, do Stack Alignment first')
            return        
        analysisOptions=['Start','Cancel']

        self.alignStackDialog=Pmw.SelectionDialog(self.imgwin,title="Select Image Channels",buttons=analysisOptions,defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.alignApplyStackNextSequence)#,buttonlength=4)
        self.alignStackDialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        

    def alignApplyStackNextSequence(self,result):
        chans=self.alignStackDialog.getcurselection()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No alignment action taken')
            self.alignStackDialog.withdraw()
            return
        if len(chans)==0:
            globalfuncs.setstatus(self.status,'Select channels for alignment first')
            return           
        self.alignStackDialog.withdraw()
        
        #check length
        if len(self.lastAlignStack)!=len(chans)-1:
            if not tkinter.messagebox.askokcancel('Apply Stack Alignment','Previous alignment has different channel length\nthan current selection.'):
                globalfuncs.setstatus(self.status,"Apply stack alignment cancelled")
                return            
        
        da=[]
        for c in chans:
            dataind=self.mapdata.labels.index(c)+2
            #worry about zooms
            dr=self.mapdata.data.get(dataind)[::-1,:]#[::-1,:,dataind]
            ##and masks???
            ##if len(self.mask.mask)!=0 and self.usemaskinimage:
            ##    dr=self.mask.mask[::-1,1]*dr
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:    
                dr=dr[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]            
            nd=ravel(dr)
            print(c,sum(nd))
            da.append(dr)
        da=array(da,dtype=np.float32)
        print(da.shape)
        for i in range(da.shape[0]-1):
            if i>len(self.lastAlignStack): continue
            resample=5
            inA=irdTiles.resample(da[i,:,:],resample)
            inB=irdTiles.resample(da[i+1,:,:],resample)
            #result=ird.translation(inA,inB)
            #print result['success'],result['tvec']/resample
            #self.lastAlignStack.append(result["tvec"]/resample)
            newdata=ird.transform_img(da[i+1,:,:],tvec=self.lastAlignStack[i])
            da[i+1,:,:]=newdata
        self.alignStackSave(da,"-stack",chans)


    def startAlignStackSequence(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        analysisOptions=['Start','Cancel']

        self.alignStackDialog=Pmw.SelectionDialog(self.imgwin,title="Select Image Channels",buttons=analysisOptions,defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.alignStackNextSequence)#,buttonlength=4)
        self.alignStackDialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
    
    def alignStackNextSequence(self,result):
        chans=self.alignStackDialog.getcurselection()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No alignment action taken')
            self.alignStackDialog.withdraw()
            return
        if len(chans)==0:
            globalfuncs.setstatus(self.status,'Select channels for alignment first')
            return           
        self.alignStackDialog.withdraw()
        self.lastAlignStack=[]
        
        da=[]
        for c in chans:
            dataind=self.mapdata.labels.index(c)+2
            #worry about zooms
            dr=self.mapdata.data.get(dataind)[::-1,:]#[::-1,:,dataind]
            ##and masks???
            ##if len(self.mask.mask)!=0 and self.usemaskinimage:
            ##    dr=self.mask.mask[::-1,1]*dr
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:    
                dr=dr[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]            
            nd=ravel(dr)
            print(c,sum(nd))
            da.append(dr)
        da=array(da,dtype=np.float32)
        print(da.shape)
        for i in range(da.shape[0]-1):
            resample=5
            inA=irdTiles.resample(da[i,:,:],resample)
            inB=irdTiles.resample(da[i+1,:,:],resample)
            result=ird.translation(inA,inB)
            print(result['success'],result['tvec']/resample)
            self.lastAlignStack.append(result["tvec"]/resample)
            newdata=ird.transform_img(da[i+1,:,:],tvec=result["tvec"]/resample)
            da[i+1,:,:]=newdata
        self.alignStackSave(da,"-stack",chans)
        

    def startAlignStack(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        analysisOptions=['Translation','Rows','Cancel']

        self.alignStackDialog=Pmw.SelectionDialog(self.imgwin,title="Select Image Channels",buttons=analysisOptions,defaultbutton='Cancel',
                                                   scrolledlist_labelpos='n',label_text='Select Channels',scrolledlist_items=self.mapdata.labels,
                                                   command=self.alignStackNext)#,buttonlength=4)
        self.alignStackDialog.component('scrolledlist').component('listbox').configure(selectmode=tkinter.EXTENDED)
        
        
    def alignStackNext(self,result):
        chans=self.alignStackDialog.getcurselection()
        if result=='Cancel':
            globalfuncs.setstatus(self.status,'No alignment action taken')
            self.alignStackDialog.withdraw()
            return
        if len(chans)==0:
            globalfuncs.setstatus(self.status,'Select channels for alignment first')
            return           
        self.alignStackDialog.withdraw()
        

        da=[]
        for c in chans:
            dataind=self.mapdata.labels.index(c)+2
            #worry about zooms
            dr=self.mapdata.data.get(dataind)[::-1,:]#[::-1,:,dataind]
            ##and masks???
            ##if len(self.mask.mask)!=0 and self.usemaskinimage:
            ##    dr=self.mask.mask[::-1,1]*dr
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:    
                dr=dr[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]            
            nd=ravel(dr)
            print(c,sum(nd))
            da.append(dr)
        da=array(da,dtype=np.float32)
        print(da.shape)
        #op=AlignStack3.AlignStack(data=[da],precision=4)
        op=[]
        t=time.process_time()
        if result in ['Translation','Rows']:
            op.TwoDAlign()
            ctext="-align"
        if result=='Rows':
            op.RowAlign(50,0.5,0.6)
            op.Finalize()
            ctext="-ralign"
        et=time.process_time()-t
        print('completed in ',et)
        print(op.TList)
        print(op.DataAligned[0].shape)
        dfinal=op.DataAligned[0]
        self.alignStackSave(dfinal,ctext,chans)
        
        
    def alignStackSave(self, dfinal, ctext, chans):
    
        for i in range(dfinal.shape[0]):
            cind=0
            noexit=1
            name=ctext
            while noexit:
                cind+=1
                name=chans[i]+name+str(cind)
                if name not in self.mapdata.labels:
                    noexit=0
            data=dfinal[i,:,:]
            ##print data
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:
                nd=zeros(self.mapdata.data.get(0).shape,dtype=np.float32)
                pm=nd[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
                nd[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]=data
                nd=nd[::-1,:]
                self.addchannel(nd,name)
            else:
                data=data[::-1,:]
                self.addchannel(data,name)   

#################################  Remove Columns

    def startRemoveColumns(self):
        if self.datachan.get()==():
            return
        globalfuncs.setstatus(self.status,"Ready")
        if not tkinter.messagebox.askokcancel('Remove Column','Click on desired column to remove'):
            globalfuncs.setstatus(self.status,"Column removal cancelled")
            return
        #get row/cancel
        #create binding
        self.maindisp.editBindingB1(False)
        self.maindisp.imframe.bind(sequence="<Button-1>",func=self.maindisp.getrow)
    
    def returnCD(self,col):
        #highlight row?
        #get num to shift
        shift=tkinter.simpledialog.askinteger(title='Remove Column',prompt='Enter number of columns to remove',initialvalue=0)
        if shift==0 or shift is None:
            globalfuncs.setstatus(self.status,"Columnn removal cancelled")
            return            
        #get cur data
        datind=self.mapdata.labels.index(self.datachan.getvalue()[0])+2
        shift=shift-1
        start=max(0,col-shift)
        end=min(col+shift+1,len(self.mapdata.data.get(datind)[0,:]))#[0,:,datind]))
        for i in range(start,end):
            for j in range(len(self.mapdata.data.get(datind)[:,i])):#[:,i,datind])):
                self.mapdata.data.putPixel([j,i,datind],0.0)
        #display
        self.domapimage()

#################################  Moments

    def startdatamoments(self):
        #test...
        globalfuncs.setstatus(self.status,"Ready")

        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.momentWindow.exist:
            ps = MomentAnalysisClass.MomentAnalysisParams(self.status, self.datachan, self.dodt, self.mask, self.maindisp, self.plotmarkermain, self.PMupdate, self.addmarker)    
            self.momentWindow.create(self.mapdata, ps)
        else:
            self.momentWindow.win.show()
        
        
################################# UV-MCA Colorspaces

    def convertMCAtoRGBtext(self):
        self.convertMCAtoColorstext('RGB')

    def convertMCAtoXYZtext(self):
        self.convertMCAtoColorstext('XYZ')

    def convertMCAtoLABtext(self):
        self.convertMCAtoColorstext('LAB')

    def convertMCAtoRGB(self):
        self.convertMCAtoColors('RGB')

    def convertMCAtoXYZ(self):
        self.convertMCAtoColors('XYZ')

    def convertMCAtoLAB(self):
        self.convertMCAtoColors('LAB')

    def convertMCAtoColors(self,cspace,returndata=False):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if self.MCAfilename=='':
            globalfuncs.setstatus(self.status,'No UV-MCA file defined')
            #show warning
            tkinter.messagebox.showwarning('UV-MCA data','Please define file with UV-MCA data first')
            return

        mt='Data dwell time (msec)?'
        coltime=tkinter.simpledialog.askfloat(title='Save Deadtime File',prompt=mt,initialvalue=100)
        coltime=coltime*1000
        
        globalfuncs.setstatus(self.status,"Calculating UV-MCA data...")
        #go thru MCA file and integrate
        startt=time.process_time()
        fid=h5py.File(self.MCAfilename)
        if "/main/mcadata" in fid:
            mcadata=fid['/main/mcadata']
        elif "/main/oodata" in fid:
            mcadata=fid['/main/oodata']
        else:
            print('no uv-mcadata found')
            return
        self.mcamaxno=mcadata.shape[1]
        maxlines=mcadata.shape[0]
        print('hdf',self.mcamaxno,maxlines)

        linenum=0
        linetot=0
        early=0
        sprs=0
        lineindex=0

        if not returndata:
            newdataX=zeros(self.mapdata.nxpts*self.mapdata.nypts,dtype=np.float32)
            newdataY=zeros(self.mapdata.nxpts*self.mapdata.nypts,dtype=np.float32)
            newdataZ=zeros(self.mapdata.nxpts*self.mapdata.nypts,dtype=np.float32)
        else:
            newdataX=zeros(maxlines,dtype=np.float32)
            newdataY=zeros(maxlines,dtype=np.float32)
            newdataZ=zeros(maxlines,dtype=np.float32)
            
        s=None
        b=None

        if os.path.exists("oospec_calibration.txt"):
            fid=open("oospec_calibration.txt",'rU')
            ls=fid.readlines()
            fid.close()
            for l in ls:
                if l=='': continue
                if l[0]=='#': continue
                if l[0]!='@': continue
                lp=l.split(':')
                print(lp)
                if lp[0]=='@slope':
                    s=float(str.strip(lp[1]))
                elif lp[0]=='@int':
                    b=float(str.strip(lp[1]))
                else:
                    print('unknown factor')
        if s is None or b is None:
            nm=np.arange(2048)*.3598+193.83  #adc
    #        nm=np.arange(2048)*.3775+314.96  #fx
        else:
            nm=np.arange(2048)*s+b
            
        if not returndata:
            goto=min(self.mapdata.nxpts*self.mapdata.nypts,maxlines)
        else:
            goto=maxlines
        
        #calculate baseline      
        avgspec=np.mean(mcadata[:,0:100])
        print('uvbase',avgspec)
        
        if 1:

            cmf=array(ciexyz._CIEXYZ_1931_table)
            ciex=np.interp(nm,cmf[:,0],cmf[:,1],left=0,right=0)
            ciey=np.interp(nm,cmf[:,0],cmf[:,2],left=0,right=0)
            ciez=np.interp(nm,cmf[:,0],cmf[:,3],left=0,right=0)
            
            for i in range(2048):
                if (i+1)%20==0: print("line: ",i+1)
                if ciex[i]==0 and ciey[i]==0 and ciez[i]==0: continue
                sp=mcadata[:,i]-avgspec
                mleng=len(sp*ciex[i])
                td=np.zeros(len(newdataX))
                td[:mleng]=sp*ciex[i]
                newdataX=newdataX + td
                td[:mleng]=sp*ciey[i]
                newdataY=newdataY + td
                td[:mleng]=sp*ciez[i]
                newdataZ=newdataZ + td
            
            #test here
            fid.close()
            
            #need to find a normalization aspect somewhere...   
            newdataX=newdataX/coltime            
            newdataY=newdataY/coltime            
            newdataZ=newdataZ/coltime            
            
            if cspace!='XYZ':
                for j in range(goto):
                    if cspace=='RGB':
                        out=colormodels.irgb_from_xyz([newdataX[j],newdataY[j],newdataZ[j]])
                    elif cspace=='LAB':
                        out=colormodels.lab_from_xyz([newdataX[j],newdataY[j],newdataZ[j]])
                    else:
                        out=[newdataX[j],newdataY[j],newdataZ[j]]
                    newdataX[j]=out[0]
                    newdataY[j]=out[1]
                    newdataZ[j]=out[2]                    
                    if (j+1)%1000==0: print("line: ",j+1)

                               
        else:

            for i in range(goto):
                sp=(mcadata[i,:]-avgspec)/100000.
                cie=array([nm,sp]).T
                xyz=ciexyz.xyz_from_spectrum(cie)
                if cspace=='RGB':
                    xyz=colormodels.irgb_from_xyz(xyz)
                if cspace=='LAB':
                    xyz=colormodels.lab_from_xyz(xyz)
                newdataX[i]=xyz[0]
                newdataY[i]=xyz[1]
                newdataZ[i]=xyz[2]
                if (i+1)%50==0: print("line: ",i+1)
                           
            fid.close()
        
        if returndata:
            print(time.process_time()-startt)
            return [newdataX,newdataY,newdataZ]
            
        #place new data into main data
        newdataX=np.reshape(newdataX,(self.mapdata.nypts,self.mapdata.nxpts))
        newdataY=np.reshape(newdataY,(self.mapdata.nypts,self.mapdata.nxpts))
        newdataZ=np.reshape(newdataZ,(self.mapdata.nypts,self.mapdata.nxpts)) 

        if cspace=='RGB': labels=['Rc','Gc','Bc']
        elif cspace=='XYZ': labels=['Xc','Yc','Zc']
        else: labels=['Lc','Ac','Bc']
        
        ladd=''
        inc=1
        ready=0
        while not ready:
            if labels[0]+ladd in self.mapdata.labels or labels[1]+ladd in self.mapdata.labels or labels[2]+ladd in self.mapdata.labels:
                ladd=str(inc)
                inc+=1
            else:
                flabels=[s+ladd for s in labels]
                ready=1
                
        self.addchannel(newdataX,flabels[0])        
        self.addchannel(newdataY,flabels[1])        
        self.addchannel(newdataZ,flabels[2])        

        print(time.process_time()-startt)
        globalfuncs.setstatus(self.status,'UV-MCA calculations complete')
        
        
    def convertMCAtoColorstext(self,cspace):
        [d1,d2,d3]=self.convertMCAtoColors(cspace,returndata=True)
        
        #write data file
        fn=globalfuncs.ask_save_file('colorexport'+cspace,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            #globalfuncs.setstatus(self.status,'Save cancelled')
            return
        if fn[-4:].lower()!='.txt': fn=fn+'.txt'
        outfile=""
        for i in range(len(d1)):
            outfile+=str(d1[i])+"\t"+str(d2[i])+"\t"+str(d3[i])+"\n"
        fid=open(fn,'w')
        fid.write(outfile)
        fid.close()
        globalfuncs.setstatus(self.status,'UV-MCA calculations complete')
        
    
#################################  Workflow Ideas

    
    def runAddin(self,**kwa):
        
        self.saveCurrentFileBufferInfo()
        
        kwargs=kwa['kwargs']
        
        # if False:
        #     out = json.dumps(kwargs)
        #     fid=open('batchCommands.json','a')
        #     fid.write(out+'\n')
        #     fid.close()
        
        print (kwargs)
        if kwargs is None:
            print ('no commands provided')
            return
        
        if 'requires' in kwargs:
            if 'data' in kwargs['requires']:
                if not self.hasdata:
                    print('No Data')
                    globalfuncs.setstatus(self.status,'No Data')
                    return
                
        if 'seq' not in kwargs:
            print ('no command sequence')
            return
        cl = kwargs['seq']
        self.batchMode.prepMacro(cl)
        rv = self.batchMode.defMacro()
        if rv: 
            self.batchMode.execute()
            globalfuncs.setstatus(self.status,'Macro call complete')
        else:
            globalfuncs.setstatus(self.status,'Macro call ERROR')

        self.loadFromCurrentFileBufferInfo()
    
    def defineCustomBatch(self):
        
        globalfuncs.setstatus(self.status,"Ready")

        if not self.scriptEditorWindow.exist:
            ps=BatchMode.ScriptEditorParams(self.status, self.batchData, self.addinmenuOpts, self.mainmenubar, self.runAddin, self.dataFileBuffer)
            self.scriptEditorWindow.create(None,ps)
        else:
            self.scriptEditorWindow.win.show()
            
    
    def reloadAddinBatch(self):
        self.readAddins(first=False)
    
    def saveAddinBatch(self):
        
        out=''
        for scr in self.batchData.values():
            out += json.dumps(scr)+'\n'

        fid=open('batchCommands.json','w')
        fid.write(out)
        fid.close()
        
        globalfuncs.setstatus(self.status,'Workflows written out...')
        self.reloadAddinBatch()

    def readAddins(self,first=False):
        if first:
            self.batchMode = BatchMode.ScriptManager(self.maindisp,self.addchannel,self.filedir.get(),self.imgwin)
        self.batchMode.update(self.dataFileBuffer,self.activeFileBuffer)
        
        try:
            fid=open('batchCommands.json','r')
            data=fid.readlines()
            fid.close()        
        except:
            print ('no scripts file')
            return
        self.batchData = {}
        for l in data:
            scr=json.loads(l)
            self.batchData[scr['title']]=scr
        #print (self.batchData)
        
        if not first:
            #clear menu
            for ms in self.addinmenuOpts.keys():
                men = self.mainmenubar.component('Workflow-menu')
                ind = men.index(ms)
                self.mainmenubar.deletemenuitems('Workflow',ind)

        #create menu options
        self.addinmenuOpts={}
        for scr in self.batchData.values():
            m=self.mainmenubar.addmenuitem('Workflow','command',label=scr['title'],command=functools.partial(self.runAddin,kwargs=scr))
            self.addinmenuOpts[scr['title']]=m
            
            
            
#################################  Export scan area info

    def exportScanArea(self):        
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return        
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_exportCoordinate.dcm'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        globalfuncs.setstatus(self.status,"Saving export of scan area coordinates...")        
        #save
        fid=open(fn,'w')
        fid.write('SMAK COORDINATE EXPORT\n')
        cexp=[self.maindisp.xsc[self.maindisp.zmxyi[0]],
              self.maindisp.xsc[self.maindisp.zmxyi[2]], 
              self.maindisp.ysc[self.maindisp.zmxyi[3]],
              self.maindisp.ysc[self.maindisp.zmxyi[1]]]
        for c in cexp:
            fid.write(str(c)+"\n")
        fid.close()
        globalfuncs.setstatus(self.status,"Export complete")
        
    def exportScanAreaDQ(self):        
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_exportQueueCoordinate.dpm'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return            
        #check if file exists
        if os.path.exists(fn):
            fid=open(fn,'r')
            l=fid.readline()
            if l!='DATASERVER REGIONS\n':
                print('Invalid parameter file!')
                fid.close()
                globalfuncs.setstatus(self.status,'Save cancelled')
                return
            fid.close()
        #ask for other info
        newregion={}
        newregion['comment']='\n'
        newregion['ooptics']='No'
        newregion['scandir']='Horizontal'
        newregion['scantype']='Continuous'
        newregion['slave']='Enabled'
        #need dwell
        dwell=tkinter.simpledialog.askfloat(title='Export Scan Area',prompt='Enter the dwell time for the new region (ms)',initialvalue=25)
        newregion['dwell']=dwell
        #need energy
        ev=tkinter.simpledialog.askfloat(title='Export Scan Area',prompt='Enter the MONO energy for the new region (eV)',initialvalue=self.mapdata.energy)
        newregion['energy']=[ev]
        #need wheel
        wheel=tkinter.simpledialog.askinteger(title='Export Scan Area',prompt='Enter the sample wheel',initialvalue=0)
        newregion['wheel']=wheel
        #need file
        filen=tkinter.simpledialog.askstring(title='Export Scan Area',prompt='Enter the file name for the region',initialvalue='')
        newregion['file']=filen
        yrange=[self.maindisp.xsc[self.maindisp.zmxyi[0]],
                self.maindisp.xsc[self.maindisp.zmxyi[2]],
                self.maindisp.xsc[1]-self.maindisp.xsc[0]]  
        newregion['yrange']=yrange
        zrange=[self.maindisp.ysc[self.maindisp.zmxyi[3]],
                self.maindisp.ysc[self.maindisp.zmxyi[1]],
                self.maindisp.xsc[0]-self.maindisp.xsc[1]] 
        newregion['zrange']=zrange
        newfrg=['image',newregion] 

        newqueue={}
        indexqueue=1
        if os.path.exists(fn):
            fid=open(fn,'rb')
            l=fid.readline()
            l=fid.read()
            newqueue=pickle.loads(l)
            fid.close()
            indexqueue=len(newqueue)+1
        newqueue[indexqueue]=newfrg 

        #save all
        fid=open(fn,'wb')
        regions=pickle.dumps(newqueue)
        fid.write(b'DATASERVER REGIONS\n')
        fid.write(regions)
        fid.close()
        globalfuncs.setstatus(self.status,"Export complete")
        
#################################  View colormaps

    def viewcolormap(self):
        self.maindisp.viewcolormap()
        globalfuncs.setstatus(self.status,"Ready")
        
    def viewtricolormap(self):
        #make a tricolor map legend
        if self.triColorWindow.exist==0:
            print("No triColor plot")
            globalfuncs.setstatus(self.status,'No tricolor plot')
            return
        numon=0
        colors=[]
        ltext=['','','']
        lcolor=['black','black','black']
        for n in (self.triColorWindow.tcred,self.triColorWindow.tcgreen,self.triColorWindow.tcblue):
            if n.getvalue()!='None':
                numon=numon+1
                if n==self.triColorWindow.tcred:
                    colors.append('red')
                    ltext[numon-1]=n.getvalue()
                    lcolor[numon-1]='red'
                if n==self.triColorWindow.tcgreen:
                    colors.append('green')
                    ltext[numon-1]=n.getvalue()
                    lcolor[numon-1]='green'
                if n==self.triColorWindow.tcblue:
                    colors.append('blue')                
                    ltext[numon-1]=n.getvalue()
                    lcolor[numon-1]='blue'
        if numon==0:
            return
        if numon==1:#monocolor linear
            linear=list(range(256))
            bar=[]
            for i in range(25):
                bar.append(linear)
            #colormap it
            bar=array(bar)
            cmap=zeros((25,256,3),dtype=np.float32)
            if not self.CMYKOn.get():
                if colors[0]=='red':
                    cmap[:,:,0]=bar
                if colors[0]=='green':
                    cmap[:,:,1]=bar
                if colors[0]=='blue':
                    cmap[:,:,2]=bar                
            else:
                if colors[0]=='red':
                    cmap[:,:,0]+=bar
                    cmap[:,:,1]+=bar
                if colors[0]=='green':
                    cmap[:,:,1]+=bar
                    cmap[:,:,2]+=bar
                if colors[0]=='blue':
                    cmap[:,:,2]+=bar
                    cmap[:,:,0]+=bar

        if numon==2:#bicolor linear
            linear=list(range(256))
            bar=[]
            for i in range(25):
                bar.append(linear)
            #colormap it
            bar=array(bar)
            cmap=zeros((25,256,3),dtype=np.float32)
            first=1
            
            for c in colors:
                if not self.CMYKOn.get():
                    if c=='red':
                        if first:
                            cmap[:,:,0]=bar
                            first=0
                        else: cmap[:,:,0]=bar[:,::-1]
                    if c=='green':
                        if first:
                            cmap[:,:,1]=bar
                            first=0
                        else: cmap[:,:,1]=bar[:,::-1]
                    if c=='blue':
                        if first:
                            cmap[:,:,2]=bar
                            first=0
                        else: cmap[:,:,2]=bar[:,::-1]

                else:
                    if c=='red':
                        if first:
                            cmap[:,:,0]+=bar
                            cmap[:,:,1]+=bar
                            first=0
                        else:
                            cmap[:,:,0]+=bar[:,::-1]
                            cmap[:,:,1]+=bar[:,::-1]
                    if c=='green':
                        if first:
                            cmap[:,:,1]+=bar
                            cmap[:,:,2]+=bar
                            first=0
                        else:
                            cmap[:,:,1]+=bar[:,::-1]
                            cmap[:,:,2]+=bar[:,::-1]
                    if c=='blue':
                        if first:
                            cmap[:,:,2]+=bar
                            cmap[:,:,0]+=bar
                            first=0
                        else:
                            cmap[:,:,2]+=bar[:,::-1]
                            cmap[:,:,0]+=bar[:,::-1]


        if numon==3:#tricolor triangle
            cmap=zeros((270,270,3),dtype=np.float32)
            for i in range(270):
                for j in range(270):
                    #find "color" of pixel i,j
                    #transform to rgb coordinates
                    jp=j-25
                    ip=i-10
                    r=jp/math.sqrt(3)*2
                    g=ip-jp/math.sqrt(3)
                    b=255-r-g
                    if r>=0 and g>=0 and b>=0:
                        if not self.CMYKOn.get():
                            cmap[i,j,0]=r
                            cmap[i,j,1]=g
                            cmap[i,j,2]=b
                        else:
                            cmap[i,j,0]=r+g
                            cmap[i,j,1]=g+b
                            cmap[i,j,2]=b+r                            
        cmap=cmap[:,::-1,:]           
        #make image of it
        cmap=cmap.astype('b')

        ppm=ImRadon.toimage(np.transpose(cmap),cmin=0,skip=1)
        self.tcmaplegend=ppm
        self.tclegendimage=ImageTk.PhotoImage(ppm)

        #ppm=Display.array2ppm(cmap)
        #self.tcmaplegend=ppm
        #self.tclegendimage=tkinter.PhotoImage(file=Display.save_ppm(ppm))
        w, h=self.tclegendimage.width(), self.tclegendimage.height()
        scalex=1
        scaley=1
        #self.tclegendimage=self.tclegendimage.zoom(scalex, scaley)
        #self.tclegendimage.configure(width=w*scalex, height=h*scaley)                    
        #create window if needed
        if not self.tclegendexist:
            self.tclegendimwin=Pmw.MegaToplevel(self.imgwin)
            self.tclegendimwin.title('Tricolor Legend Display')
            self.tclegendimwin.userdeletefunc(func=self.killtclegendimwin)
            hf=self.tclegendimwin.interior()
            hf.config(bg='black')
            self.tclegendimframe=tkinter.Canvas(hf,bg='black',borderwidth=2, relief=tkinter.FLAT, height=250, width=250, cursor='crosshair',highlightcolor='black',highlightbackground='black',selectbackground='black',selectforeground='black')
            self.tclegendimframe.pack(side=tkinter.LEFT,fill=tkinter.X)
            self.tclegenditems=[]
            self.tclegendexist=1
            #numbers
            ltf=tkinter.Frame(hf,bg='black')
            ltf.pack(side=tkinter.LEFT,fill='both')
            self.legmax=tkinter.Label(ltf,text="",anchor=tkinter.W,bg='black')
            self.legmin=tkinter.Label(ltf,text="",anchor=tkinter.W,bg='black')
            self.legmin.pack(side=tkinter.BOTTOM,fill=tkinter.X)
            self.legmax.pack(side=tkinter.TOP,fill=tkinter.X)
            self.legendimageexists=1
        #clear        
        if self.tclegenditems !=[] :
            for i in self.tclegenditems:
                self.tclegendimframe.delete(i)
            self.tclegenditems=[]
        #rescale canvas and slider
        self.tclegendimframe.config(height=h*scaley,width=w*scalex)
        self.tclegenditems.append(self.tclegendimframe.create_image((w*scalex+scalex)/2,(h*scaley+scaley)/2,anchor='center', image=self.tclegendimage))
        #place text if needed
        if numon in (1,2):
            self.legmax.config(fg=lcolor[0])
            self.legmin.config(fg=lcolor[1])
            globalfuncs.setstatus(self.legmax,ltext[0])
            globalfuncs.setstatus(self.legmin,ltext[1])
        if numon==3:
            #block bicolor...
            self.legmax.config(fg='black')
            self.legmin.config(fg='black')
            #aaply new
            self.tclegenditems.append(self.tclegendimframe.create_text(135,15,text=ltext[0],fill=lcolor[0]))
            self.tclegenditems.append(self.tclegendimframe.create_text(260,260,text=ltext[1],fill=lcolor[1]))
            self.tclegenditems.append(self.tclegendimframe.create_text(15,260,text=ltext[2],fill=lcolor[2]))
            
    def killtclegendimwin(self):
        self.tclegendexist=0
        self.tclegendimwin.destroy()
        
    def savecolormap(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        self.maindisp.viewcolormap()
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_legend.jpg'
        fn=globalfuncs.ask_save_file(fn,self.filedir.get())
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        self.maindisp.legendimwin.show()
        self.root.update_idletasks()
        globalfuncs.setstatus(self.status,"Saving legend display...")
        self.maindisp.savecmlegendimage(fn)
        globalfuncs.setstatus(self.status,"Legend display saved in: "+fn)
        
    def savetricolormap(self):
        globalfuncs.setstatus(self.status,"Ready")
        if not self.hasdata:
            print('No Data')
            globalfuncs.setstatus(self.status,'No Data')
            return
        if not self.triColorWindow.tcimageexists:
            print('No tricolor map')
            globalfuncs.setstatus(self.status,'No tricolor map')
            return
        self.viewtricolormap()
        #get file name
        fn=globalfuncs.trimdirext(self.dataFileBuffer[self.activeFileBuffer]['fname'])+'_legend.jpg'
        fn=globalfuncs.ask_save_file(fn,'')
        if fn=='':
            print('Save cancelled')
            globalfuncs.setstatus(self.status,'Save cancelled')
            return
        self.tclegendimwin.show()
        self.root.update_idletasks()
        globalfuncs.setstatus(self.status,"Saving legend display...")
        self.tclegendimwin.lift()
        self.imgwin.update()
        #if on windows:
        if os.sys.platform=='win32':
            rx=int(self.tclegendimwin.winfo_rootx())
            ry=int(self.tclegendimwin.winfo_rooty())
            rw=int(self.tclegendimwin.winfo_width())
            rh=int(self.tclegendimwin.winfo_height())
            screencapture.capture(rx,ry,rw,rh,fn)
            #im=ImageGrab.grab((rx,ry,rx+rw,ry+rh))
            #im.save(fn)
        else:
            rx=int(self.tclegendimwin.winfo_rootx())
            ry=int(self.tclegendimwin.winfo_rooty())
            rw=int(self.tclegendimwin.winfo_width())
            rh=int(self.tclegendimwin.winfo_height())
            screencapture.capture(rx,ry,rw,rh,fn)
            #p=ImageFile.Parser()
            #p.feed(self.tcmaplegend)
            #im=p.close()
            #im.save(fn)
        globalfuncs.setstatus(self.status,"Legend display saved in: "+fn)    


    def getChannelNameViaDialog(self):
        self.channamedialog=Pmw.SelectionDialog(self.imgwin,title='Channel Selection:',buttons=('OK','Cancel'),
                                              defaultbutton='OK',scrolledlist_labelpos='n',
                                              label_text='Select channel Classifier:',
                                              scrolledlist_items=self.mapdata.labels,command=self.getChannelNameViaDialog_b)
                                              
        self.channamedialog.show()                                     

    def getChannelNameViaDialog_b(self,result):
        self.channamedialog.withdraw()
        if result=='Cancel' or self.channamedialog.getcurselection()==():
            self.PCA_cluster_classifier=None
        else: 
            datind=self.mapdata.labels.index(self.channamedialog.getcurselection()[0])+2
            dr=self.mapdata.data.get(datind)[::-1,:]
            if self.maindisp.zmxyi[0:4]!=[0,0,-1,-1]:
                dr=dr[self.maindisp.zmxyi[1]:self.maindisp.zmxyi[3],self.maindisp.zmxyi[0]:self.maindisp.zmxyi[2]]
            self.PCA_cluster_classifier=ravel(dr)
        self.channamedialog.withdraw()

    def showclickhelp(self):
        self.helpdialog=Pmw.TextDialog(self.imgwin, scrolledtext_labelpos='n',title='Microtoolkit Help',
                defaultbutton=0,label_text='Microtoolkit tricks and tips!')
        help="""
        Channel Selection

        right-CLICK on channel name to bring up additional display windows
        
        Display

        Use "View - Change Min Size" to change display size
        Use "Legend - Show Scalebar" to show display scale
        Use "left DOUBLE CLICK" to show horizontal/vertical line profile and MCA if chosen
            On MacOS, use double tap.
        Use "right CLICK" to show pop-up menu to
            flip display axes
            show MCA
            clear ZOOM
        Use "SHIFT-left CLICK" to select points to average for MCA display
        Use "CTRL-left CLICK" to select area to zoom
            ZOOM area is same for tri-color plots (use same command in tri-color display too)
            ZOOM area is also selected for Correlation plots
        Use "ALT-left CLICK" to do arbitrary line profile
            On MacOS, use "Option-left CLICK"

        MCA Files

        Use "MCA Spectra - Define MCA file" to select MCA for viewing
        Use "MCA Spectra - Set MCA multi-file lines" to allow correct parsing of MCA files collected at multiple energies
        Use "MCA Spectra - Split MCA multi-files" to separate multiple energy MCA files into separate files

        SAVING JPG images:

        As a general rule, the toolkit is doing a screen capture in many cases.  Best to make sure that there
        are no windows obscuring the desired display if at all possible!  When in doubt, use the default file
        name and change it later if desired.
        """
        self.helpdialog.insert('end', help)
        self.helpdialog.configure(text_state='disabled')
        self.helpdialog.show()
        


################## Start Loop
        
class Start:
    Main(root)


root.mainloop()


#if __name__ == '__main__':
#    Main(root)
#    





