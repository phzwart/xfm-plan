# -*- coding: utf-8 -*-
"""
Created on Mon May 22 08:32:13 2023

@author: samwebb
"""
#standard
import importlib.util
import imutils
import math
import os
import tkinter
from tkinter.ttk import Button


#third party
import cv2
import imreg_dft as ird
from imreg_dft import tiles as irdTiles
import Pmw
import numpy as np

#local
import align_images 
import globalfuncs
from MasterClass import MasterClass

if importlib.util.find_spec('RegistrationCNN') is None:
    hasCNN=False
else:
    if True:
        import RegistrationCNN
        from utils.utils import tps_warp, tps_warp2D
        hasCNN=True
    else:
        hasCNN=False
        

def padArraySize(a,b):
    print (a.shape,b.shape)
    if a.shape==b.shape: return a,b
    del0 = b.shape[0]-a.shape[0]
    del1 = b.shape[1]-a.shape[1]
    if a.shape[0]<b.shape[0]:
        a = np.pad(a,((0,del0),(0,0)),mode='constant',constant_values=0)
    else:
        b = np.pad(b,((0,del0),(0,0)),mode='constant',constant_values=0)
    if a.shape[1]<b.shape[1]:
        a = np.pad(a,((0,0),(0,del1)),mode='constant',constant_values=0)
    else:
        b = np.pad(b,((0,0),(0,del1)),mode='constant',constant_values=0)
    print (a.shape,b.shape)
    return a,b

class EntryActive:
    def __init__(self,master,textl,width=60):
        filebar=tkinter.Frame(master,bd=2,background='#d4d0c8')
        self.entry=Pmw.EntryField(filebar, label_text=textl,labelpos=tkinter.W,validate='real',entry_width=width,command=None,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.entry.pack(side=tkinter.LEFT,padx=5,pady=2,fill=tkinter.X)
        filebar.pack(side=tkinter.TOP,padx=2,pady=2,fill=tkinter.X)

    def get(self):
        return self.entry.get()
    
    def set(self,value):
        self.entry.setvalue(value)
    
    def setActive(self):
        self.entry.component('entry').configure(state=tkinter.NORMAL)
        self.entry.component('label').configure(fg='black')
        
    def setDisabled(self):
        self.entry.component('entry').configure(state=tkinter.DISABLED)
        self.entry.component('label').configure(fg='gray70')
    

class PFileEntry:
    
    def __init__(self,master,textl,width=60):
        filebar=tkinter.Frame(master, relief=tkinter.SUNKEN,bd=2,background='#d4d0c8')
        self.fileentry=Pmw.EntryField(filebar, label_text=textl,labelpos=tkinter.W,validate=None,entry_width=width,command=None,hull_background='#d4d0c8',label_background='#d4d0c8')
        self.fileentry.pack(side=tkinter.LEFT,padx=5,pady=2,fill=tkinter.X)
        self.b=Button(filebar,text="Open",command=self.getfilename,style='OPEN.TButton',width=7)
        self.b.pack(side=tkinter.LEFT,padx=2,pady=2)
        filebar.pack(side=tkinter.TOP,padx=2,pady=2,fill=tkinter.X)

    def getfilename(self):
        defaults=[("MPM param files","*.mpm"),("all files","*")]
        fn=globalfuncs.ask_for_file(defaults,'')
        globalfuncs.entry_replace(self.fileentry,fn)

    def get(self):
        return self.fileentry.get()
    
    def setActive(self):
        self.fileentry.component('entry').configure(state=tkinter.NORMAL)
        self.fileentry.component('label').configure(fg='black')
        self.b.configure(state=tkinter.NORMAL)
        
    def setDisabled(self):
        self.fileentry.component('entry').configure(state=tkinter.DISABLED)
        self.fileentry.component('label').configure(fg='gray70')
        self.b.configure(state=tkinter.DISABLED)

class ExportRegWindowParams():
    def __init__(self,displayParams, dataFileBuffer, activeFileBuffer, posarg, addchannel):
        self.displayParams = displayParams
        self.dataFileBuffer = dataFileBuffer
        self.activeFileBuffer=activeFileBuffer
        self.posarg = posarg
        self.addchannel = addchannel

class ExportRegWindow(MasterClass):

    def _create(self):
        #make window
        self.win=Pmw.MegaToplevel(self.imgwin)
        self.win.title('Export With Image Registration Window')
        self.win.userdeletefunc(func=self.kill)
        h=self.win.interior()
        h.configure(background='#d4d0c8')
        
        hh=Pmw.ScrolledFrame(h,usehullsize=1,vertflex='fixed',horizflex='fixed',
                     hscrollmode='static',vscrollmode='static',
                     hull_width=800,hull_height=750)
        hh.interior().configure(background='#d4d0c8')
        hh.pack(side=tkinter.TOP,pady=2)
        ii=hh.interior()    
        
        #Menu bar??
        f=tkinter.Frame(ii,background='#d4d0c8')
        f.pack(side=tkinter.TOP,fill='both',pady=5)
        g=tkinter.Frame(ii,background='#d4d0c8')
        g.pack(side=tkinter.TOP,fill='both',pady=10)
        
        #new channel name and action buttons
        lf = tkinter.Frame(f,background='#d4d0c8')
        lf.pack(side=tkinter.LEFT,fill='both',expand=1,pady=2)        
        rf = tkinter.Frame(f,background='#d4d0c8')
        rf.pack(side=tkinter.LEFT,fill='both',expand=1,pady=2) 
        
        l=tkinter.Label(lf,text='Source Parameters',bd=2,relief=tkinter.RAISED,background='#d4d0c8',width=40)
        l.pack(side=tkinter.TOP,fill=tkinter.X,expand=1,pady=3,anchor=tkinter.N)
        self.exsourcefile=Pmw.ScrolledListBox(lf,labelpos='n',label_text='Export Source File',listbox_height=4,
                        selectioncommand=self.checkexsource,listbox_exportselection=0,listbox_takefocus=tkinter.TRUE,
                        hull_background='#d4d0c8',label_background='#d4d0c8')
        self.exsourcefile.setlist(self.ps.dataFileBuffer.keys())
        self.exsourcefile.pack(side=tkinter.TOP,fill=tkinter.X,expand=1,pady=3,padx=3,anchor=tkinter.N)
        self.exsourcechan=Pmw.ScrolledListBox(lf,labelpos='n',label_text='Export Source Master Chan',listbox_height=5,
                        selectioncommand=None,listbox_exportselection=0,listbox_takefocus=tkinter.TRUE,
                        hull_background='#d4d0c8',label_background='#d4d0c8')
        self.exsourcechan.pack(side=tkinter.TOP,fill=tkinter.X,expand=1,pady=3,padx=3,anchor=tkinter.N)
        self.exsourcedata=Pmw.ScrolledListBox(lf,labelpos='n',label_text='Export Source Channels',listbox_height=7,
                        selectioncommand=None,listbox_exportselection=0,listbox_takefocus=tkinter.TRUE,
                        hull_background='#d4d0c8',label_background='#d4d0c8')        
        self.exsourcedata.component('listbox').configure(selectmode=tkinter.EXTENDED)
        self.exsourcedata.pack(side=tkinter.TOP,fill=tkinter.X,expand=1,pady=3,padx=3,anchor=tkinter.N)

        l=tkinter.Label(rf,text='Destination Parameters',bd=2,relief=tkinter.RAISED,background='#d4d0c8',width=40)
        l.pack(side=tkinter.TOP,fill=tkinter.X,expand=0,pady=3)        
        self.exportdestfile=Pmw.ScrolledListBox(rf,labelpos='n',label_text='Export Destination File',listbox_height=4,
                        selectioncommand=self.checkdestsource,listbox_exportselection=0,listbox_takefocus=tkinter.TRUE,
                        hull_background='#d4d0c8',label_background='#d4d0c8')
        self.exportdestfile.setlist(self.ps.dataFileBuffer.keys())
        self.exportdestfile.pack(side=tkinter.TOP,fill=tkinter.X,expand=0,pady=3,padx=3)
        self.exportdestchan=Pmw.ScrolledListBox(rf,labelpos='n',label_text='Export Destination Master Chan',listbox_height=5,
                        selectioncommand=None,listbox_exportselection=0,listbox_takefocus=tkinter.TRUE,
                        hull_background='#d4d0c8',label_background='#d4d0c8')
        self.exportdestchan.pack(side=tkinter.TOP,fill=tkinter.X,expand=0,pady=3,padx=3)

        l=tkinter.Label(g,text='Registration Type',bd=2,relief=tkinter.RAISED,background='#d4d0c8')        
        l.pack(side=tkinter.TOP,fill=tkinter.X,expand=1,pady=3)         
        self.regtype=Pmw.RadioSelect(g,buttontype='radiobutton',orient='horizontal',command=self.checkregtype,hull_background='#d4d0c8')
        regalgos = ['ORB','AKAZE','SIFT','SimTrans','CNN','Manual Data']
        if not hasCNN:
            regalgos.pop(3)
        for text in regalgos:
            self.regtype.add(text,background='#d4d0c8')
        self.regtype.pack(side=tkinter.TOP,padx=3,pady=3)

        self.regmatch=Pmw.RadioSelect(g,buttontype='radiobutton',orient='horizontal',command=None,hull_background='#d4d0c8')
        for text in ['Pct','Knn']:
            self.regmatch.add(text,background='#d4d0c8')
        self.regmatch.pack(side=tkinter.TOP,padx=3,pady=3)

        self.regMaxFeatures=EntryActive(g,'maxFeatures',30)
        self.regKeepPercent=EntryActive(g,'keepPercent',30)
        self.regPfileSourceSel=PFileEntry(g,'Source Points File')
        self.regPfileDestSel=PFileEntry(g,'Dest. Points File')
        Pmw.alignlabels([self.regPfileSourceSel.fileentry,self.regPfileDestSel.fileentry])
        Pmw.alignlabels([self.regMaxFeatures.entry,self.regKeepPercent.entry])
        self.regMaxFeatures.set(200000)
        self.regKeepPercent.set(0.5)
        self.regtype.invoke('ORB')
        self.regmatch.invoke('Pct')

        self.docalcbut=Button(ii,text='Do Registration',command=self.docalculation,style='GREEN.TButton',width=15)
        self.docalcbut.pack(side=tkinter.TOP,fill=tkinter.Y,pady=5)


    def checkregtype(self, *args):
        if self.regtype.getvalue()=='Manual Data':
            self.regPfileSourceSel.setActive()
            self.regPfileDestSel.setActive()
            self.regMaxFeatures.setDisabled()
            self.regKeepPercent.setDisabled()
        else:
            self.regPfileSourceSel.setDisabled()
            self.regPfileDestSel.setDisabled()
            self.regMaxFeatures.setActive()
            self.regKeepPercent.setActive()


    def checkexsource(self):
        self.exsourcechan.selection_clear()
        self.exsourcedata.selection_clear()
        source=self.exsourcefile.getvalue()[0]
        self.exsourcechan.setlist(self.ps.dataFileBuffer[source]['data'].labels)        
        self.exsourcedata.setlist(self.ps.dataFileBuffer[source]['data'].labels) 
    
    def checkdestsource(self):
        self.exportdestchan.selection_clear()
        dest=self.exportdestfile.getvalue()[0]
        self.exportdestchan.setlist(self.ps.dataFileBuffer[dest]['data'].labels)

    def docalculation(self):
        pass
        #do registration
        source=self.exsourcefile.getvalue()[0]
        dest=self.exportdestfile.getvalue()[0]
        if self.regtype.getvalue()=='Manual Data':
            #check fileentries
            if not (os.path.exists(self.regPfileSourceSel.get()) and os.path.exists(self.regPfileDestSel.get())):
                print ('Choose MPM files for manual registration...')
                globalfuncs.setstatus(self.ps.displayParams.status,'Choose MPM files for manual registration...')
                return
            sourceCoordPts = self.readPMfile(self.regPfileSourceSel.get())
            sourcePts = self.convertlist(sourceCoordPts,self.ps.dataFileBuffer[source]['data'],self.ps.dataFileBuffer[source]['zoom'])
            targetCoordPts = self.readPMfile(self.regPfileDestSel.get())
            targetPts = self.convertlist(targetCoordPts,self.ps.dataFileBuffer[dest]['data'],self.ps.dataFileBuffer[dest]['zoom'])
            if sourcePts is None or targetPts is None or len(sourcePts)==0 or len(targetPts)==0:
                print ('Choose valid MPM files for manual registration...')
                globalfuncs.setstatus(self.ps.displayParams.status,'Choose valid MPM files for manual registration...')
                return
            pts = [np.array(sourcePts),np.array(targetPts)]
        else:
            pts = None
        #get source  data images
        datind=self.ps.dataFileBuffer[source]['data'].labels.index(self.exsourcechan.getvalue()[0])+2
        image = self.ps.dataFileBuffer[source]['data'].data.get(datind)
        if self.ps.dataFileBuffer[source]['zoom'][2] != -1 and self.ps.dataFileBuffer[source]['zoom'][3] != -1:
            image=image[::-1,:]
            image=image[self.ps.dataFileBuffer[source]['zoom'][1]:self.ps.dataFileBuffer[source]['zoom'][3],self.ps.dataFileBuffer[source]['zoom'][0]:self.ps.dataFileBuffer[source]['zoom'][2]]
            image=image[::-1,:]
        # get dest/target data image
        datind=self.ps.dataFileBuffer[dest]['data'].labels.index(self.exportdestchan.getvalue()[0])+2
        template = self.ps.dataFileBuffer[dest]['data'].data.get(datind)
        if self.ps.dataFileBuffer[dest]['zoom'][2] != -1 and self.ps.dataFileBuffer[dest]['zoom'][3] != -1:
            template=template[::-1,:]
            template=template[self.ps.dataFileBuffer[dest]['zoom'][1]:self.ps.dataFileBuffer[dest]['zoom'][3],self.ps.dataFileBuffer[dest]['zoom'][0]:self.ps.dataFileBuffer[dest]['zoom'][2]]
            template=template[::-1,:]        
        print("[INFO] aligning images...")
        
        if self.regtype.getvalue()=='CNN':
            reg = RegistrationCNN.CNN()
            X,Y,Z = reg.register(align_images.make_color(template), align_images.make_color(image))
            alignedcolor = tps_warp(Y,Z,align_images.make_color(image),align_images.make_color(template).shape) 
            tempGray = align_images.convert(template)
            aligned = cv2.cvtColor(alignedcolor, cv2.COLOR_BGR2GRAY)
            
        elif self.regtype.getvalue()=='SimTrans':

            image = align_images.convert(image) 
            template = align_images.convert(template)

            resample = 1
            inImage = irdTiles.resample(image,resample)
            inTemplate = irdTiles.resample(template,resample) 
            inImage,inTemplate = padArraySize(inImage, inTemplate)
            
            result = ird.similarity(inTemplate,inImage,constraints={'scale':[1.0,0.5]},numiter=3)
            print(result['success'],result['tvec']/float(resample),result['angle'],result['scale'])
            print ("translation vector:",result['tvec']/float(resample))
            print ("angle:",result['angle'])
            print ("scale:",result['scale'])
            
            aligned=ird.transform_img(inImage,tvec=result["tvec"]/float(resample),scale=result['scale'],angle=result['angle'])
            tempGray = inTemplate
            
        else:
            
            mkn=False
            if self.regmatch.getvalue()=='Knn': mkn=True
            
            aligned,t,tempGray = align_images.align_images(image, template, debug=True, maxFeatures=200000, keepPercent=0.5,
                         ptsOverride=pts, color=False, method = self.regtype.getvalue(), matchKnn = mkn )
            if t is None:
                globalfuncs.setstatus(self.ps.displayParams.status,'Homography matrix failure')
                print ('Homography matrix failure')     
                return
            #check validity?
            print ("translation x:",t['tx'])
            print ("translation y:",t['ty'])
            print ("scale x:",t['scx'])
            print ("scale y:",t['scy'])
            print ("shear:",t['sh'])
            print ("rotate:",math.degrees(t['rot']))           


    
        #popup window
        aligned = imutils.resize(aligned[::-1], width=700)
        tempGray = imutils.resize(tempGray[::-1], width=700)
        # our first output visualization of the image alignment will be a
        # side-by-side comparison of the output aligned image and the
        # template
        stacked = np.hstack([aligned, tempGray])
        # our second image alignment visualization will be *overlaying* the
        # aligned image on the template, that way we can obtain an idea of
        # how good our image alignment is
        overlay = tempGray.copy()
        output = aligned.copy()


        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        # show the two output image alignment visualizations
        cv2.imshow("Image Alignment Stacked", stacked)
        cv2.imshow("Image Alignment Overlay", output)
        cv2.waitKey(0)        


            
        #error check on shear?
        if self.regtype.getvalue() not in ['CNN','SimTrans'] and abs(t['sh'])>0.1:
            if not tkinter.messagebox.askokcancel(title='Regestration Error', message='Tranfrom shear is abnormally high.  OK or cancel?'):
                globalfuncs.setstatus(self.ps.displayParams.status,'Export with image registration cancelled')
                print ('Export with image registration cancelled')

                try:
                    cv2.destroyWindow("Image Alignment Stacked")
                    cv2.destroyWindow("Image Alignment Overlay")
                    cv2.destroyWindow("Matched Keypoints")
                except:
                    print("window closed")

                return
        #ask ok to export
        if not tkinter.messagebox.askyesno(title='Export with Regestration', message='Accept this transformation and export data?'):
            globalfuncs.setstatus(self.ps.displayParams.status,'Export with image registration cancelled')
            print ('Export with image registration cancelled')

            try:
                cv2.destroyWindow("Image Alignment Stacked")
                cv2.destroyWindow("Image Alignment Overlay")
                cv2.destroyWindow("Matched Keypoints")
            except:
                print("window closed")

            return            
        
        try:
            cv2.destroyWindow("Image Alignment Stacked")
            cv2.destroyWindow("Image Alignment Overlay")
            cv2.destroyWindow("Matched Keypoints")
        except:
            print("window closed")

        #save data
        
        for ec in self.exsourcedata.getvalue():
            datind=self.ps.dataFileBuffer[source]['data'].labels.index(ec)+2
            image = self.ps.dataFileBuffer[source]['data'].data.get(datind)
            if self.ps.dataFileBuffer[source]['zoom'][2] != -1 and self.ps.dataFileBuffer[source]['zoom'][3] != -1:
                image=image[::-1,:]
                image=image[self.ps.dataFileBuffer[source]['zoom'][1]:self.ps.dataFileBuffer[source]['zoom'][3],self.ps.dataFileBuffer[source]['zoom'][0]:self.ps.dataFileBuffer[source]['zoom'][2]]
                image=image[::-1,:]    
            
            if self.regtype.getvalue()=='CNN':
                aligneddata = tps_warp2D(Y,Z,image,template.shape) 
                aligneddata=aligneddata[::-1,:]
            elif self.regtype.getvalue()=='SimTrans':
                #image,inImageXtra = padArraySize(image,inImage)
                aligneddata = ird.transform_img(image,tvec=result["tvec"]/float(resample),scale=result['scale'],angle=result['angle'])
            else:
                aligneddata = cv2.warpPerspective(image, t['H'], t['wh'])
                aligneddata=aligneddata[::-1,:]
            #worry about zooms...
            insert=np.zeros(self.ps.dataFileBuffer[dest]['data'].data.get(0).shape,dtype=np.float32)
            if self.ps.dataFileBuffer[dest]['zoom'][2] != -1 and self.ps.dataFileBuffer[dest]['zoom'][3] != -1:
                insert=insert[::-1,:]
                insert[self.ps.dataFileBuffer[dest]['zoom'][1]:self.ps.dataFileBuffer[dest]['zoom'][3],self.ps.dataFileBuffer[dest]['zoom'][0]:self.ps.dataFileBuffer[dest]['zoom'][2]]=aligneddata
                insert=insert[::-1,:]
            else:
                insert=aligneddata
            ind=1
            ok = False
            basename = ec+"-ex"
            newname=globalfuncs.fixlabelname(basename)
            while not ok:                
                if newname in self.ps.dataFileBuffer[dest]['data'].labels:
                    newname=globalfuncs.fixlabelname(basename+'_'+str(ind))
                    ind+=1
                else:
                    ok = True
            print(self.mapdata.energy,ec,basename,newname)
            #add the channel
            self.ps.addchannel(insert,newname,fbuffer=dest)
                    
        xycs=[]
        for datind in [0,1]:
            image = self.ps.dataFileBuffer[source]['data'].data.get(datind)
            if self.ps.dataFileBuffer[source]['zoom'][2] != -1 and self.ps.dataFileBuffer[source]['zoom'][3] != -1:
                image=image[::-1,:]
                image=image[self.ps.dataFileBuffer[source]['zoom'][1]:self.ps.dataFileBuffer[source]['zoom'][3],self.ps.dataFileBuffer[source]['zoom'][0]:self.ps.dataFileBuffer[source]['zoom'][2]]
                image=image[::-1,:]    
            
            if self.regtype.getvalue()=='CNN':
                aligneddata = tps_warp2D(Y,Z,image,template.shape) 
                aligneddata=aligneddata[::-1,:]
            elif self.regtype.getvalue()=='SimTrans':
                #image,aligned = padArraySize(image,aligned)
                aligneddata = ird.transform_img(image,tvec=result["tvec"]/float(resample),scale=result['scale'],angle=result['angle'])
            else:
                aligneddata = cv2.warpPerspective(image, t['H'], t['wh'])
                aligneddata=aligneddata[::-1,:]
            #worry about zooms...
            insert=np.zeros(self.ps.dataFileBuffer[dest]['data'].data.get(0).shape,dtype=np.float32)
            if self.ps.dataFileBuffer[dest]['zoom'][2] != -1 and self.ps.dataFileBuffer[dest]['zoom'][3] != -1:
                insert=insert[::-1,:]
                insert[self.ps.dataFileBuffer[dest]['zoom'][1]:self.ps.dataFileBuffer[dest]['zoom'][3],self.ps.dataFileBuffer[dest]['zoom'][0]:self.ps.dataFileBuffer[dest]['zoom'][2]]=aligneddata
                insert=insert[::-1,:]
            else:
                insert=aligneddata            
            xycs.append(insert)
            
        #now make a mapindex for the transformed coordinates...
        mapindexdata=np.zeros((self.ps.dataFileBuffer[dest]['data'].nypts,self.ps.dataFileBuffer[dest]['data'].nxpts),dtype=np.float32)
        for my in range(mapindexdata.shape[0]):
            for mx in range(mapindexdata.shape[1]):
                xc=xycs[0][my,mx]
                yc=xycs[1][my,mx]
                if xc==yc==0: continue
                xi=globalfuncs.indexme(self.ps.dataFileBuffer[source]['data'].xvals,xc)
                yi=globalfuncs.indexme(self.ps.dataFileBuffer[source]['data'].yvals,yc)

                if xi>=len(self.ps.dataFileBuffer[source]['data'].xvals) or xi==0: 
                    #print (mx,my,xi,yi,xc,yc,"x")
                    continue
                if yi>=len(self.ps.dataFileBuffer[source]['data'].yvals): 
                    #print (mx,my,xi,yi,xc,yc,"y")
                    continue

                mpi=self.ps.dataFileBuffer[source]['data'].mapindex[yi,xi]

                mapindexdata[my,mx] = mpi
            print (mx,my)#,xi,yi,xc,yc,mpi)
        ind=1
        ok = False
        basename = "MInd-ex"
        newname=globalfuncs.fixlabelname(basename)
        while not ok:                
            if newname in self.ps.dataFileBuffer[dest]['data'].labels:
                newname=globalfuncs.fixlabelname(basename+'_'+str(ind))
                ind+=1
            else:
                ok = True
        print(self.mapdata.energy,ec,basename,newname)
        #add the channel
        self.ps.addchannel(mapindexdata,newname,fbuffer=dest)
        
        self.ps.dataFileBuffer[dest]['transform']['name']=source
        self.ps.dataFileBuffer[dest]['transform']['mapindex']=mapindexdata

        globalfuncs.setstatus(self.ps.displayParams.status,'Export with image registration complete')
        print ('Export with image registration complete')

        
    def readPMfile(self,fn):
        #read first line
        dlist=[]
        fid=open(fn,'rU')
        l=fid.readline()
        if l!='SMAK MARKERS\n':
            print('Invalid parameter file!')
            fid.close()
            return None
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
            if not invalid:
                #add point   
                np=[xp,yp]
                dlist.append(np)
            else:
                print('line not valid',invalid)    
        return dlist

    def convertlist(self,dlist,datainfo,zmxyi):
        clist=[]
        for xyp in dlist:
            xi,yi=self.invcoordinates(xyp[0],xyp[1],datainfo,zmxyi)
            clist.append([xi,yi])
        return clist
    
    def invcoordinates(self,x,y,datainfo,zmxyi):
        x=float(x)
        y=float(y)
        xind=globalfuncs.indexme(datainfo.xvals,x)
        yind=globalfuncs.indexme(datainfo.yvals,y)

        #if self.xdir==-1:
        #    xind=self.raw.shape[0]-xind-1
        #if self.ydir==-1:
        #    yind=self.raw.shape[1]-yind-1
        #zoom problems???
        if zmxyi[0:4]!=[0,0,-1,-1]:
            if xind<zmxyi[0] or xind>zmxyi[2]:
                xind=-100
            else:
                xind=xind-zmxyi[0]
            if yind<zmxyi[1] or yind>zmxyi[3]:
                yind=-100
            else:
                yind=yind-zmxyi[1]
        #if self.flipVAR.get():
        #    t=xind
        #    xind=yind
        #    yind=t
        #xp=int(xind*self.pixscale[0])+3
        #yp=int(yind*self.pixscale[1])+3
        return xind,yind