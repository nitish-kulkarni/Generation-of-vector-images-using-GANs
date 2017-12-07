from sklearn.cluster import KMeans
import numpy as np
import pickle
import os
import cv2
from collections import Counter

loaddir='sketch_png'
savedir='sketch_cls'
outsize=128
numclusters=5
finaldict={}
with open('categories.txt') as file:
	for line in file:
		ipdir=os.path.join(loaddir,line.strip())
		classfile=os.listdir(ipdir)
		classdata=np.zeros((len(classfile),outsize**2))
		for cfile in range(len(classfile)):
			ipfilename=os.path.join(ipdir,classfile[cfile])
			tempimage=cv2.imread(ipfilename,0)
			tempimage=cv2.resize(tempimage,(outsize,outsize))
			classdata[cfile,:]=tempimage.ravel()
			classfile[cfile]=classfile[cfile].split('.')[0]
		classdata=classdata/255.
		kmeansout=KMeans(n_clusters=numclusters,random_state=0).fit(classdata)
		kmeanscls=list(kmeansout.labels_)
		print str(Counter(kmeanscls))
		finaldict[line.strip()]=dict(zip(classfile,kmeanscls))
		opdir=os.path.join(savedir,line.strip())
		if not os.path.exists(opdir):
			os.mkdir(opdir)
		outputfiles=kmeansout.cluster_centers_.reshape((numclusters,outsize,outsize))
		for i in range(numclusters):
			opfilename=os.path.join(opdir,line.strip()+'%'+str(i).zfill(2)+'.png')
			cv2.imwrite(opfilename,(outputfiles[i,:,:]*255).astype(int))
		print line.strip()+' Done......'

pickle.dump(finaldict,open('clusters.p','wb'))