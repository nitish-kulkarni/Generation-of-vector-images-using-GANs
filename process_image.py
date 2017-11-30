import cv2
import numpy as np
import os

loaddir='raw_images'
savedir='pixel_images'

with open('categories.txt') as file:
	for line in file:
		cat=line.strip()
		ipdir=os.path.join(loaddir,cat)
		opdir=os.path.join(savedir,cat)
		if not os.path.exists(opdir):
			os.mkdir(opdir)
		ipfiles=os.listdir(ipdir)
		assert len(ipfiles)==5
		for i in range(len(ipfiles)):
			fname=ipfiles[i]
			checkname=fname.split('%')
			assert checkname[0]==cat and int(checkname[1][:2])==i
			extension=checkname[1].split('.')[1]
			ipfilename=os.path.join(ipdir,fname)
			opfilename=os.path.join(opdir,fname.replace(extension,'jpg'))
			img=cv2.imread(ipfilename)
			img=cv2.resize(img,(299,299))
			cv2.imwrite(opfilename,img)

