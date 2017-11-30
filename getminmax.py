import os
import numpy as np
files=os.listdir('sketch_sml')
maxx=[]
minx=[]
for file in files:
	ipfile=os.path.join('sketch_sml',file)
	out=np.load(ipfile)
	for i in range(len(out)):
		minx+=[np.min(out[i,0])]
		maxx+=[np.max(out[i,0])]
print len(minx)
print len(maxx)
minxm=np.mean(np.array(minx))
maxxm=np.mean(np.array(maxx))
print minxm
print maxxm