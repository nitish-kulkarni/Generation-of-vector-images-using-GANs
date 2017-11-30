#Python3 Code
import cairosvg
import os

loaddir='sketch_vis'
savedir='sketch_png'

with open('categories.txt') as file:
	for line in file:
		ipdir=os.path.join(loaddir,line.strip())
		opdir=os.path.join(savedir,line.strip())
		if not os.path.exists(opdir):
			os.mkdir(opdir)
		classfile=os.listdir(ipdir)
		for cfile in classfile:
			ipfilename=os.path.join(ipdir,cfile)
			sfile=cfile.split('.')[0]+'.png'
			opfilename=os.path.join(opdir,sfile)
			svg_code=''
			with open(ipfilename) as forg:
				for oneline in forg:
					svg_code+=oneline.strip()
			cairosvg.svg2png(bytestring=svg_code,write_to=opfilename)
		print (line.strip()+' Done......')