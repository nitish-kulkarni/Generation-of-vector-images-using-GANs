import numpy as np
import os
import svgwrite

seed=10
num_samples=1000
loaddir='sketch_ful'
savedir='sketch_sml'
visldir='sketch_vis'

def to_big_strokes(stroke, max_len=250):
  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  if(l>max_len):
    result=stroke[:maxlen,:]
    result[maxlen-1,4]=1
  else:
    result[0:l, 0:2]=stroke[:, 0:2]
    result[0:l, 3]=stroke[:, 2]
    result[0:l, 2]=1-result[0:l, 3]
    result[l:, 4]=1
  return result

def get_bounds(data, factor=1):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0
  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)
  return (min_x, max_x, min_y, max_y)

def draw_strokes(data, factor=1, svg_filename='sample.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()

def visualize_sketches(oneclass,classname):
	direcname=os.path.join(visldir,classname)
	if not os.path.exists(direcname):
		os.mkdir(direcname)
	for sample in range(len(oneclass)):
		filename=os.path.join(direcname,classname+'%'+str(oneclass[sample,1]).zfill(5)+'.svg')
		draw_strokes(oneclass[sample,0],svg_filename=filename)

with open('categories.txt') as file:
	for line in file:
		ipfile=os.path.join(loaddir,'sketchrnn%2F'+line.strip()+'.npz')
		oneclass=np.load(ipfile)
		oneclass=oneclass['train']
		classnum=np.arange(len(oneclass))+1
		oneclass=np.vstack((oneclass,classnum)).T
		np.random.seed(seed)
		np.random.shuffle(oneclass)
		oneclass=oneclass[:num_samples,:]
		visualize_sketches(oneclass,line.strip())
		for i in range(len(oneclass)):
			oneclass[i,0]=to_big_strokes(oneclass[i,0])
		opfile=os.path.join(savedir,'sketchrnn%2F'+line.strip()+'.npy')
		np.save(opfile,oneclass)
		print line.strip()+' Done......'