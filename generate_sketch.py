import pickle
import numpy as np
import svgwrite

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

def to_normal_strokes(big_stroke):
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


sketch=pickle.load(open('simplesketch.p','rb'))
factor=1
temp=sketch[:,2:]
temp=np.exp(temp)
temp=temp/np.sum(temp,axis=1,keepdims=True)
temp[:,1]=temp[:,1]*factor
temp2=np.argmax(temp,axis=1)
newsketch=np.zeros((sketch.shape[0],3))
newsketch[np.arange(sketch.shape[0]),temp2]=1
sketch[:,2:]=newsketch
sketch[:,:2]=200*sketch[:,:2]

sketch3=to_normal_strokes(sketch)
draw_strokes(sketch3)

