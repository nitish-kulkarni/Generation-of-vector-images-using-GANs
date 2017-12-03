from keras.preprocessing import image
from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input	
import pickle
import os
import svgwrite

from sketch_train import *
from sketch_model import *
from sketch_utils import *
from rnn import *

def load_image(path):
	img = image.load_img(path, target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return np.asarray(x)

def load_encoding_model():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	return model

def get_encoding(model, img):
	global counter
	image = load_image(img)
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	return pred

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = './tmp/sketch_rnn/svg/sample.svg'):
	tf.gfile.MakeDirs(os.path.dirname(svg_filename))
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

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
 	def get_start_and_end(x):
		x = np.array(x)
		x = x[:, 0:2]
		x_start = x[0]
		x_end = x.sum(axis=0)
		x = x.cumsum(axis=0)
		x_max = x.max(axis=0)
		x_min = x.min(axis=0)
		center_loc = (x_max+x_min)*0.5
		return x_start-center_loc, x_end
 
	x_pos = 0.0
	y_pos = 0.0
	result = [[x_pos, y_pos, 1]]
	for sample in s_list:
		s = sample[0]
		grid_loc = sample[1]
		grid_y = grid_loc[0]*grid_space+grid_space*0.5
		grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
		start_loc, delta_pos = get_start_and_end(s)

		loc_x = start_loc[0]
		loc_y = start_loc[1]
		new_x_pos = grid_x+loc_x
		new_y_pos = grid_y+loc_y
		result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

		result += s.tolist()
		result[-1][2] = 1
		x_pos = new_x_pos+delta_pos[0]
		y_pos = new_y_pos+delta_pos[1]

	return np.array(result)

def encode(img_embedding):
	return sess.run(eval_model.batch_z, feed_dict={eval_model.image: [img_embedding]})[0]

def decode(z_input, temperature=1.0, factor=0.2):
	strokes, _ = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=[z_input])
	return strokes


data_dir = '.'
model_dir = './tmp/sketch_rnn/models/default'

sketch_dir='sketch_sml'
image_dir='pixel_images'
clusterinfo=pickle.load(open('clusters.p','rb'))


_, _, test_set, hps_model, eval_hps_model, sample_hps_model = load_env(data_dir, model_dir)

reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# loads the weights from checkpoint into our model
load_checkpoint(sess, model_dir)

categories=[]
with open('categories.txt') as f:
	for line in f:
		categories+=[line.strip()]

image_embed={}
model=load_encoding_model()
for category in categories:
	catimagedir=os.path.join(image_dir,category)
	for img in os.listdir(catimagedir):
		inputimage=os.path.join(catimagedir,img)
		image_embed[img.split('.')[0]]=get_encoding(model,inputimage)
	print category +'Done'

# Get the list of test images
test_cat_num=[]
test = open('test.txt')
for line in test:
	line = line.replace('\n', '')
	test_cat_num+=[line]

output=[]
for category in categories:
	inputfile=os.path.join(sketch_dir,'sketchrnn%2F'+category+'.npy')
	sketches=np.load(inputfile)
	for i in range(len(sketches)):

		# Infer on the testing set
		cat_num = category+'%'+str(sketches[i,1]).zfill(5)
		if cat_num in test_cat_num:
			corrimage=category+'%'+str(clusterinfo[category][cat_num]).zfill(2)

			# Target sketch
			sketch = sketches[i,0]

			# Feed the image embedding to get the predicted sketch
			z = encode(image_embed[corrimage])
			pred_sketch = decode(z)

			# Convert to normal strokes
			sketch = to_normal_strokes(sketch)
			pred_sketch = to_normal_strokes(pred_sketch)

			output+=[[cat_num,corrimage,sketch,pred_sketch]]

			print(cat_num + ' done')

pickle.dump(output,open('sketch_results.p','wb'))
