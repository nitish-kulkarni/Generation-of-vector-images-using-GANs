from keras.preprocessing import image
from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input	
import pickle
import os
import svgwrite
import sys

from sketch_train import *
from sketch_model import *
from sketch_utils import *
from rnn import *

from evaluation.process_data import save_sketch_as_png

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

def encode(img_embedding):
	return sess.run(eval_model.batch_z, feed_dict={eval_model.image: [img_embedding]})[0]

def decode(z_input, temperature=1.0, factor=0.2):
	strokes, _ = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=[z_input])
	return strokes


model_dir = './tmp/sketch_rnn/models/default'
image_path = sys.argv[1]

_, _, test_set, hps_model, eval_hps_model, sample_hps_model = load_env(None, model_dir)

reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# loads the weights from checkpoint into our model
load_checkpoint(sess, model_dir)

# Get the image embedding
encode_model = load_encoding_model()
img_embed = get_encoding(encode_model,image_path)

# Feed the image embedding to get the predicted sketch
z = encode(img_embed)
sketch = decode(z)

# Convert to normal strokes
sketch = to_normal_strokes(sketch)

# Draw the strokes and save
name, _ = image_path.split('.')
save_sketch_as_png(sketch, filename = '%s_sketch.png' % name)
