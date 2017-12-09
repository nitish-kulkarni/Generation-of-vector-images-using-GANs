import pickle
import numpy as np

from evaluation.process_data import *
import evaluation.score
import random
import cv2

with open('categories.txt', 'rb') as fp:
    categories = {}
    count = 0
    for category in fp:
        category = category.replace('\n', '').replace('\r', '')

        categories[category] = count
        count += 1

with open('sketch_results.p', 'rb') as fp:
    data = pickle.load(fp)

data = processed_data(data)

sketch_imgs = []
pred_sketch_imgs = []
cats = []

for cat, img_id in data:
    sketches = get_sketches(data, cat, img_id)
    sketch_ids = get_sketch_ids(data, cat, img_id)
    pred_sketches = get_pred_sketches(data, cat, img_id)

    for i, sketch_id in enumerate(sketch_ids):
        sketch_name = '%s_sketch_%d' % (cat, sketch_id)
        pred_sketch_name = '%s_pred_sketch_%d' % (cat, sketch_id)

        save_sketch_as_png(sketches[i], filename='sketch.png')
        save_sketch_as_png(pred_sketches[i], filename='pred_sketch.png')

        sketch_img = cv2.imread('sketch.png', cv2.IMREAD_GRAYSCALE)
        sketch_img = cv2.resize(sketch_img, (100, 100)).reshape((-1))

        pred_sketch_img = cv2.imread('pred_sketch.png', cv2.IMREAD_GRAYSCALE)
        pred_sketch_img = cv2.resize(pred_sketch_img, (100, 100)).reshape((-1))

        sketch_imgs.append(sketch_img)
        pred_sketch_imgs.append(pred_sketch_img)
        cats.append(categories[cat])

        print(cat, i)

np.save('sketch_imgs.npy', np.array(sketch_imgs))
np.save('pred_sketch_imgs.npy', np.array(pred_sketch_imgs))
np.save('categories.npy', np.array(cats))
