
import pickle
import numpy as np

from process_data import *
from score import *
with open('sketch_results.p', 'rb') as fp:
    data = pickle.load(fp)
data = processed_data(data)

count = 0
all_scores = []
all_pred_scores = []
for cat, img_id in data:
    scores = sketch_scores(data, cat, img_id)
    pred_scores = pred_sketch_scores(data, cat, img_id)
    if len(scores) > 0:
        all_scores.append(scores)
        all_pred_scores.append(pred_scores)
        print cat, img_id, scores.mean(), pred_scores.mean()
        # if count == 5:
        #     break
    count += 1


# for cat, img_id in data:
#     sketches = get_sketches(data, cat, img_id)
#     sketch_ids = get_sketch_ids(data, cat, img_id)
#     pred_sketches = get_pred_sketches(data, cat, img_id)

#     for i, sketch_id in enumerate(sketch_ids):
#         sketch_name = '%s_sketch_%d' % (cat, sketch_id)
#         pred_sketch_name = '%s_pred_sketch_%d' % (cat, sketch_id)
#     save_sketch_as_png(sketches[i], filename='results/%s.png' % sketch_name)
#     save_sketch_as_png(pred_sketches[i], filename='results/%s.png' % pred_sketch_name)
