
import pickle
import numpy as np

from process_data import *
import score

with open('sketch_results.p', 'rb') as fp:
    data = pickle.load(fp)
data = processed_data(data)

count = 0
all_scores = []
for cat, img_id in data:
    scores = score.sketch_scores(data, cat, img_id)
    all_scores.append(scores)
    print cat, img_id, scores.mean()
    if count == 5:
        break

