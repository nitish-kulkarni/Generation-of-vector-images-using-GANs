import numpy as np

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from process_data import *

def get_class_data(data, cls_name):
    return [d for d in data if d[0].split('%')[0] == cls_name]

def get_sketch_from_id(sketches, sketch_id):
    sketches = [d for d in sketches if int(d[0].split('%')[1]) == sketch_id]
    return sketches[0] if len(sketches) == 1 else []

def sketch_to_xy(sketch):
    xy = sketch[:, :2][::-1].cumsum(axis=0)
    xy[:, 0] *= -1
    
    # MinMax Normalize
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    
    def normalise(a, amin, amax):
        l = amax - amin
        a -= amin
        return (a - amin) / l if l > 0 else a

    # Reposition
    xy[:, 0] = normalise(xy[:, 0], xmin, xmax)
    xy[:, 1] = normalise(xy[:, 1], ymin, ymax)
    xy -= xy[0,:] # Start at origin

    return xy

def num_strokes(sketch):
    return sketch[:, 2].sum()

def plot_xy(xy):
    plt.plot(xy[:, 0], xy[:, 1])
    plt.plot(xy[0, 0], xy[0, 1], 'ro')
    plt.plot(xy[-1, 0], xy[-1, 1], 'bo')
    plt.show()

def dtw(xy1, xy2):
    distance, path = fastdtw(xy1, xy2, dist=euclidean)
    return distance / len(path)

def min_dtw_dist(sketch, sketches):
    xy = sketch_to_xy(sketch)
    return min([dtw(xy, sketch_to_xy(i)) for i in sketches])

def pred_sketch_scores(data, cat, img_id, score_func=min_dtw_dist, pred_sketches=None):
    sketches = get_sketches(data, cat, img_id)
    if not pred_sketches:
        pred_sketches = get_pred_sketches(data, cat, img_id)
    excluded_sketches = [get_sketches(data, cat, img_id, exclude_id=sk_id) for sk_id in get_sketch_ids(data, cat, img_id)]
    sketch_scores = np.array([score_func(sketches[i], excluded_sketches[i]) for i in range(len(excluded_sketches))])
    distribution = stats.norm(sketch_scores.mean(), sketch_scores.std())

    pred_scores = np.array([score_func(pred_sketch, sketches) for pred_sketch in pred_sketches])
    return np.array([1 - distribution.cdf(i) for i in pred_scores])

def sketch_scores(data, cat, img_id, score_func=min_dtw_dist):
    sketches = get_sketches(data, cat, img_id)
    excluded_sketches = [get_sketches(data, cat, img_id, exclude_id=sk_id) for sk_id in get_sketch_ids(data, cat, img_id)]
    sketch_scores = np.array([score_func(sketches[i], excluded_sketches[i]) for i in range(len(excluded_sketches))])
    distribution = stats.norm(sketch_scores.mean(), sketch_scores.std())
    return np.array([1 - distribution.cdf(i) for i in sketch_scores])

# def num_stroke_acc(deta):
#     num_strokes_test = 
#     for cat, imid in deta:

# def score()
# def dtw_score(data, sketch_id, cls_name):
#     sketches = get_class_data(data, cls_name)
#     other_sketches = [d[2] for d in sketches if int(d[0].split('%')[1]) != sketch_id]
#     # sketch = get_sketch_from_id(sketches, sketch_id)[2]
#     sketch = get_sketch_from_id(get_class_data(data, 'car'), sketch_id)[2]
#     return min_dtw_dist(sketch, other_sketches)

