import numpy as np
import matplotlib.pyplot as plt

SKETCHES = 'sketches'
PRED_SKETCHES = 'pred_sketches'
SKETCH_IDS = 'sketch_ids'

def strokes_from_sketch(sketch):
    indices = np.where(sketch[:, 2] == 1)[0] + 1
    strokes = []
    prev_idx = 0
    stroke = stroke_to_xy(sketch)
    for idx in indices:
        strk = stroke[prev_idx:idx, :]
        if len(strk) > 0:
            strokes.append(strk)
        prev_idx = idx
    return strokes

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

def save_sketch_as_png(sketch, filename='sample.png'):
    plt.figure(1)
    strokes = strokes_from_sketch(sketch)
    for xy in strokes:
        plt.plot(xy[:, 0], xy[:, 1], 'b')
#         plt.plot(xy[0, 0], xy[0, 1], 'ro')
#         plt.plot(xy[-1, 0], xy[-1, 1], 'bo')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(1)

def plot_sketch(sketch):
    plt.figure()
    strokes = strokes_from_sketch(sketch)
    for xy in strokes:
        plt.plot(xy[:, 0], xy[:, 1], 'b')
#         plt.plot(xy[0, 0], xy[0, 1], 'ro')
#         plt.plot(xy[-1, 0], xy[-1, 1], 'bo')
    plt.show()

def stroke_to_xy(stroke):    
    return sketch_to_xy(stroke)

def processed_data(data):
    all_data = {}
    for d in data:
        cat = d[0].split('%')[0]        # category
        img_id = int(d[1].split('%')[1])   # image_name
        sketch_id = int(d[0].split('%')[1])

        key = (cat, img_id)
        if key not in all_data:
            all_data[key] = {SKETCHES: [], PRED_SKETCHES: [], SKETCH_IDS: []}
        all_data[key][SKETCHES].append(d[2])
        all_data[key][PRED_SKETCHES].append(d[3])
        all_data[key][SKETCH_IDS].append(sketch_id)
    return all_data

def get_sketches(data, cat, img_id, exclude_id=None):
    sketches = data[(cat, img_id)][SKETCHES]
    if exclude_id != None:
        ids = data[(cat, img_id)][SKETCH_IDS]
        sketches = [sketches[i] for i in range(len(sketches)) if ids[i] != exclude_id]
    return sketches

def get_sketch_ids(data, cat, img_id):
    return data[(cat, img_id)][SKETCH_IDS]

def get_pred_sketches(data, cat, img_id):
    return data[(cat, img_id)][PRED_SKETCHES]

