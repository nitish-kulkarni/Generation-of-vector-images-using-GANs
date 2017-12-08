import numpy as np

SKETCHES = 'sketches'
PRED_SKETCHES = 'pred_sketches'
SKETCH_IDS = 'sketch_ids'

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

