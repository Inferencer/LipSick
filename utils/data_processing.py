import csv
import numpy as np
import random


def load_landmark_openface(csv_path):
    '''
    load openface landmark from .csv file
    '''
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data_all = [row for row in reader]
    x_list = []
    y_list = []
    for row_index,row in enumerate(data_all[1:]):
        frame_num = float(row[0])
        if int(frame_num)!= row_index+1:
            return None
        x_list.append([float(x) for x in row[5:5+68]])
        y_list.append([float(y) for y in row[5+68:5+68 + 68]])
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    landmark_array = np.stack([x_array,y_array],2)
    return landmark_array


# Example fix inside the compute_crop_radius function
def compute_crop_radius(video_size, landmark_data_clip, random_scale=None):
    video_w, video_h = video_size[0], video_size[1]
    landmark_max_clip = np.max(landmark_data_clip, axis=1)
    if random_scale is None:
        random_scale = 1.1  # Set a fixed scaling factor instead of a random one could play around with it, 1.05 was the default but i felt it was too small

    radius_h = (landmark_max_clip[:, 1] - landmark_data_clip[:, 29, 1]) * random_scale
    radius_w = (landmark_data_clip[:, 54, 0] - landmark_data_clip[:, 48, 0]) * random_scale
    radius_clip = np.max(np.stack([radius_h, radius_w], axis=1), axis=1) // 2
    
    radius_max = np.max(radius_clip)
    radius_max = (int(radius_max / 4) + 1) * 4
    
    clip_min_h = landmark_data_clip[:, 29, 1] - radius_max
    clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 2 + radius_max // 4
    clip_min_w = landmark_data_clip[:, 33, 0] - radius_max - radius_max // 4
    clip_max_w = landmark_data_clip[:, 33, 0] + radius_max + radius_max // 4
    
    condition1 = np.min(clip_min_h) >= 0
    condition2 = np.min(clip_min_w) >= 0
    condition3 = np.max(clip_max_h) <= video_h
    condition4 = np.max(clip_max_w) <= video_w

    if condition1 and condition2 and condition3 and condition4:
        return True, radius_max
    else:
        return False, None
    
    return True, radius_max, lowest, highest, average

