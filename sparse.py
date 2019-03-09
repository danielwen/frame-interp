import sys
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (25,25),
                  maxLevel = 4,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def get_sequence_paths(folder):
    pattern = os.path.join(folder, "**", "*.jpg")
    paths = glob(pattern, recursive=True)
    sequence_paths = {}

    for path in paths:
        directory, _ = os.path.split(path)
        if directory not in sequence_paths:
            sequence_paths[directory] = []
        sequence_paths[directory].append(path)

    return sequence_paths

def load_preprocess(path):
    img = cv.imread(path)
    return img

def get_target_indices(flow, h, w):
    I, J = np.mgrid[:h, :w]
    I = np.clip(I + flow[:, :, 0], 0, h - 1)
    J = np.clip(J + flow[:, :, 1], 0, w - 1)
    return I, J

def fetch(flow, target):
    h, w, _ = target.shape
    I, J = get_target_indices(flow, h, w)
    return target[I, J, :]

def put(source, flow):
    h, w, _ = source.shape
    I, J = get_target_indices(flow, h, w)
    result = np.zeros_like(source)
    mask = np.ones((h, w, 1))
    result[I, J] = source
    mask[I, J] = 0.
    return result, mask

def imshow(im, ax):
    ax.imshow(np.flip(np.round(255 * im).astype("int"), -1))

def interpolate(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    frame1_float = frame1.astype("float64") / 255
    frame2_float = frame2.astype("float64") / 255

    p1 = cv.goodFeaturesToTrack(gray1, mask = None, **feature_params)
    print(p1.shape)
    p2, st, err = cv.calcOpticalFlowPyrLK(gray1, gray2, p1, None, **lk_params)

    fig = plt.figure()
    ax = fig.gca()
    imshow((frame1_float + frame2_float) / 2, ax)
    good1 = p1[st == 1]
    good2 = p2[st == 1]
    lines = np.concatenate([good1, good2], axis=-1).reshape(-1, 2, 2)
    lc = LineCollection(lines)
    ax.add_collection(lc)
    # ax.autoscale()
    plt.show()


    # src2tgt = fetch(flow_int, frame2_float)
    # values = (frame1_float + src2tgt) / 2
    # result, mask = put(values, half_flow)
    # result += mask * frame2_float
    # result_int = np.round(255 * result).astype("int")
    assert False
    # return result

def main():
    folder = sys.argv[1]
    sequence_paths = get_sequence_paths(folder)
    dists = []

    for directory, paths in tqdm(sequence_paths.items()):
        print(directory)
        images = list(map(load_preprocess, sorted(paths)))

        for i in range(len(images) - 2):
            frame1 = images[i]
            mid_frame = images[i + 1]
            frame2 = images[i + 2]
            interp = interpolate(frame1, frame2)
            dist = np.mean((interp - mid_frame)**2)
            dists.append(dist)

    avg_dist = 255 * np.mean(dist)
    print(avg_dist)


main()
