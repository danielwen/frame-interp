import sys
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

def imshow(im):
    plt.figure()
    plt.imshow(np.flip(np.round(255 * im).astype("int"), -1))

def interpolate(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    frame1_float = frame1.astype("float64") / 255
    frame2_float = frame2.astype("float64") / 255

    flow = cv.calcOpticalFlowFarneback(gray1, gray2, None,
        0.5, 4, 25, 5, 7, 1.5, 0)
    flow_int = np.round(flow).astype("int")
    half_flow = np.round(flow / 2).astype("int")

    src2tgt = fetch(flow_int, frame2_float)
    values = (frame1_float + src2tgt) / 2
    imshow((frame1_float + frame2_float) / 2)
    h, w, _ = flow.shape
    y, x = np.mgrid[:h, :w][:, 0:h:16, 0:w:14]
    u = flow[0:h:16, 0:w:14, 0]
    v = flow[0:h:16, 0:w:14, 1]
    plt.quiver(x, y, u, -v)
    imshow(src2tgt)
    # imshow(values)
    imshow((frame2_float + src2tgt) / 2)
    plt.show()
    result, mask = put(values, half_flow)
    result += mask * frame2_float
    # result_int = np.round(255 * result).astype("int")
    # plot(frame1, frame2, result_int, flow)
    assert False
    return result

def plot(frame1, frame2, result, flow):
    # plt.figure()
    # plt.imshow(frame1)
    # plt.figure()
    plt.imshow(result)
    # plt.figure()
    # plt.imshow(frame2)
    # plt.figure()
    # h, w, _ = flow.shape
    # y, x = np.mgrid[:h, :w][:, 0:h:16, 0:w:14]
    # u = flow[0:h:16, 0:w:14, 0]
    # v = flow[0:h:16, 0:w:14, 1]
    # plt.quiver(x, y, u, v)
    plt.show()

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
