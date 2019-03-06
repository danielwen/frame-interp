import sys
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2

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
    img = cv2.imread(path).astype("float64")
    img /= 255.0
    return img

def main():
    folder = sys.argv[1]
    sequence_paths = get_sequence_paths(folder)
    start_dists = []
    end_dists = []
    interp_dists = []

    for paths in tqdm(sequence_paths.values()):
        images = list(map(load_preprocess, sorted(paths)))

        for i in range(len(images) - 2):
            start_frame = images[i]
            mid_frame = images[i + 1]
            end_frame = images[i + 2]
            interp = (start_frame + end_frame) / 2
            for dists, pred in zip([start_dists, end_dists, interp_dists],
                    [start_frame, end_frame, interp]):
                dist = np.mean((pred - mid_frame)**2)
                dists.append(dist)

    avg_dists = tuple([255 * np.mean(dists)
        for dists in (start_dists, end_dists, interp_dists)])
    print("Identity1: %.2f | Identity2: %.2f | NaiveInterp: %.2f" % avg_dists)


main()
