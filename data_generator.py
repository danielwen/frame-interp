import os
import glob
import numpy as np
import cv2

'''
    Input: 
        folder - String
    Output: 
        dictionary - String -> String array
        contains mapping from video name to an array containing frame names
'''
def get_sequence_paths(folder):
    pattern = os.path.join(folder, "**", "*.jpg")
    paths = glob.glob(pattern, recursive=True)
    sequence_paths = {}

    for path in paths:
        directory, _ = os.path.split(path)
        if directory not in sequence_paths:
            sequence_paths[directory] = []
        sequence_paths[directory].append(path)

    return sequence_paths


def load_preprocess(path):
    img = cv2.imread(path).astype("float32")
    return np.flip(img, axis=-1) / 255.


def horizontal_flip(image):
    return np.flip(image, axis=1)


class DataGenerator(object):
    def __init__(self, folder, batch_size, image_size, latent_size, n_context,
            test=False):
        sequence_paths = get_sequence_paths(folder)
        self.batch_size = batch_size
        self.image_size = image_size
        self.latent_size = latent_size
        self.n_context = n_context
        self.test = test
        self.data = []

        for paths in sequence_paths.values():
            paths = tuple(sorted(paths))

            for i in range(len(paths) - n_context):
                self.data.append(paths[i : i + n_context + 1])

        self.N = len(self.data)
        self.steps = (self.N + batch_size - 1) // batch_size
        self.reset()

    def reset(self):
        self.indices = np.random.permutation(self.N)
        self.i = 0

    def make_batch(self, indices):
        n = len(indices)
        frames = np.zeros((n, self.image_size, self.image_size, 3))
        contexts = np.zeros((n, self.image_size, self.image_size, 3 * self.n_context))
        noises = np.random.normal(size=(n, self.latent_size))
        dummy = np.zeros((n, 1))

        for i, idx in enumerate(indices):
            images = [load_preprocess(path) for path in self.data[idx]]

            if np.random.choice([True, False]):
                images = [horizontal_flip(image) for image in images]

            before, frame, after = images
            context = np.concatenate([before, after], axis=-1)

            frames[i] = frame
            contexts[i] = context

        if self.test:
            return [noises, contexts], [frames]

        return [frames, noises, contexts], [frames, dummy]

    def __iter__(self):
        return self

    def __next__(self):
        indices = self.indices[self.i : self.i + self.batch_size]
        self.i += self.batch_size

        if self.i >= self.N:
            self.reset()

        return self.make_batch(indices)


if __name__ == "__main__":
    folder = "DAVIS_Train_Val"
    data_gen = DataGenerator(folder, 16, 256, 512, 2)
    result = next(data_gen)