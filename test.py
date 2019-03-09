import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path1 = "challenge/JPEGImages/480p/ocean-birds/00000.jpg"
path2 = "challenge/JPEGImages/480p/ocean-birds/00002.jpg"

def imshow(im):
    plt.figure()
    plt.imshow(np.flip(np.round(255 * im).astype("int"), -1))

def get_target_indices(flow, h, w):
    I, J = np.mgrid[:h, :w]
    I = np.clip(I + flow[:, :, 0], 0, h - 1)
    J = np.clip(J + flow[:, :, 1], 0, w - 1)
    return I, J

def fetch(flow, target):
    h, w, _ = target.shape
    I, J = get_target_indices(flow, h, w)
    return target[I, J, :]

flow = np.load("flow.npy")
frame1 = cv.imread(path1)
frame2 = cv.imread(path2)
frame1_float = frame1.astype("float64") / 255
frame2_float = frame2.astype("float64") / 255

flow_int = np.round(flow).astype("int")

src2tgt = fetch(flow_int, frame2_float)

imshow((frame1_float + frame2_float) / 2)
h, w, _ = flow.shape
y, x = np.mgrid[:h, :w][:, 0:h:16, 0:w:14]
u = flow[0:h:16, 0:w:14, 0]
v = flow[0:h:16, 0:w:14, 1]
plt.quiver(x, y, u, -v)
imshow(src2tgt)
imshow((frame1_float + src2tgt) / 2)
imshow((frame2_float + src2tgt) / 2)
plt.show()
