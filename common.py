import plotly.express as px
import numpy as np
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'Apple SD Gothic Neo'


def load_ir_in_csv(filename: str) -> np.ndarray:
    WIDTH = 320
    return np.genfromtxt('./plant-ir/121.csv', delimiter=',',
                         skip_header=2, usecols=range(1, WIDTH+1))


def show_image(img, title=""):
    fig = px.imshow(img, title=title)
    fig.show()


def show_histogram(img):
    fig = px.histogram(img.ravel())
    fig.show()


def get_average_tempeature(ir: np.ndarray, mask: np.ndarray):
    assert len(ir.shape) == 2
    assert len(mask.shape) == 2
    num_region_pixels = 0
    sum_temperature = 0
    for y, x in np.ndindex(ir.shape):
        if mask[y, x] != 0:
            sum_temperature += ir[y, x]
            num_region_pixels += 1

    return sum_temperature / num_region_pixels
