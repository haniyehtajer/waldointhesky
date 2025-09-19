import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.convolution


 # example usage  peak_x_values, peak_y_values, convolved_image = centroid_tools.convolve_peaks(data, threshold = 14, std=2.5)

def get_image(path):
    data = fits.getdata(path)
    return data

def get_peaks(image, threshold):
    peak_x_values = []
    peak_y_values = []
    for j in range(np.shape(image)[1]):  # x
        for i in range(np.shape(image)[0]):  # y
            try:
                if image[i, j] > threshold:
                    point = image[i, j]
                    north = image[i-1, j]
                    south = image[i+1, j]
                    east = image[i, j+1]
                    west = image[i, j-1]
                    ne = image[i-1, j+1]
                    nw = image[i-1, j-1]
                    se = image[i+1, j+1]
                    sw = image[i+1, j-1]
                    max_ = np.argmax(np.array([point, np.mean([north, south, east, west, ne, nw, se, sw])]))
                    if max_ == 0:
                        peak_x_values.append(j)
                        peak_y_values.append(i)
            except IndexError:
                pass  # skip edges
    return np.array(peak_x_values), np.array(peak_y_values)



def convolve_peaks(data, threshold, std):
    gauss = Gaussian2DKernel(x_stddev=std)

    convolved_image = convolve(data, gauss)

    peak_x_values, peak_y_values = get_peaks(data, threshold = threshold)
    return peak_x_values, peak_y_values, convolved_image