import numpy as np


def gaussian2D(radius, mu):
    """
    2D Gaussian function
        radius: grid of "radius" values distance from some source center
                (defined by calculated centroids)
        mu: width of the Gaussian (standard deviation)
    """
    return 1/(mu**2*2*np.pi)*np.exp(-0.5*((radius)/mu)**2)

def make_psf_image(data_image, centroid_x, centroid_y, psf_mu):
    """
    makes an image (same size as data_image) containing a single gaussian PSF
    centered at centroid_x, centroid_y.
        image_shape: (y,x) shape of the image cutout
        centroid_x: x position of the star centroid
        centroid_y: y position of the star centroid
    """
    # make a meshgrid (~coordinates) based on the image shape
    xx, yy = np.meshgrid(range(0, data_image.shape[1]), range(0, data_image.shape[0]))
    # calculate the radius from the center (centroid_x, centroid_y) at every position in the meshgrid
    # allow centroid_x/centroid_y to be scalars or 1D arrays of centroids
    centroid_x = np.asarray(centroid_x)
    centroid_y = np.asarray(centroid_y)

    if centroid_x.ndim == 0 and centroid_y.ndim == 0:
        # single centroid: keep original 2D shape
        xx_rel_flat = xx - float(centroid_x)
        yy_rel_flat = yy - float(centroid_y)
    else:
        # multiple centroids: produce a stack with shape (H, W, N)
        centroid_x = centroid_x.ravel()
        centroid_y = centroid_y.ravel()
        xx_rel_flat = xx[..., None] - centroid_x[None, None, :]
        yy_rel_flat = yy[..., None] - centroid_y[None, None, :]
    radius = np.sqrt(xx_rel_flat **2 + yy_rel_flat**2)
    # define the PSF image using radius and the size of the star ~psf_mu
    psf_image = gaussian2D(radius=radius, mu=psf_mu)
    return psf_image

def second_moment(image_cutout, centroid_x, centroid_y, start_x, start_y):
    """
    calculate the second moment of an image cutout around a centroid position
    """
    x_size, y_size = image_cutout.shape
    xx, yy = np.meshgrid(np.arange(start_x, start_x + x_size),
                         np.arange(start_y, start_y + y_size))
    x_width = np.sqrt(np.sum((image_cutout*(xx - centroid_x))**2))/np.sqrt(np.sum(image_cutout**2))
    y_width = np.sqrt(np.sum((image_cutout*(yy - centroid_y))**2))/np.sqrt(np.sum(image_cutout**2))
    return (x_width, y_width)

def find_best_second_moment(centroid_xs, centroid_ys, img_data):
    """
    find the second moment at every centroid position provided, and return the 
    approximate most common value - this is a good value for the PSF mu
    """
    moments_x = []
    moments_y = []
    for peak_x, peak_y in zip(centroid_xs, centroid_ys):
        image_cutout = img_data[(peak_x - 5):(peak_x + 5), (peak_y - 5):(peak_y + 5)]
        start_x = int(peak_x - 5)
        start_y = int(peak_y - 5)
        
        moment_x, moment_y = second_moment(image_cutout, peak_x, peak_y, start_x, start_y)
        moments_x.append(moment_x)
        moments_y.append(moment_y)
    moments_sq = np.sqrt(np.array(moments_x)**2 + np.array(moments_y)**2)

    _ = plt.hist(moments_sq, bins=40)
    # identify the bin with the most counts, what is the value
    best_moment = _[1][np.where(_[0] == np.max(_[0]))[0]]

    return best_moment

def make_lots_psf_image(data_image, centroid_x, centroid_y, psf_mu):
    """
    makes an image (same size as data_image) containing a single gaussian PSF
    centered at centroid_x, centroid_y.
        image_shape: (y,x) shape of the image cutout
        centroid_x: x position of the star centroid
        centroid_y: y position of the star centroid
    """
    # make a meshgrid (~coordinates) based on the image shape
    xx, yy = np.meshgrid(range(0, data_image.shape[1]), range(0, data_image.shape[0]))
    # calculate the radius from the center (centroid_x, centroid_y) at every position in the meshgrid
    xx_rel_flat = (xx - centroid_x)
    yy_rel_flat = (yy - centroid_y)
    radius = np.sqrt(xx_rel_flat **2 + yy_rel_flat**2)
    # define the PSF image using radius and the size of the star ~psf_mu
    psf_image = gaussian2D(radius=radius, mu=psf_mu)
    return psf_image



# need a function to assess the quality of the PSF subtraction
## idea: make a stamp cutout around the data image, subtract the PSF image 
##       made from `make_psf_image`, subtract/calculate chi2?