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
    xx_rel_flat = (xx - centroid_x)
    yy_rel_flat = (yy - centroid_y)
    radius = np.sqrt(xx_rel_flat **2 + yy_rel_flat**2)
    # define the PSF image using radius and the size of the star ~psf_mu
    psf_image = gaussian2D(radius=radius, mu=psf_mu)
    return psf_image



# need a function to assess the quality of the PSF subtraction
## idea: make a stamp cutout around the data image, subtract the PSF image 
##       made from `make_psf_image`, subtract/calculate chi2?