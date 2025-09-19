import numpy as np
import pandas as pd
import astropy.io.fits as pf
import starsub

def subtract_models(model_im,real_im, q=0.025,qlow=None,qhigh=None,**kwargs):
    fig,axs = plt.subplots(1,3,dpi=500)
    titles = ['Image','Model','Residual']

    if qlow is None:
        qlow = q
    else:
        print('Ignoring q, using qlow')
    if qhigh is None:
        qhigh = 1. - q    
    else:
        print('Ignoring q, using qhigh')
                     
    vmin,vmax = np.nanquantile(real_im, [qlow,qhigh])

    residual = real_im - model_im
    arrs = [real_im,model_im,residual]
    for i,ax in enumerate(axs):
        ax.imshow(arrs[i],origin='lower',cmap='plasma')
        ax.set_title(titles[i],fontsize='large')

    plt.show()

    


    