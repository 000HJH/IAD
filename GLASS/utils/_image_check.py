import PIL
import numpy as np
import torch
def save_img(X, save_dir_file,gray=False ):
    X_sqz = np.squeeze(X) #[1,3,224,224] -> [3,224,224]

    try:
        X_sqz = X_sqz.detach().cpu().numpy()
    except:    
        X_sqz = X_sqz
    
    if gray: #gray scale
        img_X = PIL.Image.fromarray((X_sqz*255).astype(np.uint8), 'L')
    else: 
        X_transpose = np.transpose(X_sqz, (1,2,0))
        img_X =  PIL.Image.fromarray((X_transpose*255).astype(np.uint8))
    img_X.save(save_dir_file)