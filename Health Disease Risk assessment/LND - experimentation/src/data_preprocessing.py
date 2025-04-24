import h5py
import config
import numpy as np

def preprocess_data():
    data = h5py.File(config.DATA_PATH,'r')
    
    ct_slices = data['ct_slices']
    slice_class = data['slice_class']
    
    ct_slices = np.array(ct_slices)
    slice_class = np.array(slice_class)

    print(ct_slices.shape)
    print(slice_class.shape)
    
    print(ct_slices)
    print(slice_class)
    
    return ct_slices, slice_class