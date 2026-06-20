import numpy as np

def trianglular_mask(long,short,shrink=0,dim=4):
    mask = np.zeros([long, short], dtype=bool)
    for i in range(long):
      mask[i,max(0,int(i/long*short)-shrink//2):int(i/long*short)+1] = True
    if dim ==3:
      return mask[np.newaxis,shrink:]
    elif dim ==4:
      return mask[np.newaxis,np.newaxis,shrink:]
    else:
      raise ValueError

def sparse_mask(long, short, kind):
    '''
    mask for sparse attention, 
    kind from LeftFloorMask, RightFloorMask, LeftRepetitiveMask and RightRepetitiveMask, 
    used in Your Local GAN
    '''
    stride = int(np.sqrt(short))
    assert long % short == 0
    multiple = long//short
    if kind in ['LeftFloorMask', 'RightFloorMask']:
        indices = []
        for row in range(short):
            for col in range(row - (row % stride), row + 1):
                indices.append([row, col])
        indices = np.array(indices)
        mask = np.zeros([short, short], dtype=bool)
        if kind == 'LeftFloorMask':          
            mask[indices[:, 0], indices[:, 1]] = True 
        else:   
            mask[indices[:, 1], indices[:, 0]] = True 

    if kind in ['LeftRepetitiveMask', 'RightRepetitiveMask']:
        if kind == 'RightRepetitiveMask':
            col_indices = np.arange(0,short,stride)
        else:
            col_indices = np.arange(stride - 1,short,stride)
        mask = np.eye(short, dtype=bool)
        for col in col_indices:
            mask[:,col] = True
    return np.vstack([mask]*multiple)

def get_grid_masks(long, short, nH):
    return np.repeat(np.array([sparse_mask(long, short,'RightFloorMask'),
    sparse_mask(long, short,'LeftFloorMask'),
    sparse_mask(long, short,'RightRepetitiveMask'),
    sparse_mask(long, short,'LeftRepetitiveMask')]), nH//4, axis=0)