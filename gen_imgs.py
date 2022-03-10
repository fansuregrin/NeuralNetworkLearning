import h5py
import numpy as np
from PIL import Image


with h5py.File('test_catvnoncat.h5') as hf:
    X = np.array(hf['test_set_x'])
i = 1
for x in X[:]:
    img = Image.fromarray(x)
    img.save(f'test_cats/{i}.jpg')
    i += 1