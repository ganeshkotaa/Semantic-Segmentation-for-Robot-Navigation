import numpy as np
from PIL import Image

mask = np.array(Image.open('results/predictions/0001TP_006690_mask.png'))
print('Unique classes in mask:', np.unique(mask))
print('\nClass counts:')
for c in np.unique(mask):
    print(f'  Class {c}: {(mask==c).sum()} pixels')
