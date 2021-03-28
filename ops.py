import numpy as np


def concatenate_features(feature, keys):
    concat = np.empty((0, 1), dtype=np.float32)
    for k in feature.keys():
        x = np.reshape(feature['k'] (-1, 1))
        concat = np.concatenate([concat, x])

    return concat
