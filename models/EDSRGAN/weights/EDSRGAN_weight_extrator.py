from model.edsr import edsr

import numpy as np
import os

def extract_weight(depth, scale, weights_file):

    # Load Model
    model = edsr(scale=scale, num_res_blocks=depth)
    model.load_weights(weights_file)

    # SoyNet weight path
    soynet_wightfile = f'../mgmt/weights/edsr-{depth}-x{scale}.weights'

    with open(soynet_wightfile, 'wb') as f:
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        weights_list = model.weights

        for idx in range(len(weights_list)):
            key = weights_list[idx].name
            w = weights_list[idx].numpy()

            if len(w.shape) == 4:
                w = w.transpose((3, 2, 0, 1))
                w = np.array(w, dtype=np.float32)

            elif len(w.shape) == 2 and "dense" in key:
                print("transpose")
                w = np.transpose(w, (1, 0))

            w.tofile(f)
            print(0, idx, key, w.shape)

        print("Done!")

if __name__ == '__main__':
    
    scale = 4     # Super-resolution factor
    depth = 16    # Number of residual blocks

    # Tensorflow weight path
    weights_dir = f'../mgmt/weights/edsr-{depth}-x{scale}'
    weights_file = os.path.join(weights_dir, 'weights.h5')

    extract_weight(depth, scale, weights_file)
