import tensorflow as tf
import numpy as np

if 1 :
    MODEL_DIR = "ssd_mobilenet_v2/saved_model"
    model = tf.saved_model.load( MODEL_DIR )

if 1 :
    WEIGHTS_FILE = "ssd_mobilenet_v2_fpnlite_640x640.weights"

    weights = model.signatures["serving_default"].variables
    with open(WEIGHTS_FILE, "wb") as f:
        ww = np.array([0] * 10, dtype=np.float32)
        ww.tofile(f)

        for idx, weight in enumerate(weights):
            ww = weight.numpy()
            if len(ww.shape)==4 :
                ww = ww.transpose((3,2,0,1))
            ww=np.array(ww, dtype=np.float32)
            ww.tofile(f)
            print(idx, ww.shape, weight.name)
    exit()
