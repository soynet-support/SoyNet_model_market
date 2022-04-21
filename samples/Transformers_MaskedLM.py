import sys
import numpy as np

sys.path.append('../')

from utils.utils import MaskedLM_Data
from include.SoyNet import *

if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    data_length = 10

    model_name = "bertMaskedLM"
    cfg_file = "../models/Transformers_MaskedLM/configs/{}.cfg".format(model_name)
    weight_file = "../models/Transformers_MaskedLM/weights/{}.weights".format(model_name)
    engine_file = "../models/Transformers_MaskedLM/engines/{}.bin".format(model_name)
    log_file = "../models/Transformers_MaskedLM/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} " \
        "INPUT_SIZE={} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize,
            data_length,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    input_raw = np.array([101, 8667, 146, 112, 182, 170, 103, 2235, 119, 102])  # Hello I'm a [MASK] model. (WordPiece)
    input_data = MaskedLM_Data(input_raw)

    # Create Output Variable
    output = np.zeros(batch_size * data_length * 2, dtype=np.uint32)

    # FeedData
    feedData(handle, input_data)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    print('test')
    
    # destroy SoyNet handle
    freeSoyNet(handle)
