import numpy as np


def MakeMultiple32(value):
    remainder = value % 32
    print(remainder)
    if remainder != 0:
        if remainder < 16:
            return value - remainder
        else:
            return value + 32 - remainder
    else:
        return value


def MaskedLM_Data(value):
    data_length = len(value)

    # input_ids
    input_data = value

    # position_ids
    for idx in range(data_length):
        input_data = np.append(input_data, [idx])

    # token_type_ids
    for idx in range(data_length):
        input_data = np.append(input_data, [0])

    return input_data


def CreateNetsizeImage(img, netw, neth, scale):
    newh = neth
    s = newh / img.shape[0]
    neww = img.shape[1] * s
    if neww > netw:
        neww = netw
        s = neww / img.shape[1]
        newh = img.shape[0] * s
    scale = 1 / s
    return int(neww), int(newh)
