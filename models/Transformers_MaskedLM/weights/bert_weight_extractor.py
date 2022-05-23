# pytorch test
# torch == 1.4.0

import time
import torch
from src.transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, PretrainedConfig
from transformers import pipeline
import torch.functional as F
import torch.nn as nn
import numpy  as np
import time

device = torch.device('cuda')

torch.manual_seed(1)

# temp = ''
# for _ in range(510):
#     temp += 'hello '
temp ="Hello I'm a [MASK] model."

# 1. pipeline
# unmasker = pipeline('fill-mask', model='bert-base-cased')
# outputs = unmasker(temp)
# print(outputs)
print("---------------------")

######################################################################################################


# 2. bertModel
#    1) Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

input_ids = torch.tensor(tokenizer.encode(temp, add_special_tokens=True), device=device).unsqueeze(0)
print ("input_ids :", input_ids)
result = tokenizer.decode(input_ids.tolist()[0])
print("decoder : ", result)

# batch
batch_size = 50
input_batch = input_ids
for i in range(batch_size-1):
    input_batch = torch.cat([input_batch, input_ids])

mask_index = torch.where(input_ids[0] == tokenizer.mask_token_id)

'''
# load weight
resolved_archive_file = 'C:\\Users\\JAIN/.cache\\huggingface\\transformers\\092cc582560fc3833e556b3f833695c26343cb54b7e88cd02d40821462a74999.1f48cab6c959fc6c360d22bea39d06959e90f5b002e77e836d2da45464875cda'
state_dict = torch.load(resolved_archive_file, map_location="cpu")

with open('weight.txt', mode = "w", encoding='utf-8') as f:
    for key in state_dict.keys():
        value = state_dict[key]
        f.write("%s : %s \n" %(key, value.shape))
'''

# full model
class MaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(PretrainedConfig.from_json_file(config), add_pooling_layer= False)
        self.cls = BertOnlyMLMHead(PretrainedConfig.from_json_file(config))

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        return prediction_scores

model = MaskModel('masked_bert_base_cased.json')
model.to(device)
weight = torch.load('final.pt')

model.load_state_dict(weight) # 구조 다름
model.eval()

# weigths extractor
weight_path = "E:/DEV4/mgmt/weights/bertMaksedLM_test.weights"

if 0:  # weight download, (0 -> off, 1 -> on)
    print()
    with open(weight_path, 'wb') as f:
        weights = model.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]

        f.write(np.array([0] * 10, dtype=np.float32))  # dummy 10 line
        if 0:  # 전체 보기
            for idx in range(len(weight_list)):
                key, value = weight_list[idx]
                if "num_batches_tracked" in key:
                    print(idx, "--------------------")
                    continue
                print(idx, key, value.shape)
            exit()

        if 1:
            for idx in range(0, len(weight_list)):  #
                key, w = weight_list[idx]
                if "position_ids" in key:
                    print("position_ids skip \n")
                    continue
                if 3 < idx and idx != 198:
                    if len(w.shape) == 2:
                        print("transpose() \n")
                        w = w.transpose(1, 0)
                        w = w.cpu().data.numpy()
                    else:
                        w = w.cpu().data.numpy()
                elif idx == 198:
                    continue
                else:
                    w = w.cpu().data.numpy()
                w.tofile(f)
                print(0, idx, key, w.shape)

            key, w = weight_list[198]
            w = w.cpu().data.numpy()
            w.tofile(f)
            print(0, 198, key, w.shape)
    print("-------------- weight create done ---------------")

for i in range(10):
    start = time.time()
    with torch.no_grad():
        outputs = model(input_batch)

        # logits = outputs[0, mask_index, :]
        probs = outputs.softmax(dim = 2)
        token = torch.argmax(probs, dim=2)

    end = time.time()
    print("fps : ", 1 / (end-start))

# prob
# prob_result = np.empty([1], dtype=np.float32)
# for b in range(probs.shape[0]): # batch
#     for s, e in enumerate(token[0]): # data_length
#         a = probs[b, s, e].cpu().data.numpy()
#         np.append(prob_result, a)

text = tokenizer.decode(token)
print("token : ", token.item())
print("text : ", text)

tokens = input_ids[0].cpu().numpy()
tokens[mask_index] = token
result = tokenizer.decode(tokens, skip_special_tokens=True)
print("result : ", result)

"""
#     2) BertModel
_keys_to_ignore_on_load_unexpected = [r"pooler"]
_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

bert = BertModel(PretrainedConfig.from_json_file('masked_bert_base_cased.json'), add_pooling_layer= False)
bert.to(device)

outputs = bert(input_ids)
last_hidden_states = outputs[0]

# with open('weight0_bert.txt', mode = "w", encoding='utf-8') as f:
#     bert_state_dict = bert.state_dict()
#     for key in bert_state_dict.keys():
#         value = bert_state_dict[key]
#         f.write("%s : %s \n" % (key, value.shape))

#     3) BertOnlyMLMHead
cls = BertOnlyMLMHead(PretrainedConfig.from_json_file('masked_bert_base_cased.json'))
cls.to(device)
outputs = cls(last_hidden_states)

# with open('weight0_cls.txt', mode = "w", encoding='utf-8') as f:
#     cls_state_dict = cls.state_dict()
#     for key in cls_state_dict.keys():
#         value = cls_state_dict[key]
#         f.write("%s : %s \n" % (key, value.shape))

logits = outputs[0, mask_index, :]

probs = F.softmax(logits)

input_ids[mask_index] = probs

tokens = input_ids

final_result = tokenizer.decode(tokens)
print(final_result)

# last_hidden_states = softmax[0]
#
# print(input_ids)
# for i in range(len(input_ids[0])):
#     # predictions = torch.argmax(last_hidden_states[0, mask_index]).item()
#     predictions = torch.argmax(last_hidden_states[0, i]).item()
#     print(predictions)
#     #
#     result = tokenizer.decode([predictions])
#     print(result)
"""





