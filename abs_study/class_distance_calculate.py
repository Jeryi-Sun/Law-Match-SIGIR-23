import json
import torch.nn as nn
import torch
pdist = nn.PairwiseDistance(p=2)
with open("IV4lawformer_embedding.json", 'r') as f:
    IV4sentence_bert_data = json.load(f)
class_0_emb_a = []
class_0_emb_b = []
class_1_emb_a = []
class_1_emb_b = []
class_2_emb_a = []
class_2_emb_b = []
for item in IV4sentence_bert_data:
    for i in range(len(item['user_emb_A'])):
        user_emb_A = item['user_emb_A'][i]
        user_emb_B = item['user_emb_B'][i]
        label = item['label'][i]
        eval("class_{}_emb_a".format(label)).append(user_emb_A)
        eval("class_{}_emb_b".format(label)).append(user_emb_B)

dis_0 = pdist(torch.tensor(class_0_emb_a), torch.tensor(class_0_emb_b))
dis_1 = pdist(torch.tensor(class_1_emb_a), torch.tensor(class_1_emb_b))
dis_2 = pdist(torch.tensor(class_2_emb_a), torch.tensor(class_2_emb_b))
print("label 0 dis {} label 1 dis {} label 2 dis {}".format(torch.mean(dis_0), torch.mean(dis_1), torch.mean(dis_2)))





