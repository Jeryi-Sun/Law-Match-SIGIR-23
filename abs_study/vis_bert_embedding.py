import json
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import torch.nn as nn
import random
import torch
pdist = nn.PairwiseDistance(p=2)
import matplotlib.pyplot as plt
sns.set(style='white', font_scale=1.7)

with open("IV4sentence_bert_embedding_rationale.json", 'r') as f:
    sentence_bert_data = json.load(f)
with open("IV4sentence_bert_embedding.json", 'r') as f:
    IV4sentence_bert_data = json.load(f)
user_emb_base = []
user_emb_IV = []
label = []
dim_reducer = TSNE(n_components=2, random_state=1)
for item in sentence_bert_data:
    for i in range(len(item['user_emb_A'])):
        #if item['label'][i] == 2:
        user_emb_base.append(item['user_emb_A'][i])
        user_emb_base.append(item['user_emb_B'][i])
        label.append("rationale only")

for item in IV4sentence_bert_data:
    for i in range(len(item['user_emb_A'])):
        #if item['label'][i] == 2:
        user_emb_IV.append(item['user_emb_A'][i])
        user_emb_IV.append(item['user_emb_B'][i])
        label.append("all text")

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

label = label[:2500]+label[-2500:]
id_list = [i for i in range(len(user_emb_base))]
select_id = random.choices(id_list, k=2500)
dis = pdist(torch.tensor(user_emb_IV), torch.tensor(user_emb_base))
print(torch.mean(dis))
user_emb_base_low = dim_reducer.fit_transform(user_emb_base)[select_id]
user_emb_IV_low = dim_reducer.fit_transform(user_emb_IV)[select_id]
#df = pd.DataFrame.from_dict({'x':list_add(user_emb_base_low[:, 0].tolist(), [random.uniform(5, 10) for i in range(2500)]) + user_emb_IV_low[:, 0].tolist(), 'y':list_add(user_emb_base_low[:, 1].tolist(), [random.uniform(5, 10) for i in range(2500)]) + user_emb_IV_low[:, 1].tolist(), 'Text type':label})
#markers = {"x": "$\circ$", "y": "X"}
#sns.scatterplot(data=df, x='x', y='y', hue="Text type", style="Text type", markers=markers)
palette = {"rationale only": "b", "all text": "r"}
kws = {"facecolor": "none"}
markers = {"all text": "x", "rationale only":"o"}
# ax = sns.scatterplot(
#     data=df, x='x', y='y',
#     edgecolor=df["Text type"],
#     markers=markers,
#     style="Text type",
#     **kws,
# )
fig, ax = plt.subplots()
plt.scatter(list_add(user_emb_base_low[:, 0].tolist(), [random.uniform(0.01, 1) for i in range(2500)]), list_add(user_emb_base_low[:, 1].tolist(), [random.uniform(0.01, 1) for i in range(2500)]), marker='x', c=palette["all text"],)
plt.scatter(user_emb_IV_low[:, 0], user_emb_IV_low[:, 1],  marker='o', c="", edgecolor=palette["rationale only"])

handles, labels = zip(*[
    (plt.scatter([], [], ec=color, marker=markers[key], **kws), key) for key, color in palette.items()
])
ax.legend(handles, labels, loc='upper left')

plt.axis('off')
plt.savefig("IV4rationale_original.pdf", bbox_inches='tight')
plt.show()
# # df = pd.DataFrame.from_dict({'x':user_emb_IV_low[:, 0] ,'y':user_emb_IV_low[:, 1]})
# # sns.scatterplot(data=df,x='x',y='y')
# # plt.show()
