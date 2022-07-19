import numpy as np
import random
import json
from sklearn.manifold import TSNE
import scipy
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from utils.snippets import *
device = torch.device('cuda:'+"0") if torch.cuda.is_available() else torch.device('cpu')

filename = "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_for_paper/data/ELAM_rationale.json"
class GlobalAveragePooling1D(nn.Module):
    """自定义全局池化
    对一个句子的pooler取平均，一个长句子用短句的pooler平均代替
    """
    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()

    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.to(torch.float)[:, :, None]
            return torch.sum(inputs * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            return torch.mean(inputs, dim=0)
def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    all_data = []
    with open(filename) as f:
        for l in f:
            all_data.append(json.loads(l))
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold, do_lower_case=True)
    model = BertModel.from_pretrained(pretrained_bert_legal_civil_fold).to(device)
    pooling = GlobalAveragePooling1D().to(device)
    def get_embedding(text):
        tokenizer_output = tokenizer(text, padding=True, truncation=True, max_length=512,
                                     return_tensors='pt').to(device)
        output_1 = model(**tokenizer_output)["last_hidden_state"]
        outputs = pooling(output_1.squeeze()).squeeze().cpu().tolist()
        return outputs
    for item in all_data:
        item["case_A_laws_text"] = "".join(item["case_A_laws_text"])
        item["case_B_laws_text"] = "".join(item["case_B_laws_text"])
    text_embeddings = []
    law_embeddings = []

    for it in all_data:
        text_embeddings.append(get_embedding(it["case_a"]))
        text_embeddings.append(get_embedding(it["case_b"]))
        law_embeddings.append(get_embedding(it["case_A_laws_text"]))
        law_embeddings.append(get_embedding(it["case_B_laws_text"]))
    random.shuffle(law_embeddings)


    dim_reducer = TSNE(n_components=1, random_state=1)
    # text_embeddings_low = np.array(dim_reducer.fit_transform(text_embeddings)).squeeze()
    # law_embeddings_low = np.array(dim_reducer.fit_transform(law_embeddings)).squeeze()
    distance = []
    for i in range(len(text_embeddings)):
        distance.append(scipy.spatial.distance.correlation(text_embeddings[i], law_embeddings[i], w=None, centered=True))
    return 1 - sum(distance)/len(distance)
print(load_data(filename))