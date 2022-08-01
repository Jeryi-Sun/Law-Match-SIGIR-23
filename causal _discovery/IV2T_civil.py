import json
import re
import pickle as pkl
from tqdm import tqdm
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from gensim.models import KeyedVectors
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import KMeans
import jieba
import random
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import traceback

addr = "/"

device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')

pretrained_bert_fold = "/code/pretrain_model/bert_legal/"
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


class Selector_1(nn.Module):
    def __init__(self):
        super(Selector_1, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold, mirror='tuna', do_lower_case=True)
        self.Pooling = GlobalAveragePooling1D()
        self.encoder = BertModel.from_pretrained(pretrained_bert_fold)
        self.max_seq_len = 512


    def predict(self, text):
        """句子列表转换为句向量
        """
        with torch.no_grad():
            bert_output = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors='pt').to(device)
            output_1 = self.encoder(**bert_output)["last_hidden_state"]
            outputs = self.Pooling(output_1.squeeze())
        return outputs

def embedding_convert(sentence, model):
    outputs = model.predict(sentence)
    return np.array(outputs.cpu())

def splite_sentence(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([；])([^”’])', r"\1\n\2", para)  # 司法案例加入中文分号，冒号暂时不加

    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可
    return para.split("\n")

def load_data():
    """
    :param addr:
    :return: accu_clearn 就是法条名称，每个案例文件里
    """

    hetongfa_dic_path = addr + "code/explanation_project/explanation_model/models_v2/data/hetongfa_law_dic.json"
    with open(hetongfa_dic_path, 'r') as f:
        fa_dic = json.load(f)
        laws_selected1 = ["中华人民共和国合同法" + item for item in fa_dic.keys()]  # todo
    minshisusongfa_dic_path = addr + "code/explanation_project/explanation_model/models_v2/data/minshisusongfa_law_dic.json"
    with open(minshisusongfa_dic_path, 'r') as f:
        fa_dic = json.load(f)
        laws_selected2 = ["中华人民共和国民事诉讼法" + item for item in fa_dic.keys()]
    laws_selected = laws_selected1 + laws_selected2

    laws_selected_section_num = []
    n = 0
    hetongfa_list_path = addr + "code/explanation_project/explanation_model/models_v2/data/hetongfa_law_list.json"
    with open(hetongfa_list_path, 'r') as f:
        hetongfa_list = json.load(f)
        for item1 in hetongfa_list:
            n = 0
            for item2 in item1:
                n += len(item2)
            laws_selected_section_num.append(n)
    minshisusongfa_list_path = addr + "code/explanation_project/explanation_model/models_v2/data/minshisusongfa_law_list.json"
    with open(minshisusongfa_list_path, 'r') as f:
        minshisusongfa_list = json.load(f)
        for item1 in minshisusongfa_list:
            n = 0
            for item2 in item1:
                n += len(item2)
            laws_selected_section_num.append(n)

    """
    load fact clean and laws 
    """
    with open("/code/explanation_project/explanation_model/models_v2/data/GCI/civil/fact_clean.json", 'r') as f:
        fact_clean = json.load(f)


    with open("/code/explanation_project/explanation_model/models_v2/data/GCI/civil/laws_clean.json", 'r') as f:
        laws_clean = json.load(f)

    """
    load wv
    """
    with open('/code/explanation_project/explanation_model/models_v2/data/GCI/civil/used_wv_civil.pkl', 'rb') as f:
        used_wv = pkl.load(f)


    """
    load sentence embedding
    """
    fact_clean_embedding = np.load("/code/explanation_project/explanation_model/models_v2/data/GCI/civil/fact_embedding_civil.npy", allow_pickle=True).tolist()

    print("load data over!")
    return fact_clean, fact_clean_embedding, laws_clean, laws_selected, used_wv, laws_selected_section_num


def extract_keywords(fact_clean, laws_clean, num_select, per_law_sample, laws_selected):
    """
    text rank 算法 可以不去停用词
    :param addr:
    :param fact_original: 初始句子
    :param accu_clean:
    :param num_select:
    :return: selected sentences  type: [word, list]
    """
    tr4s = TextRank4Sentence()
    fact_laws = {}
    for l in tqdm(range(len(laws_selected))):
        fact_laws[laws_selected[l]] = []
        for i in range(len(laws_clean)):
            if laws_selected[l] in laws_clean[i]:
                fact_laws[laws_selected[l]] += fact_clean[i]
    original_combine_sentence = []
    sentence_fact_laws = {}
    for item in tqdm(fact_laws.keys()):
        if len(fact_laws[item]) > per_law_sample:
            fact_laws[item] = random.sample(fact_laws[item], per_law_sample)
        tr4s.analyze("。".join(fact_laws[item]), lower=True, source='all_filters')
        keysentences = tr4s.get_key_sentences(num=num_select, sentence_min_len=10)
        original_combine_sentence += [item.sentence for item in keysentences]
        sentence_fact_laws[item] = [item.sentence for item in keysentences]

    return original_combine_sentence, sentence_fact_laws


def cluster_sentences(original_combine_sentence, num_clusters, wv_from_text):
    print('cluster keyword...')
    if len(original_combine_sentence) < num_clusters:
        num_clusters = len(original_combine_sentence)
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40,
                        init='k-means++')
    sentence_embedding = []
    for sentence in original_combine_sentence:
        words = jieba.lcut(sentence, cut_all=False)
        key_wv = []
        for i in words:
            if i in wv_from_text.keys():
                key_wv.append(wv_from_text[i])
            if key_wv == []:
                print("error at cluster")
        sentence_embedding.append(np.mean(key_wv, axis=0))
    result = km_cluster.fit_predict(sentence_embedding)

    clustered = {}
    for i in range(len(original_combine_sentence)):
        if not result[i] in clustered:
            clustered[result[i]] = [original_combine_sentence[i]]
        else:
            clustered[result[i]].append(original_combine_sentence[i])

    combine_key = [list(set(i)) for i in clustered.values()]

    clustered_2 = {}
    for i in range(len(sentence_embedding)):
        if not result[i] in clustered_2:
            clustered_2[result[i]] = [sentence_embedding[i]]
        else:
            clustered_2[result[i]].append(sentence_embedding[i])

    combine_key_embedding = [list(i) for i in clustered_2.values()]

    return combine_key, combine_key_embedding

all_d = 0
count_n = 0

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def find_factors(fact_clean_embedding, laws_clean, laws_selected, combine_sent_embedding):
    print('find factors...')
    y = np.zeros((len(fact_clean_embedding), len(laws_selected)), dtype=np.int8)
    cnt_pos = 0
    for i in range(len(laws_clean)):
        for j in range(len(laws_selected)):
            if laws_selected[j] in laws_clean[i]:
                y[i][j] = 1
        if sum(y[i]) > 0:
            cnt_pos += 1

    factor = np.zeros((cnt_pos, len(combine_sent_embedding)), dtype=np.int8)
    factor_embedding = np.zeros((cnt_pos, len(combine_sent_embedding)), dtype=np.double)
    idx = np.zeros((cnt_pos), dtype=np.int64)
    word_idx = []
    word_key = []
    for i in range(len(laws_clean)):
        if i % 1000 == 0:
            print(i, len(word_idx))

        if np.sum(y[i]) == 0:
            # if cnt_neg == cnt_pos:
            #     continue
            # else:
            #     cnt_neg += 1  # pos neg 量一致
            continue

        idx[len(word_idx)] = i
        cur = np.zeros(len(combine_sent_embedding), dtype=np.int8)
        word_idx_cur = []
        word_key_cur = []
        cos_similarity = -1*np.ones(len(combine_sent_embedding))
        for k in range(len(combine_sent_embedding)):
            f = False
            for p in combine_sent_embedding[k]:
                if f:
                    break
                for j in range(len(fact_clean_embedding[i])):
                    d = cosine_similarity(fact_clean_embedding[i][j], p)
                    cos_similarity[k] = max(cos_similarity[k], d)
                    if d > 1.0:  # todo
                        f = True
                        word_idx_cur.append(j)
                        word_key_cur.append(k)
                        break

            if f:
                cur[k] = 1
        factor[len(word_idx)] = cur
        factor_embedding[len(word_idx)] = cos_similarity
        global count_n
        global all_d
        all_d += np.sum(cos_similarity)
        count_n += len(cos_similarity)
        if count_n % 1000 == 0:
            print(all_d/count_n)

        word_idx.append(word_idx_cur)
        word_key.append(word_key_cur)

    return y, factor, idx, word_idx, word_key, factor_embedding



def dump_data(combine_sentence, laws_selected, y, factor, factor_embedding, idx, word_idx, word_key, sentence_fact_laws, law_type, section_idx):
    print('dump data...')
    name_contain_dict = {}
    for i in range(len(combine_sentence)):
        name_contain_dict['x' + str(i)] = combine_sentence[i]

    for i in range(len(laws_selected)):
        name_contain_dict['y' + str(i)] = laws_selected[i]

    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_name'.format(law_type, section_idx) + '.json', 'w') as f:
        json.dump(name_contain_dict, f, ensure_ascii=False)


    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}'.format(law_type, section_idx) + '.tsv', 'w') as f:
        for i in range(len(combine_sentence)):
            f.write('x' + str(i) + '\t')
        for i in range(len(laws_selected)):
            if i < len(laws_selected) - 1:
                f.write('y' + str(i) + '\t')
            else:
                f.write('y' + str(i) + '\n')

        for i in range(len(word_idx)):
            for j in factor[i]:
                f.write(str(j) + '\t')
            for j in range(len(laws_selected)):
                f.write(str(y[idx[i]][j]))
                if j < len(laws_selected) - 1:
                    f.write('\t')
            f.write('\n')


    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_embedding'.format(law_type, section_idx) + '.tsv', 'w') as f:
        for i in range(len(combine_sentence)):
            f.write('x' + str(i) + '\t')
        for i in range(len(laws_selected)):
            if i < len(laws_selected) - 1:
                f.write('y' + str(i) + '\t')
            else:
                f.write('y' + str(i) + '\n')

        for i in range(len(word_idx)):
            for j in factor_embedding[i]:
                f.write(str(j) + '\t')
            for j in range(len(laws_selected)):
                f.write(str(y[idx[i]][j]))
                if j < len(laws_selected) - 1:
                    f.write('\t')
            f.write('\n')
    """
        将law对应的sentence保存下来
    """
    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_fact4laws'.format(law_type, section_idx) + '.json', 'w') as f:
        json.dump(sentence_fact_laws, f, ensure_ascii=False)

    data_new = {
        'combine_key': combine_sentence,
        'word_idx': word_idx,
        'word_key': word_key,
        'idx': idx,
        }
    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_data_{}'.format(law_type, section_idx) + '.pkl', 'wb') as f:
        pkl.dump(data_new, f)

    # with open(addr + '/code/explanation_project/explanation_model/models_v2/data/text' + '.tsv', 'w') as f:
    #     for i in range(len(word_idx)):
    #         f.write(str(idx[i]) + '\t'
    #                 + fact_clean[idx[i]] + '\t')
    #         for j in laws_clean[idx[i]]:
    #             f.write(j + ' ')
    #         f.write('\n')

def build_causal_graph(law_type, section_idx):
    from castle.algorithms import PC
    import pandas as pd
    """
    输入： factor:包含了构图数据，fact4law包含了law对应的fact  name 包含了 不同 x y所对应得信息  xingfa_law_dic 包含了不同法条名称对应的信息
    输出： 条数名：[contain(str), fact(list), nodes(list)]
    """
    factors = pd.read_csv(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_embedding'.format(law_type, section_idx) + '.tsv', delimiter='\t')
    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_fact4laws'.format(law_type, section_idx) + '.json', 'r') as f:
        fact4law = json.load(f)

    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_name'.format(law_type, section_idx) + '.json', 'r') as f:
        name = json.load(f)
    count_x, count_y =0, 0
    for item in name.keys():
        if 'x' in item:
            count_x += 1
        else:
            count_y += 1
    hetongfa_dic_path = addr + "code/explanation_project/explanation_model/models_v2/data/hetongfa_law_dic.json"
    minshisusongfa_dic_path = addr + "code/explanation_project/explanation_model/models_v2/data/minshisusongfa_law_dic.json"

    with open(hetongfa_dic_path, 'r') as f:
        hetongfa_dic = json.load(f)
    with open(minshisusongfa_dic_path, 'r') as f:
        minshisusongfa_dic = json.load(f)
    hetongfa_dic.update(minshisusongfa_dic)
    try:
        X = factors.to_numpy(dtype=np.double)
        X += np.random.normal(loc=0, scale=0.000001, size=X.shape)
        len_X = X.shape[0]
        if len_X > 5000:
            random_idx = np.random.choice(len_X, 5000, replace=False)
            X = X[random_idx]
        print("X shape", X.shape)
        pc = PC()
        pc.learn(X)
        causal_matrix = pc.causal_matrix
        focus_part = causal_matrix[count_y:, :count_x]
        final_output = {}
        for i in range(count_y):
            item = name['y'+str(i)]
            final_output[item] = [hetongfa_dic[re.findall("第.*?条", item)[0]], fact4law[item]]
            nodes_list = []
            for j in range(count_x):
                if focus_part[i][j] == 1:
                    nodes_list.append(name['x'+str(j)])
            final_output[item].append(nodes_list)
    except Exception as e:
        traceback.print_exc()
        final_output = {}
        for i in range(count_y):
            item = name['y' + str(i)]
            final_output[item] = [hetongfa_dic[re.findall("第.*?条", item)[0]], fact4law[item], []]

    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_final_data'.format(law_type, section_idx) + '.json', 'w') as f:
        json.dump(final_output, f, ensure_ascii=False)


def dump_fact_laws_clean(fact_original, laws_clean, num_select_per_doc):
    tr4s = TextRank4Sentence()
    fact_clean = []
    for l in tqdm(range(len(fact_original))):
        tr4s.analyze("".join(fact_original[l]), lower=True, source='all_filters')
        keysentences = tr4s.get_key_sentences(num=num_select_per_doc, sentence_min_len=10)
        fact_clean.append([item.sentence for item in keysentences])

    with open("/code/explanation_project/explanation_model/models_v2/data/GCI/civil/fact_clean.json", 'w') as f:
        json.dump(fact_clean, f, ensure_ascii=False)

    with open("/code/explanation_project/explanation_model/models_v2/data/GCI/civil/laws_clean.json", 'w') as f:
        json.dump(laws_clean, f, ensure_ascii=False)

def convert_data2embedding(law_type, section_idx):
    import copy
    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_final_data'.format(law_type, section_idx) + '.json', 'r') as f:
        sentence_data = json.load(f)
    model = Selector_1()
    model.to(device)
    laws2embedding = {}
    longset_embedding = 0
    for item in sentence_data.keys():
        embeddings = []
        if sentence_data[item][2] == []:
            if sentence_data[item][1] == []:
                embeddings.append(embedding_convert(sentence_data[item][0], model).tolist())
            else:
                embeddings.append(np.average([embedding_convert(sent, model) for sent in sentence_data[item][1]], axis=0).tolist())

        else:
            for li in sentence_data[item][2]:
                embeddings.append(np.average([embedding_convert(sent, model) for sent in li], axis=0).tolist())

        laws2embedding[sentence_data[item][0]] = embeddings
        longset_embedding = max(len(embeddings), longset_embedding)
    print("longest sentence num: ", longset_embedding)
    for item in laws2embedding.keys():
        while len(laws2embedding[item]) <longset_embedding:
            laws2embedding[item].append(copy.deepcopy(laws2embedding[item][-1]))
        laws2embedding[item] = [laws2embedding[item], embedding_convert(item, model).tolist()]
    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_embedding4laws'.format(law_type, section_idx) + '.json', 'w') as f:
        json.dump(laws2embedding, f, ensure_ascii=False)

def cat2onefile(law_type, laws_selected_section_num):
    import copy
    datas = []
    for section_idx, num in enumerate(laws_selected_section_num):
        with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/{}_{}_embedding4laws'.format(
                law_type, section_idx) + '.json', 'r') as f:
            datas.append(json.load(f))
    len_est = 0
    for law_dic in datas:
        for item in law_dic.keys():
            len_est = max(len(law_dic[item][0]), len_est)
    file_dic = {}
    for law_dic in datas:
        for item in law_dic.keys():
            while len(law_dic[item][0])<len_est:
                law_dic[item][0].append(copy.deepcopy(law_dic[item][0][-1]))
            file_dic[item] = law_dic[item]
    with open(addr + '/code/explanation_project/explanation_model/models_v2/data/GCI/civil/Casual_Law.json', 'w') as f:
        json.dump(file_dic, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    num_section_selected = 10  # 每个section最多选十个句子
    num_clusters_rate = 5  # 最终聚类数目  设置为 laws_selected 的倍数
    num_select = 20  # 每个法条选num个相关的句子
    graph_samples_num = 5  # sample出5个图来
    per_law_sample = 100
    begin = 0
    end = 0
    new_start = False
    fact_clean, fact_clean_embedding, laws_clean, laws_selected_raw, wv_from_text, laws_selected_section_num = load_data()
    if new_start:
        for idk, num in enumerate(laws_selected_section_num):
            end += num
            laws_selected = laws_selected_raw[begin:end]
            num_clusters = num_clusters_rate * len(laws_selected)
            original_combine_sentence, sentence_fact_laws = \
                    extract_keywords(fact_clean, laws_clean,  num_select, per_law_sample, laws_selected)


            combine_sentence, combine_sentence_embedding = cluster_sentences(original_combine_sentence, num_clusters, wv_from_text)
            y, factor, idx, word_idx, word_key, factor_embedding = find_factors(fact_clean_embedding, laws_clean, laws_selected, combine_sentence_embedding) # 这里用 fact_original_new 还是 fact_clean需要 todo

            dump_data(combine_sentence, laws_selected, y, factor, factor_embedding, idx, word_idx, word_key, sentence_fact_laws, "minfa", idk)
            begin = end
    else:
        for idk, num in enumerate(laws_selected_section_num):
            print("~~~~~", idk, "~~~~~~~~")
            build_causal_graph("minfa", idk)
            convert_data2embedding("minfa", idk)
        cat2onefile("minfa", laws_selected_section_num)



"""
user dict 从embedding中取出来放入jieba中分词
"""






