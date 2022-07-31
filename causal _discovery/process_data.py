"""
先分词，再处理一下词向量与数据
"""
import json
import gensim
import jieba
from tqdm import tqdm
import re
import numpy as np
import pickle as pkl
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from gensim.models import KeyedVectors
addr = "/home/zhongxiang_sun/code/"
section_list = ["本院查明", "审理经过", "公诉机关称"]  # 本院认为去除
num_section_selected = 10
num_select_per_doc = 10
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

def extract_vocab_from_tx():
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format('/home/zhongxiang_sun/code/explanation_project/GCI/data/Tencent_AILab_ChineseEmbedding.bin',binary=True)
    vocab = wv_from_text.key_to_index
    vocab_list = list(vocab.keys())
    with open("/home/zhongxiang_sun/code/explanation_project/GCI/data/tx_vocab.txt", 'w') as f:
        for item in tqdm(vocab_list):
            f.write(item)
            f.write('\n')

def get_clean_fact_laws():
    # process2_public = addr + "code/law_project/data/process2_data/criminal/public/case_all.json"
    # process2_economy = addr + "code/law_project/data/process2_data/criminal/economy/case_all.json"
    # process2_obstruct_public = addr + "code/law_project/data/process2_data/criminal/obstruct_public/case_all.json"
    # process1 = addr + "code/law_project/data/process_data/刑事.json"
    process2_real_right = addr + "/law_project/data/process2_data/civil/real_right/case_all.json"
    process2_contract = addr + "/law_project/data/process2_data/civil/contract/all_doc/process_data/case_all.json"
    process2_knowledge = addr + "/law_project/data/process2_data/civil/knowledge/case_all.json"
    process2_company = addr + "/law_project/data/process2_data/civil/company/case_all.json"
    process1 = addr + "/law_project/data/process_data/民事.json"
    file_list = [process1, process2_real_right, process2_knowledge, process2_company, process2_contract]
    laws_clean = []
    fact_original = []
    for file in file_list:
        with open(file, 'r') as f:
            for item_list in json.load(f).values():
                for item in item_list:
                    if item == None:
                        continue
                    if 'judgeyear' not in item.keys() or eval(item['judgeyear'])<2018 or "applicable_law" not in item.keys():
                        continue
                    para_dic = []
                    for section in item['paragraphs']:
                        if section['tag'] in section_list:
                            para_dic += splite_sentence(section['content'])[:num_section_selected]
                    fact_original.append(para_dic)
                    laws_clean.append(item['applicable_law'])

    tr4s = TextRank4Sentence()
    fact_clean = []
    for l in tqdm(range(len(fact_original))):
        tr4s.analyze("".join(fact_original[l]), lower=True, source='all_filters')
        keysentences = tr4s.get_key_sentences(num=num_select_per_doc, sentence_min_len=10)
        fact_clean.append([item.sentence for item in keysentences])

    with open("/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data/GCI/civil/fact_clean.json", 'w') as f:
        json.dump(fact_clean, f, ensure_ascii=False)

    with open("/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data/GCI/civil/laws_clean.json", 'w') as f:
        json.dump(laws_clean, f, ensure_ascii=False)



def split_fact():
    #jieba.load_userdict("/home/zhongxiang_sun/code/explanation_project/GCI/data/tx_vocab.txt")
    with open("/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data/GCI/civil/fact_clean.json", 'r') as f:
        fact_clean = json.load(f)
    fact_split = []
    key_wv = set()
    for fact in tqdm(fact_clean):
        for sentence in fact:
            """
            分词
            """
            words = jieba.lcut(sentence, cut_all=False)
            key_wv.update(words)
            fact_split.append(words)

    wv_from_text = KeyedVectors.load_word2vec_format(
        '/home/zhongxiang_sun/code/explanation_project/GCI/data/Tencent_AILab_ChineseEmbedding.bin', binary=True)
    print('Done loading word embedding')

    used_wv = {}
    oov = set()
    exact_oov = 0
    vocab = wv_from_text.key_to_index
    vocab_list = list(vocab.keys())
    for j in tqdm(key_wv):
        if not j in used_wv:
            if j in vocab_list:
                used_wv[j] = wv_from_text.get_vector(j).tolist()
            else:
                oov.add(j)
                ebd = []
                for k in j:
                    if k in vocab_list:
                        ebd.append(wv_from_text.get_vector(k).tolist())
                if len(ebd) > 0:
                    used_wv[j] = np.mean(np.array(ebd), axis=0)
                else:
                    used_wv[j] = np.random.rand(200) * 2 - 1
                    exact_oov += 1

    print(len(used_wv), len(oov), exact_oov)
    with open('/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data_utils/used_wv_civil.pkl', 'wb') as f:
        pkl.dump(used_wv, f)


def get_fact_embedding():
    #jieba.load_userdict("/home/zhongxiang_sun/code/explanation_project/GCI/data/tx_vocab.txt")
    with open("/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data/GCI/civil/fact_clean.json", 'r') as f:
        fact_clean = json.load(f)

    with open('/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data_utils/used_wv_civil.pkl', 'rb') as f:
        wv = pkl.load(f)
    fact_embedding = []
    for fact in tqdm(fact_clean):
        doc_embedding = []
        for sentence in fact:
            """
            分词
            """
            sentence_embedding = []
            words = jieba.lcut(sentence, cut_all=False)
            for w in words:
                sentence_embedding.append(wv[w])
            doc_embedding.append(np.average(sentence_embedding, axis=-2).tolist())
        fact_embedding.append(doc_embedding)
    np.save("/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data_utils/fact_embedding_civil.npy", fact_embedding)
#get_clean_fact_laws()
split_fact()
get_fact_embedding()
# np_array = np.load("/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data_utils/fact_embedding.npy", allow_pickle=True)
# print()

