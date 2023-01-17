"""
分类任务 + margin loss
"""
import sys
sys.path.append("../../")
import datetime
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
import re
from gensim.summarization import bm25
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import logging
from utils.snippets import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
import pickle
import jieba.posseg as pseg
# 基本参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size_train', type=int, default=3, help='batch size')
parser.add_argument('--batch_size_test', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='legal_bert', help='[nezha, legal_bert, lawformer]')
parser.add_argument('--checkpoint', type=str, default="./weights/predict_model", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--maxlen', type=int, default=1024, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--cuda_pos', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=1, help='max length of each case')
parser.add_argument('--G_k', type=int, default=5, help='max number of law articles')
parser.add_argument('--train', type=bool, default=True, help='whether train')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='whether train')
parser.add_argument('--log_name', type=str, default="predictor_GE", help='whether train')
parser.add_argument('--margin', type=float, default=0.01, help='margin')
parser.add_argument('--weight', type=float, default=1., help='gold_weight')
parser.add_argument('--gold_margin', type=float, default=0., help='gold_margin')
parser.add_argument('--gold_weight', type=float, default=1., help='gold_weight')
parser.add_argument('--scale_in', type=float, default=10., help='scale_in')
parser.add_argument('--scale_out', type=float, default=10., help='scale_out')
parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup_steps')
parser.add_argument('--accumulate_step', type=int, default=12, help='accumulate_step')
parser.add_argument('--data_type', type=str, default="e-CAIL", help='[ELAM, CAIL]')
parser.add_argument('--mode_type', type=str, default="graph", help='[all_sents, wo_rationale, rationale]')
parser.add_argument('--eval_metric', type=str, default="linear_out", help='[linear_out, cosine_out]')
parser.add_argument('--candidate_IV_from', type=str, default='judged', help='[judged discovery]')
parser.add_argument('--update_treatment_type', type=str, default='agg', help='[agg, Sth]')
parser.add_argument('--PLM_update_s1', action='store_true', help='if do PLM update in stage one')
parser.add_argument('--PLM_update_s2_by_IV', action='store_true', help='if do PLM update in stage two by IV')
parser.add_argument('--all_judged', action='store_true', help='use judged articles for query to test')
parser.add_argument('--random_all', action='store_true', help='use random all articles for query to test')
parser.add_argument("--random_query", action='store_true', help="use random only for query")
parser.add_argument('--random_article', action='store_true', help='use random articles for query to test')
parser.add_argument('--IV_together', action='store_true', help='if use IV together do it')
parser.add_argument('--IV_loss_weight', type=float, default=1.0, help='IV loss weight')
parser.add_argument('--together_type', type=str, default='mlp', help='[attention, mlp]')
parser.add_argument('--bm25', action='store_true', help='if use bm25')
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='../logs/{}_model_name_{}_dataset_name_{}.log'.format(args.log_name, args.model_name, args.data_type),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
if args.data_type == 'e-CAIL':
    casual_law_file_path = "../data/Casual_Law_civil.json"
else:
    casual_law_file_path = "../data/Casual_Law.json"
vocab2id_doc_path = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/vocab2id_doc.pickle"
with open(vocab2id_doc_path, 'rb') as f:
    vocab2id_doc = pickle.load(f)
if args.data_type == 'e-CAIL':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/stage3/data_prediction_rationale_wlaw.json"
elif args.data_type == 'ELAM':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_{}.json".format(args.mode_type)
else:
    exit()
def get_similarity_laws(text, tokenizer, model, Pooling, casual_law_values_tensor):
    tokenizer_output = tokenizer(text, padding=True, truncation=True, max_length=args.bert_maxlen, return_tensors='pt').to(device)
    output_1 = model(**tokenizer_output)["last_hidden_state"]
    outputs = Pooling(output_1.squeeze()).squeeze()
    laws_score = torch.max(torch.matmul(F.normalize(casual_law_values_tensor, p=2, dim=-1), F.normalize(outputs, p=2, dim=-1)), dim=-1)[0]
    similarity_index = torch.argsort(laws_score, dim=-1, descending=True)[:args.G_k]
    return [casual_law_keys[i] for i in similarity_index]

def get_similarity_laws_mse(text, tokenizer, model, Pooling, casual_law_values_tensor):
    """
    考虑到bert词向量不能直接cosine similarity，这里用了mse
    :param text:
    :param tokenizer:
    :param model:
    :param Pooling:
    :param casual_law_keys_tensor:
    :param casual_law_values_tensor:
    :return:
    """
    tokenizer_output = tokenizer(text, padding=True, truncation=True, max_length=args.bert_maxlen, return_tensors='pt').to(device)
    output_1 = model(**tokenizer_output)["last_hidden_state"]
    outputs = Pooling(output_1.squeeze()).squeeze()
    laws_score = torch.max(torch.mean(F.mse_loss(casual_law_values_tensor, outputs.expand_as(casual_law_values_tensor), reduction='none'), dim=-1), dim=-1)[0]
    similarity_index = torch.argsort(laws_score, dim=-1, descending=True)[:args.G_k]
    return [casual_law_keys[i] for i in similarity_index]

## 这里修改成一个embedding的形式 也就是返回的是embedding而不是texts


def tokenization(text, stopwords, stop_flag):
    """
    用于分词的函数
    :param filename:
    :param stopword:
    :param stop_flag:
    :return:
    """
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

def get_BM25_SCORE(main_text, pair_list, stopwords=[]):
    """
    返回main_text 与 pair_text 之间的分数
    :param main_text:
    :param match_list:
    :param stopwords 停用词
    :return:
    """
    corpus = []
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    for text in pair_list:
        corpus.append(tokenization(text, stopwords, stop_flag))
    bm25Model = bm25.BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    query = tokenization(main_text, stopwords, stop_flag)
    scores = bm25Model.get_scores(query)
    return scores

def get_similarity_laws_BM25(text, tokenizer, model, Pooling, casual_law_values_tensor):
    """
    考虑到bert词向量不能直接cosine similarity，这里用了mse
    :param text:
    :param tokenizer:
    :param model:
    :param Pooling:
    :param casual_law_keys_tensor:
    :param casual_law_values_tensor:
    :return:
    """
    laws_score = get_BM25_SCORE(text, casual_law_keys)
    similarity_index = torch.argsort(torch.tensor(laws_score), dim=-1, descending=True)[:args.G_k]
    return [casual_law_keys[i] for i in similarity_index]
if args.data_type == 'e-CAIL':
    def load_data(filename):
        """加载数据
                返回：[{...}]
                """
        all_data = []
        with open(filename) as f:
            for l in f:
                all_data.append(json.loads(l))
        if args.data_type == 'Lecard' or args.data_type == 'e-CAIL':
            random.shuffle(all_data)
        all_data = all_data
        """
        add laws features to sentence
        """
        casual_law_values_tensor = torch.tensor(casual_law_values, device=device)
        tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold, do_lower_case=True)
        model = BertModel.from_pretrained(pretrained_bert_legal_civil_fold).to(device)
        pooling = GlobalAveragePooling1D().to(device)
        for item in tqdm(all_data):
            if args.bm25:
                laws_a = get_similarity_laws_BM25(item["case_a"], tokenizer, model, pooling,
                                                  casual_law_values_tensor)
            else:
                laws_a = get_similarity_laws_mse(item["case_b"], tokenizer, model, pooling,
                                                 casual_law_values_tensor)
            if args.candidate_IV_from == "judged":
                laws_b = item["case_B_laws_text"]
                if len(laws_b) == 0:
                    laws_b = get_similarity_laws_mse(item["case_b"], tokenizer, model, pooling,
                                                     casual_law_values_tensor)
            elif args.candidate_IV_from == "discovery":
                laws_b = get_similarity_laws(item["case_b"], tokenizer, model, pooling, casual_law_values_tensor)
            else:
                print("name error")
                exit()
            if args.all_judged:
                """
                todo for test
                """
                laws_a = item["case_A_laws_text"]
                if len(laws_a) == 0:
                    laws_a = get_similarity_laws_mse(item["case_a"], tokenizer, model, pooling,
                                                     casual_law_values_tensor)
            if args.random_article:
                if args.random_all:
                    laws_a = random.sample(casual_law_keys, args.G_k)
                else:
                    laws_a = random.sample(casual_law_keys, args.G_k // 2) + laws_a[args.G_k // 2:]
                if args.random_query:
                    pass
                else:
                    if args.random_all:
                        laws_b = random.sample(casual_law_keys, args.G_k)
                    else:
                        laws_b = random.sample(casual_law_keys, args.G_k // 2) + laws_b[args.G_k // 2:]
            item["laws_a_embedding"] = "".join(laws_a)
            item["laws_b_embedding"] = "".join(laws_b)

        return all_data
else:
    def load_data(filename):
        """加载数据
        返回：[{...}]
        """
        all_data = []
        with open(filename) as f:
            for l in f:
                all_data.append(json.loads(l))
        if args.data_type == 'Lecard' or args.data_type == 'e-CAIL':
            random.shuffle(all_data)
        all_data = all_data
        """
        add laws features to sentence
        """
        casual_law_values_tensor = torch.tensor(casual_law_values, device=device)
        tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold, do_lower_case=True)
        model = BertModel.from_pretrained(pretrained_bert_legal_civil_fold).to(device)
        pooling = GlobalAveragePooling1D().to(device)
        for item in tqdm(all_data):
            if args.bm25:
                laws_a = get_similarity_laws_BM25(item["source_2_a"], tokenizer, model, pooling, casual_law_values_tensor)
            else:
                laws_a = get_similarity_laws_mse(item["source_2_a"], tokenizer, model, pooling, casual_law_values_tensor)
            if args.candidate_IV_from == "judged":
                laws_b = re.split('[LO]|[LI]', item["source_1_dis"][2])[1:]
                if len(laws_b) == 0:
                    laws_b = get_similarity_laws_mse(item["source_2_b"], tokenizer, model, pooling, casual_law_values_tensor)
            elif args.candidate_IV_from == "discovery":
                laws_b = get_similarity_laws(item["source_2_b"], tokenizer, model, pooling, casual_law_values_tensor)
            else:
                print("name error")
                exit()
            if args.all_judged:
                """
                todo for test
                """
                laws_a = item["case_A_laws_text"]
                if len(laws_a) == 0:
                    laws_a = get_similarity_laws_mse(item["source_2_a"], tokenizer, model, pooling, casual_law_values_tensor)
            if args.random_article:
                if args.random_all:
                    laws_a = random.sample(casual_law_keys, args.G_k)
                else:
                    laws_a = random.sample(casual_law_keys, args.G_k // 2) + laws_a[args.G_k // 2:]
                if args.random_query:
                    pass
                else:
                    if args.random_all:
                        laws_b = random.sample(casual_law_keys, args.G_k)
                    else:
                        laws_b = random.sample(casual_law_keys, args.G_k // 2) + laws_b[args.G_k // 2:]
            item["laws_a_embedding"] = "".join(laws_a)
            item["laws_b_embedding"] = "".join(laws_b)

        return all_data

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

def load_checkpoint(model, optimizer, trained_epoch, file_name=None):
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"], strict=False)
    #optimizer.load_state_dict(save_params["optimizer"])


def save_checkpoint(model, optimizer, trained_epoch, model_name=None):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.checkpoint):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + "{}_model_name_{}_dataset_name_{}.pkl".format(args.log_name, model_name, args.data_type)
    torch.save(save_params, filename)


class PredictorDataset(Dataset):
    """
    input data predictor convert的输出就OK
    """

    def __init__(self, data):
        super(PredictorDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        注意exp的为 match dismatch midmatch
        :param index:
        :return:
        """
        if args.data_type == 'e-CAIL':
            return self.data[index]['case_a'], self.data[index]['case_b'], self.data[index]["laws_a_embedding"], \
                   self.data[index]["laws_b_embedding"], self.data[index]['label'], \
                   self.data[index]['exp'][0], self.data[index]['exp'][1], self.data[index]['exp'][2], self.data[index][
                       'explanation']
        else:
            return self.data[index]['source_2_a'], self.data[index]['source_2_b'], self.data[index]["laws_a_embedding"], \
                   self.data[index]["laws_b_embedding"], self.data[index]['label'], \
                   self.data[index]['exp'][0], self.data[index]['exp'][1], self.data[index]['exp'][2], self.data[index][
                       'explanation']

class Collate:
    def __init__(self):
        if args.model_name == 'nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.max_seq_len = args.maxlen
        elif args.model_name == 'legal_bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_criminal_fold)
            self.max_seq_len = args.bert_maxlen
        elif args.model_name == 'legal_bert_civil':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold)
            self.max_seq_len = args.bert_maxlen
        elif args.model_name == 'lawformer':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_lawformer_fold)
            self.max_seq_len = args.maxlen

    def __call__(self, batch):
        text_a, text_b, text_z_a, text_z_b, labels, exp_match, exp_dismatch, exp_midmatch, gold_exp = [], [], [], [], [], \
                                                                                                      [], [], [], []
        for item in batch:
            text_a.append(item[0])
            text_b.append(item[1])
            text_z_a.append(item[2])
            text_z_b.append(item[3])
            labels.append(item[4])
            exp_match.append(item[5])
            exp_dismatch.append(item[6])
            exp_midmatch.append(item[7])
            gold_exp.append(item[8])
        dic_data_a = self.tokenizer.batch_encode_plus(text_a, padding=True, truncation=True,
                                                      max_length=self.max_seq_len, return_tensors='pt')
        dic_data_b = self.tokenizer.batch_encode_plus(text_b, padding=True, truncation=True,
                                                    max_length=self.max_seq_len, return_tensors='pt')
        dic_data_z_a = self.tokenizer.batch_encode_plus(text_z_a, padding=True, truncation=True,
                                                    max_length=self.max_seq_len, return_tensors='pt')
        dic_data_z_b = self.tokenizer.batch_encode_plus(text_z_b, padding=True, truncation=True,
                                                    max_length=self.max_seq_len, return_tensors='pt')
        dic_match = self.tokenizer.batch_encode_plus(exp_match, padding=True, truncation=True,
                                                     max_length=self.max_seq_len, return_tensors='pt')
        dic_dismatch = self.tokenizer.batch_encode_plus(exp_dismatch, padding=True, truncation=True,
                                                        max_length=self.max_seq_len, return_tensors='pt')
        dic_midmatch = self.tokenizer.batch_encode_plus(exp_midmatch, padding=True, truncation=True,
                                                        max_length=self.max_seq_len, return_tensors='pt')
        dic_gold_exp = self.tokenizer.batch_encode_plus(gold_exp, padding=True, truncation=True,
                                                        max_length=self.max_seq_len, return_tensors='pt')
        return dic_data_a, dic_data_b, torch.tensor(labels), dic_match, dic_midmatch, \
               dic_dismatch, dic_gold_exp, dic_data_z_a, dic_data_z_b



def build_pretrain_dataloader(data, batch_size, shuffle=True, num_workers=0, random=True, drop_last=True):
    data_generator =PredictorDataset(data)
    collate = Collate()
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=drop_last
    )

class IV_net(nn.Module):
    def __init__(self, num_node_features):
        super(IV_net, self).__init__()
        if args.IV_together:
            if args.together_type == 'mlp_old':
                self.toge_mlp_a = nn.Sequential(nn.Linear(3*num_node_features, num_node_features),
                                          nn.LeakyReLU())
                self.toge_mlp_b = nn.Sequential(nn.Linear(3 * num_node_features, num_node_features),
                                                nn.LeakyReLU())
            elif args.together_type == 'mlp':
                self.toge_mlp_a = nn.Sequential(nn.Linear(2*num_node_features, 1),
                                          nn.Sigmoid())
                self.toge_mlp_b = nn.Sequential(nn.Linear(2*num_node_features, 1),
                                          nn.Sigmoid())
        self.mlp_a = nn.Sequential(nn.Linear(num_node_features, num_node_features),
                                   nn.LeakyReLU(),
                                   nn.Linear(num_node_features, num_node_features))
        self.mlp_b = nn.Sequential(nn.Linear(num_node_features, num_node_features),
                                   nn.LeakyReLU(),
                                   nn.Linear(num_node_features, num_node_features))

        self.s1_loss_func = nn.MSELoss()
        if args.update_treatment_type == 'Cat':
            self.aggregator_a = nn.Sequential(nn.Linear(2 * num_node_features, num_node_features), nn.LeakyReLU())
            self.aggregator_b = nn.Sequential(nn.Linear(2 * num_node_features, num_node_features), nn.LeakyReLU())
        else:
            self.aggregator_a = nn.Sequential(nn.Linear(2*num_node_features, num_node_features),
                                            nn.LeakyReLU(),
                                            nn.Linear(num_node_features, num_node_features//2),
                                            nn.LeakyReLU(),
                                            nn.Linear(num_node_features//2, 1),
                                            nn.Sigmoid())
            self.aggregator_b = nn.Sequential(nn.Linear(2*num_node_features, num_node_features),
                                            nn.LeakyReLU(),
                                            nn.Linear(num_node_features, num_node_features//2),
                                            nn.LeakyReLU(),
                                            nn.Linear(num_node_features//2, 1),
                                            nn.Sigmoid())

    def forward(self, output_text_z_a, output_text_z_b, x_batch_A, x_batch_B, stage_1=False):
        if args.IV_together:
            if args.together_type == 'mlp_old':
                output_text_z_a = self.toge_mlp_a(torch.cat([output_text_z_a, output_text_z_b, x_batch_A], dim=-1))
                output_text_z_b = self.toge_mlp_b(torch.cat([output_text_z_a, output_text_z_b, x_batch_B], dim=-1))
                output_A = self.mlp_a(output_text_z_a)
                output_B = self.mlp_b(output_text_z_b)
            elif args.together_type == 'mlp':
                output_text_z = torch.stack([output_text_z_a, output_text_z_b], dim=1)
                output_text_A_a_sig = self.toge_mlp_a(torch.cat([output_text_z_a, x_batch_A], dim=-1))
                output_text_A_b_sig = self.toge_mlp_a(torch.cat([output_text_z_b, x_batch_A], dim=-1))
                output_text_B_a_sig = self.toge_mlp_b(torch.cat([output_text_z_a, x_batch_B], dim=-1))
                output_text_B_b_sig = self.toge_mlp_b(torch.cat([output_text_z_b, x_batch_B], dim=-1))
                weight_a = F.softmax(torch.cat([output_text_A_a_sig, output_text_A_b_sig], dim=-1), dim=-1).unsqueeze(-1)
                weight_b = F.softmax(torch.cat([output_text_B_a_sig, output_text_B_b_sig], dim=-1), dim=-1).unsqueeze(-1)
                IV_z_a = torch.sum(torch.mul(output_text_z, weight_a), dim=-2)
                IV_z_b = torch.sum(torch.mul(output_text_z, weight_b), dim=-2)
                output_A = self.mlp_a(IV_z_a)
                output_B = self.mlp_b(IV_z_b)
            elif args.together_type == 'attention':
                output_text_z = torch.stack([output_text_z_a, output_text_z_b], dim=1)
                output_text_z_permute = output_text_z.permute(0, 2, 1)
                weight_a = F.softmax(torch.bmm(x_batch_A.unsqueeze(1), output_text_z_permute).squeeze(),
                                     dim=-1).unsqueeze(-1)
                output_text_z_a_att = torch.sum(torch.mul(output_text_z, weight_a), dim=-2)
                weight_b = F.softmax(torch.bmm(x_batch_B.unsqueeze(1), output_text_z_permute).squeeze(),
                                     dim=-1).unsqueeze(-1)
                output_text_z_b_att = torch.sum(torch.mul(output_text_z, weight_b), dim=-2)
                output_A = self.mlp_a(output_text_z_a_att)
                output_B = self.mlp_b(output_text_z_b_att)
            else:
                print("name error")
                exit(-1)
        else:
            output_A = self.mlp_a(output_text_z_a)
            output_B = self.mlp_b(output_text_z_b)


        s1_loss_A = self.s1_loss_func(output_A, x_batch_A.detach())
        s1_loss_B = self.s1_loss_func(output_B, x_batch_B.detach())

        output_A = output_A.detach()
        output_B = output_B.detach()
        if stage_1:
            return s1_loss_A, s1_loss_B
        #cos_reg = torch.mean(torch.abs(torch.nn.CosineSimilarity()(output-x_batch.detach(), output)))
        if args.update_treatment_type == 'agg':
            user_weight_A = self.aggregator_a(torch.cat([output_A, x_batch_A], dim=-1))
            user_emb_A = user_weight_A * output_A + (1 - user_weight_A) * x_batch_A

            user_weight_B = self.aggregator_b(torch.cat([output_B, x_batch_B], dim=-1))
            user_emb_B = user_weight_B * output_B + (1 - user_weight_B) * x_batch_B
        elif args.update_treatment_type == 'Cat':
            user_emb_A = self.aggregator_a(torch.cat([output_A, x_batch_A], dim=-1))
            user_emb_B = self.aggregator_b(torch.cat([output_B, x_batch_B], dim=-1))
        elif args.update_treatment_type == 'Sth':
            user_emb_A = x_batch_A + (output_A - x_batch_A.detach())
            user_emb_B = x_batch_B + (output_B - x_batch_B.detach())
        elif args.update_treatment_type == 'no_causal':
            user_emb_A = x_batch_A - output_A
            user_emb_B = x_batch_B - output_B
        else:
            RuntimeError('name error at IV update_treatment_type')
        return user_emb_A, user_emb_B, s1_loss_A, s1_loss_B

class PredictorModel(nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        if args.model_name=='nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.model = BertModel.from_pretrained(pretrained_nezha_fold)
        elif args.model_name == 'legal_bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
            self.model = BertModel.from_pretrained(pretrained_bert_fold)
        elif args.model_name == 'lawformer':
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.model = AutoModel.from_pretrained("thunlp/Lawformer")

        self.configuration = self.model.config

        self.n = 2
        self.linear1 = nn.Sequential(
            nn.Linear(self.n*self.configuration.hidden_size, self.configuration.hidden_size),  # self.hidden_dim * 2 for bi-GRU & concat AB
            nn.LeakyReLU(),
            )


        self.linear2_match = nn.Sequential(
                                     nn.Linear(self.configuration.hidden_size+self.configuration.hidden_size, self.configuration.hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.configuration.hidden_size, 1),
                                     nn.Sigmoid()
                                     )
        self.linear2_midmatch = nn.Sequential(
                                     nn.Linear(self.configuration.hidden_size+self.configuration.hidden_size, self.configuration.hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.configuration.hidden_size, 1),
                                     nn.Sigmoid()
                                     )
        self.linear2_dismatch = nn.Sequential(
                                     nn.Linear(self.configuration.hidden_size+self.configuration.hidden_size, self.configuration.hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.configuration.hidden_size, 1),
                                     nn.Sigmoid()
                                     )
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


    def forward(self, IV_model, law_a, law_b, text_a, text_b, match, midmatch, dismatch, gold_exp, batch_label, model_type='train', IV_stage_1=False):
        output_law_a = self.model(**law_a)['pooler_output']
        output_law_b = self.model(**law_b)['pooler_output']
        output_text_a = self.model(**text_a)['pooler_output']
        output_text_b = self.model(**text_b)['pooler_output']

        if IV_stage_1:
            """
               是否对PLM进行更新
            """
            if args.PLM_update_s1:
                s1_loss_A, s1_loss_B = IV_model(output_law_a, output_law_b,
                                                output_text_a, output_text_b, stage_1=IV_stage_1)
            else:
                s1_loss_A, s1_loss_B = IV_model(output_law_a.detach(), output_law_b.detach(), output_text_a.detach(), output_text_b.detach(), stage_1=IV_stage_1)
            return s1_loss_A, s1_loss_B
        user_emb_A, user_emb_B, s1_loss_A, s1_loss_B = IV_model(output_law_a, output_law_b, output_text_a,
                                                                output_text_b)
        output_exp1 = self.model(**match)['pooler_output']
        output_exp2 = self.model(**midmatch)['pooler_output']
        output_exp3 = self.model(**dismatch)['pooler_output']
        gold_exp_pl = self.model(**gold_exp)['pooler_output']
        data_p = torch.cat([user_emb_A, user_emb_B], dim=-1)

        query = self.linear1(data_p)
        class_match = self.linear2_match(torch.cat([query, output_exp1], dim=-1))
        class_midmatch = self.linear2_midmatch(torch.cat([query, output_exp2], dim=-1))
        class_dismatch = self.linear2_dismatch(torch.cat([query, output_exp3], dim=-1))
        """
        算一个query与三个exp + golden的cos
        """
        exps = torch.stack([output_exp3, output_exp2, output_exp1], dim=1)  # (batch_size, 3, dim) 还是要把dismatch放前面
        query_1 = query.unsqueeze(1).repeat(1, 3, 1)  # (batch_size, 3, dim)
        in_cos_score = self.cos(exps, query_1)
        golden_cos_similarity = self.cos(gold_exp_pl, query)
        """
        样本间对比操作
        query 与 其他数据的exp算得分
        """
        if model_type == 'train':
            select = exps[:, batch_label.squeeze(), :]
            fi_select = select.permute([1, 0, 2])  # (batch_size, batch_size, dim)
            out_cos_score = self.cos(fi_select, query.unsqueeze(-2))
            output_scores = torch.cat((class_dismatch, class_midmatch, class_match), dim=-1)
            return {"exp_score":output_scores, "in_cos_score":in_cos_score, "golden_cos_score":golden_cos_similarity, "out_cos_score":out_cos_score}, s1_loss_A, s1_loss_B   # 需要两个mask 一个对角线mask 另一个label mask
        else:
            output_scores = torch.cat((class_dismatch, class_midmatch, class_match), dim=-1)
            return {"exp_score":output_scores, "in_cos_score":in_cos_score}, s1_loss_A, s1_loss_B   # 需要两个mask 一个对角线mask 另一个label mask

def in_class_loss(score, summary_score=None, gold_margin=0, gold_weight=1):
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss = gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


def out_class_loss(score, summary_score=None, margin=0, weight=1):
    select = torch.le(torch.eye(len(summary_score), device=device), 0)
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    select = select.contiguous().view(-1)
    ones = torch.ones_like(pos_score, device=device)
    loss_func = torch.nn.MarginRankingLoss(margin, reduction='none')
    TotalLoss = weight * torch.sum(loss_func(pos_score, neg_score, ones)*select)
    return TotalLoss


def train_valid(model, IV_model, train_dataloader, valid_dataloader, test_dataloader):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in list(model.named_parameters())], 'weight_decay_rate': args.weight_decay,
         'lr': args.lr})
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in list(IV_model.named_parameters())], 'weight_decay_rate': args.weight_decay,
         'lr': args.lr})
    optimizer = torch.optim.Adam(params=optimizer_grouped_parameters)
    early_stop = EarlyStop(args.early_stopping_patience)
    early_stop_test = EarlyStop(args.early_stopping_patience)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        epoch_loss_stage_1, epoch_loss = 0.0, 0.0
        current_step_stage_1, current_step = 0, 0
        IV_loss = 0.0
        model = model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train_stage_1',  ncols=100)
        for batch_data_stage_1 in pbar:
            text_batch_a, text_batch_b, label_batch, match_batch, midmatch_batch, dismatch_batch, gold_exp_batch, law_a, law_b = batch_data_stage_1

            law_a, law_b, text_batch_a, text_batch_b, match_batch, dismatch_batch, midmatch_batch, label_batch, gold_exp_batch = \
                law_a.to(device), law_b.to(device), text_batch_a.to(device), text_batch_b.to(device), match_batch.to(
                    device), dismatch_batch.to(device), midmatch_batch.to(device), label_batch.to(
                    device), gold_exp_batch.to(device)
            s1_loss_A, s1_loss_B = model(IV_model, law_a, law_b, text_batch_a, text_batch_b, match_batch,
                                                 midmatch_batch, dismatch_batch, gold_exp_batch,
                                                 label_batch, model_type='train', IV_stage_1=True)

            stage1_loss = s1_loss_A + s1_loss_B
            stage1_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = stage1_loss.cpu().detach().item()
            epoch_loss_stage_1 += loss_item
            current_step_stage_1 += 1
            pbar.set_description("stage_1 train loss {:.4f}".format(epoch_loss_stage_1 / current_step_stage_1))
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train')
        for batch_data in pbar:
            text_batch_a, text_batch_b, label_batch, match_batch, midmatch_batch, dismatch_batch, gold_exp_batch, law_a, law_b = batch_data

            law_a, law_b, text_batch_a, text_batch_b, match_batch, dismatch_batch, midmatch_batch, label_batch, gold_exp_batch =  \
            law_a.to(device), law_b.to(device), text_batch_a.to(device), text_batch_b.to(device), match_batch.to(device), dismatch_batch.to(device), midmatch_batch.to(device), label_batch.to(device), gold_exp_batch.to(device)
            scores, s1_loss_A, s1_loss_B = model(IV_model, law_a, law_b, text_batch_a, text_batch_b, match_batch, midmatch_batch, dismatch_batch, gold_exp_batch,
                           label_batch, model_type='train')

            """
            match midmatch dismatch
            """

            linear_similarity, gold_similarity, in_cos_score, out_cos_score = scores['exp_score'], scores['golden_cos_score'], scores['in_cos_score'], scores['out_cos_score']

            loss_in_class = args.scale_in * in_class_loss(in_cos_score, gold_similarity, args.gold_margin, args.gold_weight)
            loss_out_class = args.scale_out * out_class_loss(out_cos_score, in_cos_score[list(range(len(in_cos_score))), label_batch.squeeze()], args.margin, args.weight)

            bce_labels = torch.zeros_like(scores['exp_score'])
            bce_labels[list(range(len(bce_labels))), label_batch] = 1
            bce_labels = bce_labels.to(device)

            loss_bce = criterion(scores['exp_score'], bce_labels)
            loss = loss_in_class + loss_out_class + loss_bce  # 多任务了属于是
            optimizer.zero_grad()
            IV_loss += (s1_loss_A + s1_loss_B).detach().cpu().item()
            loss.backward()
            optimizer.step()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {} IV_loss {:.4f}".format(epoch_loss / current_step, IV_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {}".format(current_step, epoch_loss / current_step))
        epoch_loss = epoch_loss / current_step
        epoch_IV_loss = IV_loss / current_step

        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f} IV_loss {:.4f}'.format(time_str, epoch, epoch_loss, epoch_IV_loss))
        logging.info('train epoch {} loss: {:.4f} IV_loss {:.4f}'.format(epoch, epoch_loss, epoch_IV_loss))
        model = model.eval()

        current_val_metric_value = evaluation(valid_dataloader, model, IV_model, epoch, 'valid')
        is_save = early_stop.step(current_val_metric_value, epoch)
        if is_save:
            save_checkpoint(model, optimizer, epoch, "IV4IOT_bert_model")
            save_checkpoint(IV_model, optimizer, epoch, "IV4IOT_IV_net")
        else:
            pass
        if early_stop.stop_training(epoch):
            logging.info(
                "early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            print(
                "early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            logging.info(
                "Best test epoch {} Best test value {}".format(
                    early_stop_test.best_epoch, early_stop_test.best_value))
            print(
                "Best test epoch {} Best test value {}".format(
                    early_stop_test.best_epoch, early_stop_test.best_value))
            break
        current_test_metric_value = evaluation(test_dataloader, model, IV_model, epoch, 'test')
        early_stop_test.step(current_test_metric_value, epoch)

def evaluation(valid_dataloader, model, IV_model, epoch, type='valid'):
    with torch.no_grad():
        correct = 0
        total = 0
        current_step = 0
        prediction_batch_list, label_batch_list = [], []
        pbar = tqdm(valid_dataloader, desc="Iteration", postfix=type)
        for batch_data in pbar:
            text_batch_a, text_batch_b, label_batch, match_batch, midmatch_batch, dismatch_batch, gold_exp_batch, law_a, law_b  = batch_data
            law_a, law_b = law_a.to(device), law_b.to(device)
            text_batch_a = text_batch_a.to(device)
            text_batch_b = text_batch_b.to(device)
            match_batch, dismatch_batch, midmatch_batch, gold_exp_batch = match_batch.to(device), dismatch_batch.to(device), midmatch_batch.to(device), gold_exp_batch.to(device)
            label_batch = label_batch.to(device)
            label_batch_list.append(label_batch)
            """
            todo 这里好好看一下注意改造 mid 0 dis 1 match 2 
            """
            output_batch, _, _ = model(IV_model, law_a, law_b, text_batch_a, text_batch_b, match_batch, midmatch_batch, dismatch_batch, gold_exp_batch,
                                 label_batch, model_type=type)
            if args.eval_metric == 'linear_out':
                _, predicted_output = torch.max(output_batch["exp_score"], -1)
            elif args.eval_metric == 'cosine_out':
                _, predicted_output = torch.max(output_batch["in_cos_score"], -1)
            else:
                exit()
            label_batch = label_batch.to(device)
            total += len(label_batch)
            prediction_batch_list.append(predicted_output)
            correct += torch.sum(torch.eq(label_batch, predicted_output))
            pbar.set_description("{} acc {}".format(type, correct / total))
            current_step += 1
            if current_step % 100 == 0:
                logging.info('{} epoch {} acc {}/{}={:.4f}'.format(type, epoch, correct, total, correct / total))
        prediction_batch_list = torch.cat(prediction_batch_list, dim=0).cpu().tolist()
        label_batch_list = torch.cat(label_batch_list, dim=0).cpu().tolist()
        accuracy = accuracy_score(label_batch_list, prediction_batch_list)
        precision_macro = precision_score(label_batch_list, prediction_batch_list, average='macro')
        recall_macro = recall_score(label_batch_list, prediction_batch_list, average='macro')
        f1_macro = f1_score(label_batch_list, prediction_batch_list, average='macro')
        precision_micro = precision_score(label_batch_list, prediction_batch_list, average='micro')
        recall_micro = recall_score(label_batch_list, prediction_batch_list, average='micro')
        f1_micro = f1_score(label_batch_list, prediction_batch_list, average='micro')
        cohen_kappa = cohen_kappa_score(label_batch_list, prediction_batch_list)
        hamming = hamming_loss(label_batch_list, prediction_batch_list)
        jaccard_macro = jaccard_score(label_batch_list, prediction_batch_list, average='macro')
        jaccard_micro = jaccard_score(label_batch_list, prediction_batch_list, average='micro')

        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} {}  acc {} precision ma {} mi {} recall  ma {} mi {} f1  ma {} mi {}'.format(time_str, type, accuracy,
                                                                                               precision_macro,
                                                                                               precision_micro,
                                                                                               recall_macro,
                                                                                               recall_micro, f1_macro,
                                                                                               f1_micro))
        print('cohen_kappa {} hamming {} jaccard_macro {} jaccard_micro {}'.format(cohen_kappa, hamming, jaccard_macro,
                                                                                   jaccard_micro))
        logging.info(
            '{} {}  acc {} precision ma {} mi {} recall  ma {} mi {} f1  ma {} mi {}'.format(time_str, type, accuracy,
                                                                                             precision_macro,
                                                                                             precision_micro,
                                                                                             recall_macro, recall_micro,
                                                                                             f1_macro, f1_micro))
        logging.info(
            'cohen_kappa {} hamming {} jaccard_macro {} jaccard_micro {}'.format(cohen_kappa, hamming, jaccard_macro,
                                                                                 jaccard_micro))
        return accuracy

def frozen_model(P_model, unfreeze_layers):
    """
    用于冻结模型
    :param model:
    :param free_layer:
    :return:
    """
    for name, param in P_model.named_parameters():
        print(name, param.size())
    print("*" * 30)
    print('\n')

    for name, param in P_model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    # 验证一下
    for name, param in P_model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

if __name__ == '__main__':
    with open(casual_law_file_path, 'r') as f:
        casual_law = json.load(f)
    casual_law_values = []
    casual_law_keys = []

    for key in casual_law.keys():
        casual_law_values.append(casual_law[key][0])
        casual_law_keys.append(key)
    data = load_data(data_predictor_json)
    train_data = data_split(data, 'train', splite_ratio=0.8, if_random=True)
    valid_data = data_split(data, 'valid', splite_ratio=0.8, if_random=True)
    test_data = data_split(data, 'test', splite_ratio=0.8, if_random=True)
    train_data_loader = build_pretrain_dataloader(train_data, args.batch_size_train, shuffle=True, random=True, drop_last=True)
    valid_data_loader = build_pretrain_dataloader(valid_data, args.batch_size_test, shuffle=False, random=False, drop_last=False)
    test_data_loader = build_pretrain_dataloader(test_data, args.batch_size_test, shuffle=False, random=False, drop_last=False)

    P_model = PredictorModel()
    IV_model = IV_net(args.input_size).to(device)
    if args.train:
        P_model = P_model.to(device)
        train_valid(P_model, IV_model, train_data_loader, valid_data_loader, test_data_loader)
    else:
        P_model = P_model.to(device)
        load_checkpoint(P_model, None, None,
                        "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/predict_model/predictor-0.pkl")
        with torch.no_grad():
            P_model.eval()
            evaluation(valid_data_loader, P_model, IV_model, 0)

