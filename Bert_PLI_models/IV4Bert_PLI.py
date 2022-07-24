"""
separately and use the no-fixed laws embedding
分stage 1 2 进行IV stage1进行 MLP Bert(optional)的更新 stage2进行主任务的更新，stage1的参数也可以更新（optional）

这一版本里第一阶段的MLP（stage 1 loss A B）只在stage 1更新
第二阶段的只更新主任务loss 所能更新的参数
"""
import sys
sys.path.append("../../")
import datetime
from sklearn.metrics import confusion_matrix
from gensim.summarization import bm25
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import gc
from torch.utils.data import Dataset, DataLoader
import logging
from utils.snippets import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from termcolor import colored
import jieba.posseg as pseg
# 基本参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size_train', type=int, default=2, help='batch size')
parser.add_argument('--batch_size_test', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='legal_bert_criminal', help='[nezha, legal_bert_criminal/civil, lawformer]')
parser.add_argument('--checkpoint', type=str, default="./weights/predict_model", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--maxlen', type=int, default=1024, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda_pos', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=1, help='max length of each case')
parser.add_argument('--train', type=int, default=1, help='whether train')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='whether train')
parser.add_argument('--early_stopping_patience_IV', type=int, default=5, help='whether train')
parser.add_argument('--data_type', type=str, default='ELAM', help='which type of data')
parser.add_argument('--log_name', type=str, default="IV4Bert_PLI", help='whether train')
parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup_steps')
parser.add_argument('--accumulate_step', type=int, default=12, help='accumulate_step')
parser.add_argument('--G_k', type=int, default=5, help='Graph top k')
parser.add_argument('--candidate_IV_from', type=str, default='judged', help='[judged discovery]')
parser.add_argument('--update_treatment_type', type=str, default='agg', help='[agg, Sth]')
parser.add_argument('--PLM_update_s1', action='store_true', help='if do PLM update in stage one')
parser.add_argument('--PLM_update_s2_by_IV', action='store_true', help='if do PLM update in stage two by IV')
parser.add_argument('--all_judged', action='store_true', help='use judged articles for query to test')
parser.add_argument('--random_article', action='store_true', help='use random articles for query to test')
parser.add_argument('--random_all', action='store_true', help='use random all articles for query to test')
parser.add_argument("--random_query", action='store_true', help="use random only for query")
parser.add_argument('--IV_together', action='store_true', help='if use IV together do it')
parser.add_argument('--weight', type=float, default=1.0, help='IV loss weight')
parser.add_argument('--together_type', type=str, default='mlp', help='[attention, mlp]')
parser.add_argument('--bm25', action='store_false', help='if use bm25')

args = parser.parse_args()
print(args)
"""
random seed set
"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='../logs/{}_model_name_{}_dataset_name_{}.log'.format(args.log_name, args.model_name, args.data_type),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志 a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
logging.info(args)
casual_law_file_path = "../data/Casual_Law.json"
if args.data_type == 'e-CAIL':
    data_predictor_json = "/data/e-CAIL_PLI.json"
elif args.data_type == 'ELAM':
    data_predictor_json = "/data/ELAM_PLI.json"
elif args.data_type == 'Lecard':
    data_predictor_json = "/data/Lecard_PLI.json"
else:
    RuntimeError('name error')

def get_similarity_laws_mse(texts, tokenizer, model, Pooling, casual_law_values_tensor):
    """
    Bert_PLI特供版 考虑到bert词向量不能直接cosine similarity，这里用了mse
    :param text:
    :param tokenizer:
    :param model:
    :param Pooling:
    :param casual_law_keys_tensor:
    :param casual_law_values_tensor:
    :return:
    """
    text_laws = []
    for text in texts:
        tokenizer_output = tokenizer(text, padding=True, truncation=True, max_length=args.bert_maxlen, return_tensors='pt').to(device)
        output_1 = model(**tokenizer_output)["last_hidden_state"]
        outputs = Pooling(output_1.squeeze()).squeeze()
        laws_score = torch.max(torch.mean(F.mse_loss(casual_law_values_tensor, outputs.expand_as(casual_law_values_tensor), reduction='none'), dim=-1), dim=-1)[0]
        similarity_index = torch.argsort(laws_score, dim=-1, descending=True)[:args.G_k]
        text_laws.append([casual_law_keys[i] for i in similarity_index])
    return text_laws

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

def get_similarity_laws_BM25(texts, tokenizer, model, Pooling, casual_law_values_tensor):
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
    text_laws = []
    for text in texts:
        laws_score = get_BM25_SCORE(text, casual_law_keys)
        similarity_index = torch.argsort(torch.tensor(laws_score), dim=-1, descending=True)[:args.G_k]
        text_laws.append([casual_law_keys[i] for i in similarity_index])
    return text_laws
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
    """
    add laws features to sentence
    """
    casual_law_values_tensor = torch.tensor(casual_law_values, device=device)
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold, do_lower_case=True)
    model = BertModel.from_pretrained(pretrained_bert_legal_civil_fold).to(device)
    pooling = GlobalAveragePooling1D().to(device)
    for item in tqdm(all_data):
        if args.bm25:
            laws_a = get_similarity_laws_BM25(item["case_a"], tokenizer, model, pooling, casual_law_values_tensor)
        else:
            laws_a = get_similarity_laws_mse(item["case_a"], tokenizer, model, pooling, casual_law_values_tensor)
        if args.candidate_IV_from == "judged":
            laws_b = item["case_B_laws_text"]
            if len(laws_b) == 0:
                laws_b = get_similarity_laws_mse(item["case_b"], tokenizer, model, pooling, casual_law_values_tensor)
        elif args.candidate_IV_from == "discovery":
            laws_b = get_similarity_laws_mse(item["case_b"], tokenizer, model, pooling, casual_law_values_tensor)
        else:
            print("name error")
            exit()
        """
        todo for test
        """
        if args.all_judged:
            """
            todo for test
            """
            laws_a = item["case_A_laws_text"]
            if len(laws_a) == 0:
                laws_a = get_similarity_laws_mse(item["case_a"], tokenizer, model, pooling, casual_law_values_tensor)
        if args.random_article:
            laws_a, laws_b = [], []
            for i in range(len(item["case_a"])):
                if args.random_all:
                    laws_a.append(random.sample(casual_law_keys, args.G_k))
                else:
                    laws_a.append(random.sample(casual_law_keys, args.G_k//2) + laws_a[i][args.G_k//2:])
            if args.random_query:
                pass
            else:
                for j in range(len(item["case_b"])):
                    if args.random_all:
                        laws_b.append(random.sample(casual_law_keys, args.G_k))
                    else:
                        laws_b.append(random.sample(casual_law_keys, args.G_k // 2) + laws_b[j][args.G_k // 2:])
        item["laws_a_embedding"] = ["".join(laws) for laws in laws_a]
        item["laws_b_embedding"] = ["".join(laws) for laws in laws_b]
    del model
    del pooling
    gc.collect()
    return all_data


class ELAM_Dataset(Dataset):
    """
    input data predictor convert的输出就OK
    """
    def __init__(self, data):
        super(ELAM_Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        注意exp的为 match dismatch midmatch
        :param index:
        :return:
        """
        return self.data[index]['case_a'], self.data[index]['case_b'],self.data[index]["laws_a_embedding"], self.data[index]["laws_b_embedding"], self.data[index]['label']


class eCAIL_Dataset(Dataset):
    """
    input data predictor convert的输出就OK
    """
    def __init__(self, data):
        super(eCAIL_Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        注意exp的为 match dismatch midmatch
        :param index:
        :return:
        """
        return self.data[index]['case_a'], self.data[index]['case_b'],self.data[index]["laws_a_embedding"], self.data[index]["laws_b_embedding"], self.data[index]['label']

class Lecard_Dataset(Dataset):
    """
    input data predictor convert的输出就OK
    """
    def __init__(self, data):
        super(Lecard_Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.data[index]['case_a'], self.data[index]['case_b'], self.data[index]["laws_a_embedding"], self.data[index]["laws_b_embedding"], self.data[index]['label']

class Collate:
    def __init__(self):
        if args.model_name == 'nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.max_seq_len = args.maxlen
        elif args.model_name == 'legal_bert_criminal':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_criminal_fold)
            self.max_seq_len = args.bert_maxlen
        elif args.model_name == 'legal_bert_civil':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold)
            self.max_seq_len = args.bert_maxlen
        elif args.model_name == 'lawformer':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_lawformer_fold)
            self.max_seq_len = args.maxlen

    def __call__(self, batch):
        main_batch, paired_batch, main_IV_batch, paired_IV_batch, label_batch = zip(*batch)
        para_num_main = len(main_batch[0])
        para_num_paired = len(paired_batch[0])

        main_batch_4bert = [[] for _ in range(para_num_main)]
        pair_batch_4bert = [[] for _ in range(para_num_paired)]
        main_IV_batch_4bert = [[] for _ in range(para_num_main)]
        pair_IV_batch_4bert = [[] for _ in range(para_num_paired)]
        for item in main_batch:
            for id, sent in enumerate(item):
                main_batch_4bert[id].append(sent)

        for item in paired_batch:
            for id, sent in enumerate(item):
                pair_batch_4bert[id].append(sent)

        for item in main_IV_batch:
            for id, sent in enumerate(item):
                main_IV_batch_4bert[id].append(sent)

        for item in paired_IV_batch:
            for id, sent in enumerate(item):
                pair_IV_batch_4bert[id].append(sent)
        main_batch_embed = []
        pair_batch_embed = []
        main_IV_batch_embed = []
        pair_IV_batch_embed = []
        for i in range(para_num_main):
            main_batch_embed.append(self.tokenizer.batch_encode_plus(main_batch_4bert[i], padding=True, truncation=True,
                                                                     max_length=self.max_seq_len, return_tensors="pt"))

        for i in range(para_num_paired):
            pair_batch_embed.append(self.tokenizer.batch_encode_plus(pair_batch_4bert[i], padding=True, truncation=True,
                                                                     max_length=self.max_seq_len, return_tensors="pt"))
        for i in range(para_num_main):
            main_IV_batch_embed.append(self.tokenizer.batch_encode_plus(main_IV_batch_4bert[i], padding=True, truncation=True,
                                                                     max_length=self.max_seq_len, return_tensors="pt"))

        for i in range(para_num_paired):
            pair_IV_batch_embed.append(self.tokenizer.batch_encode_plus(pair_IV_batch_4bert[i], padding=True, truncation=True,
                                                                     max_length=self.max_seq_len, return_tensors="pt"))
        label_batch = torch.tensor(label_batch)
        return main_batch_embed, pair_batch_embed, main_IV_batch_embed, pair_IV_batch_embed, label_batch


def build_pretrain_dataloader(data, batch_size, shuffle=True, num_workers=4, data_type='ELAM'):
    """
    :param file_path: 文件位置
    :param batch_size: bs
    :param shuffle:
    :param num_workers:
    :param data_type: [ELAM, e-CAIL, Lecard]
    :param data_usage: [train, test, valid]
    :return:
    """
    if data_type=='ELAM':
        data_generator = ELAM_Dataset(data)
    elif data_type=='e-CAIL':
        data_generator = eCAIL_Dataset(data)
    elif data_type=='Lecard':
        data_generator = Lecard_Dataset(data)
    else:
        RuntimeError('name error')

    collate = Collate()
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )

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

class PredictorModel(nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        if args.model_name=='nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.model = BertModel.from_pretrained(pretrained_nezha_fold)
        elif args.model_name == 'legal_bert_criminal':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_criminal_fold)
            self.model = BertModel.from_pretrained(pretrained_bert_legal_criminal_fold)
        elif args.model_name == 'legal_bert_civil':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_legal_civil_fold)
            self.model = BertModel.from_pretrained(pretrained_bert_legal_civil_fold)
        elif args.model_name == 'lawformer':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_lawformer_fold)
            self.model = AutoModel.from_pretrained(pretrained_lawformer_fold)

        self.configuration = self.model.config

        self.rnn = nn.GRU(2 * self.configuration.hidden_size, self.configuration.hidden_size, batch_first=True,
                          num_layers=1,
                          bidirectional=True
                          )
        self.attention_transformer = nn.Linear(2 * self.configuration.hidden_size, 2 * self.configuration.hidden_size)
        self.n = 2
        if args.data_type == 'Lecard':
            self.FFN = nn.Sequential(
                nn.Linear(self.n * self.configuration.hidden_size, self.configuration.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size // 2),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size // 2, 4),
            )

        elif args.data_type == 'e-CAIL' or args.data_type == 'ELAM':
            self.FFN = nn.Sequential(
                nn.Linear(self.n * self.configuration.hidden_size, self.configuration.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size // 2),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size // 2, 3),
            )

    def text_encode(self, text):
        sen_embed = self.model(**text)
        return sen_embed.pooler_output

    def forward(self, text_a, text_b, IV_model, text_batch_z_a, text_batch_z_b, IV_stage_1=False):
        text_embedding = []
        s1_loss_A_list, s1_loss_B_list = [], []
        if args.data_type == 'e-CAIL' or args.data_type == 'ELAM':
            for i in range(len(text_a)):
                main_para_batch = text_a[i].to(device)
                pair_para_batch = text_b[i].to(device)
                main_para_z_batch = text_batch_z_a[i].to(device)
                pair_para_z_batch = text_batch_z_b[i].to(device)
                main_case_encode = self.text_encode(main_para_batch)
                pair_case_encode = self.text_encode(pair_para_batch)
                main_case_z_encode = self.text_encode(main_para_z_batch)
                pair_case_z_encode = self.text_encode(pair_para_z_batch)
                if IV_stage_1:
                    s1_loss_A, s1_loss_B = IV_model(main_case_z_encode.detach(), pair_case_z_encode.detach(), main_case_encode.detach(), pair_case_encode.detach(), stage_1=IV_stage_1)
                    s1_loss_A_list.append(s1_loss_A)
                    s1_loss_B_list.append(s1_loss_B)
                else:
                    new_main_case_encode, new_pair_case_encode, s1_loss_A, s1_loss_B = IV_model(main_case_z_encode, pair_case_z_encode, main_case_encode, pair_case_encode, stage_1=IV_stage_1)
                    s1_loss_A_list.append(s1_loss_A)
                    s1_loss_B_list.append(s1_loss_B)
                    text_embedding.append(torch.cat([new_main_case_encode, new_pair_case_encode], dim=-1))
        elif args.data_type == 'Lecard':
            main_para_batch = text_a[0].to(device)
            main_case_encode = self.text_encode(main_para_batch)
            main_para_z_batch = text_batch_z_a[0].to(device)
            main_case_z_encode = self.text_encode(main_para_z_batch)
            for i in range(len(text_b)):
                pair_para_batch = text_b[i].to(device)
                pair_case_encode = self.text_encode(pair_para_batch)
                pair_para_z_batch = text_batch_z_b[i].to(device)
                pair_case_z_encode = self.text_encode(pair_para_z_batch)
                if IV_stage_1:
                    s1_loss_A, s1_loss_B = IV_model(main_case_z_encode.detach(), pair_case_z_encode.detach(), main_case_encode.detach(), pair_case_encode.detach(), stage_1=IV_stage_1)
                    s1_loss_A_list.append(s1_loss_A)
                    s1_loss_B_list.append(s1_loss_B)
                else:
                    new_main_case_encode, new_pair_case_encode, s1_loss_A, s1_loss_B = IV_model(main_case_z_encode, pair_case_z_encode, main_case_encode, pair_case_encode, stage_1=IV_stage_1)
                    s1_loss_A_list.append(s1_loss_A)
                    s1_loss_B_list.append(s1_loss_B)
                    text_embedding.append(torch.cat([new_main_case_encode, new_pair_case_encode], dim=-1))
        if IV_stage_1:
            return torch.sum(torch.stack(s1_loss_A_list)), torch.sum(torch.stack(s1_loss_B_list))
        text_embedding = torch.stack(text_embedding, dim=1)  # (batch_size, para_len, emb_dim)
        rnn_out, hidden = self.rnn(text_embedding)  # rnn_out: B * M * 2H, hidden: 2 * B * H
        attention_query, _ = torch.max(rnn_out, dim=1)  # attention_query: B * 2H
        attention_query = self.attention_transformer(attention_query)
        rnn_out_permute = rnn_out.permute(0, 2, 1)
        weight_alpha = F.softmax(torch.bmm(attention_query.unsqueeze(1), rnn_out_permute).squeeze(),
                                 dim=-1).unsqueeze(-1)
        attention_output = torch.sum(torch.mul(rnn_out, weight_alpha), dim=-2)
        output = self.FFN(attention_output)
        return output, torch.sum(torch.stack(s1_loss_A_list)), torch.sum(torch.stack(s1_loss_B_list))


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

        elif args.update_treatment_type == 'Sth':
            user_emb_A = x_batch_A + (output_A - x_batch_A.detach())
            user_emb_B = x_batch_B + (output_B - x_batch_B.detach())
        elif args.update_treatment_type == 'no_causal':
            user_emb_A = x_batch_A - output_A
            user_emb_B = x_batch_B - output_B
        else:
            RuntimeError('name error at IV update_treatment_type')
        return user_emb_A, user_emb_B, s1_loss_A, s1_loss_B


def load_checkpoint(model, file_name=None):
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"])


def save_checkpoint(model, optimizer, trained_epoch, model_name):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + "{}_model_name_{}_dataset_name_{}.pkl".format(args.log_name, model_name, args.data_type)
    torch.save(save_params, filename)


def train_valid(model, IV_model, train_dataloader, valid_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in list(model.named_parameters())], 'weight_decay_rate': args.weight_decay, 'lr': args.lr})
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in list(IV_model.named_parameters())], 'weight_decay_rate': args.weight_decay, 'lr': args.lr})
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
            text_batch_a, text_batch_b, text_batch_z_a, text_batch_z_b, label_batch = batch_data_stage_1
            s1_loss_A, s1_loss_B = model(text_batch_a, text_batch_b, IV_model, text_batch_z_a, text_batch_z_b, IV_stage_1=True)
            stage1_loss = s1_loss_A + s1_loss_B
            stage1_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = stage1_loss.cpu().detach().item()
            epoch_loss_stage_1 += loss_item
            current_step_stage_1 += 1
            pbar.set_description("stage_1 train loss {:.4f}".format(epoch_loss_stage_1 / current_step_stage_1))

        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train', ncols=100)
        for batch_data in pbar:
            text_batch_a, text_batch_b, text_batch_z_a, text_batch_z_b, label_batch = batch_data
            label_batch = label_batch.to(device)
            scores, s1_loss_A, s1_loss_B = model(text_batch_a, text_batch_b, IV_model, text_batch_z_a, text_batch_z_b)
            loss = criterion(scores, label_batch) + args.weight*(s1_loss_A + s1_loss_B)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            IV_loss += (s1_loss_A + s1_loss_B).detach().cpu().item()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {:.4f} IV_loss {:.4f} ".format(epoch_loss / current_step, IV_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {:.4f}  IV_loss {:.4f} ".format(current_step, epoch_loss / current_step,  IV_loss / current_step))
        epoch_loss = epoch_loss / current_step
        epoch_IV_loss = IV_loss / current_step


        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f} IV_loss {:.4f}'.format(time_str, epoch, epoch_loss, epoch_IV_loss ))
        logging.info('train epoch {} loss: {:.4f} IV_loss {:.4f}'.format(epoch, epoch_loss, epoch_IV_loss ))
        model = model.eval()

        current_val_metric_value = evaluation(valid_dataloader, model, IV_model, epoch, 'valid')
        is_save = early_stop.step(current_val_metric_value, epoch)
        if is_save:
            save_checkpoint(model, optimizer, epoch, "IV4bert_PLI_model")
            save_checkpoint(IV_model, optimizer, epoch, "IV4bert_PLI_IV_net")
        else:
            pass
        if early_stop.stop_training(epoch):
            logging.info(
                "early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            print("early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
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
        all_true_label = []
        all_prediction_label = []
        pbar = tqdm(valid_dataloader, desc="Iteration", postfix=type)
        for batch_data in pbar:
            text_batch_a, text_batch_b, text_batch_z_a, text_batch_z_b, label_batch = batch_data
            label_batch = label_batch.to(device)
            scores, _, _ = model(text_batch_a, text_batch_b, IV_model, text_batch_z_a, text_batch_z_b)
            total += len(label_batch)
            prediction_label = torch.argmax(scores, dim=-1).to(torch.long)
            correct += torch.sum(torch.eq(label_batch, prediction_label))
            all_true_label += label_batch.cpu().tolist()
            all_prediction_label += prediction_label.cpu().tolist()
            pbar.set_description("{} acc {}".format(type, correct / total))
            current_step += 1
            if current_step % 100 == 0:
                logging.info('{} epoch {} acc {}/{}={:.4f}'.format(type, epoch, correct, total, correct / total))
        print(confusion_matrix(all_true_label, all_prediction_label))
        accuracy = accuracy_score(all_true_label,  all_prediction_label)
        precision_macro = precision_score(all_true_label,  all_prediction_label, average='macro')
        recall_macro = recall_score(all_true_label,  all_prediction_label, average='macro')
        f1_macro = f1_score(all_true_label,  all_prediction_label, average='macro')
        precision_micro = precision_score(all_true_label,  all_prediction_label, average='micro')
        recall_micro = recall_score(all_true_label,  all_prediction_label, average='micro')
        f1_micro = f1_score(all_true_label,  all_prediction_label, average='micro')
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(colored('{} {} epoch {} acc {}/{}={:.4f}'.format(time_str, type, epoch, correct, total, correct / total),'red'))
        logging.info('{} epoch {} acc {}/{}={:.4f}'.format(type, epoch, correct, total, correct / total))
        print('{} {}  acc {} precision ma {} mi {} recall  ma {} mi {} f1  ma {} mi {}'.format(time_str, type, accuracy,
                                                                                               precision_macro,
                                                                                               precision_micro,
                                                                                               recall_macro,
                                                                                               recall_micro, f1_macro,
                                                                                               f1_micro))
        logging.info(
            '{} {}  acc {} precision ma {} mi {} recall  ma {} mi {} f1  ma {} mi {}'.format(time_str, type, accuracy,
                                                                                             precision_macro,
                                                                                             precision_micro,
                                                                                             recall_macro, recall_micro,
                                                                                             f1_macro, f1_micro))

        return correct / total

def frozen_model(P_model, unfreeze_layers):
    """
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
    P_model = PredictorModel().to(device)
    IV_model = IV_net(args.input_size).to(device)

    data = load_data(data_predictor_json)
    train_data = data_split(data, 'train', splite_ratio=0.8)
    valid_data = data_split(data, 'valid', splite_ratio=0.8)
    test_data = data_split(data, 'test', splite_ratio=0.8)
    train_data_loader = build_pretrain_dataloader(train_data, args.batch_size_train, shuffle=True, data_type=args.data_type)
    valid_data_loader = build_pretrain_dataloader(valid_data, args.batch_size_test, shuffle=False, data_type=args.data_type)
    test_data_loader = build_pretrain_dataloader(test_data, args.batch_size_test, shuffle=False, data_type=args.data_type)


    if args.train:
        train_valid(P_model, IV_model, train_data_loader, valid_data_loader, test_data_loader)
    else:
        if args.update_treatment_type == 'agg':
            P_model_path = "/Bert_PLI_model_agg.pkl"
            IV_model_path = "/Bert_PLI_IV_net_agg.pkl"
        elif args.update_treatment_type == 'Sth':
            P_model_path = "/Bert_PLI_model_Sth.pkl"
            IV_model_path = "/Bert_PLI_IV_net_Sth.pkl"
        load_checkpoint(P_model, P_model_path)
        load_checkpoint(IV_model, IV_model_path)
        with torch.no_grad():
            P_model.eval()
            IV_model.eval()
            evaluation(valid_data_loader, P_model, IV_model, 0, "valid")
            evaluation(valid_data_loader, P_model, IV_model, 0, "test")



