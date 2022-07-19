"""
冻结bert
"""
import sys
sys.path.append("../../")
import datetime
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import logging
from utils.snippets import *
from utils.SimCLS import RankingLoss
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from termcolor import colored
# 基本参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size_train', type=int, default=2, help='batch size')
parser.add_argument('--batch_size_test', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='legal_bert_criminal', help='[nezha, legal_bert_criminal/civil, lawformer]')
parser.add_argument('--checkpoint', type=str, default="./weights/baselines", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--maxlen', type=int, default=1024, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--seed', type=int, default=1, help='max length of each case')
parser.add_argument('--train', type=int, default=1, help='whether train')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='whether train')
parser.add_argument('--data_type', type=str, default='e-CAIL', help='which type of data')
parser.add_argument('--log_name', type=str, default="baseline_sentence_bert", help='whether train')
parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup_steps')
parser.add_argument('--accumulate_step', type=int, default=12, help='accumulate_step')
parser.add_argument('--save_embedding', type=int, default=1, help='save embedding')
parser.add_argument('--rationale', type=int, default=1, help='if rationale')
args = parser.parse_args()
print(args)
logging.info(args)
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
if args.data_type == 'e-CAIL':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_for_paper/data/e-CAIL.json"
elif args.data_type == 'ELAM':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_for_paper/data/ELAM_rationale.json"
elif args.data_type == 'Lecard':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_for_paper/data/Lecard.json"
else:
    RuntimeError('name error')

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
        if args.rationale:
            return self.data[index]['case_a_rationale'], self.data[index]['case_a_rationale'], self.data[index]['label']
        else:
            return self.data[index]['case_a'], self.data[index]['case_b'], self.data[index]['label']


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
        return self.data[index]['case_a'], self.data[index]['case_b'], self.data[index]['label']

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
        注意exp的为 match dismatch midmatch
        :param index:
        :return:
        """
        if self.data[index]['label'] == 3:
            self.data[index]['label'] = 2
        return self.data[index]['case_a'], self.data[index]['case_b'], self.data[index]['label']

class Collate:
    def __init__(self):
        if args.model_name=='nezha':
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
        text_a, text_b, labels = [], [], []
        for item in batch:
            text_a.append(item[0])
            text_b.append(item[1])
            labels.append(item[2])
        dic_data_a = self.tokenizer.batch_encode_plus(text_a, padding=True, truncation=True,
                                                      max_length=self.max_seq_len, return_tensors='pt')
        dic_data_b = self.tokenizer.batch_encode_plus(text_b, padding=True, truncation=True,
                                                     max_length=self.max_seq_len, return_tensors='pt')

        return dic_data_a, dic_data_b, torch.tensor(labels)



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


class PredictorModel(nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        if args.model_name=='nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.model = AutoModel.from_pretrained(pretrained_nezha_fold)
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

        self.n = 3
        if args.data_type == 'Lecard':
            self.FFN = nn.Sequential(
                nn.Linear(self.n*self.configuration.hidden_size, self.configuration.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size//2),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size//2, 3),  # todo
                )

        elif args.data_type == 'e-CAIL' or args.data_type == 'ELAM':
            self.FFN = nn.Sequential(
                nn.Linear(self.n*self.configuration.hidden_size, self.configuration.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size//2),
                nn.LeakyReLU(),
                nn.Linear(self.configuration.hidden_size//2, 3),
                )


    def forward(self, text_a, text_b):
        # output_text_a = self.model(**text_a)[0][:, 0, :]
        # output_text_b = self.model(**text_b)[0][:, 0, :]
        output_text_a = self.model(**text_a).pooler_output
        output_text_b = self.model(**text_b).pooler_output
        if args.save_embedding:
            return output_text_a, output_text_b
        data_p = torch.cat([output_text_a, output_text_b, torch.abs(output_text_a-output_text_b)], dim=-1)
        output = self.FFN(data_p)

        return output

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
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + "{}_model_name_{}_dataset_name_{}.pkl".format(args.log_name, model_name, args.data_type)
    torch.save(save_params, filename)


def train_valid(model, train_dataloader, valid_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in list(model.named_parameters())], 'weight_decay_rate': args.weight_decay, 'lr': args.lr})
    optimizer = torch.optim.Adam(params=optimizer_grouped_parameters)
    early_stop = EarlyStop(args.early_stopping_patience)
    early_stop_test = EarlyStop(args.early_stopping_patience)
    all_step_cnt = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        current_step = 0
        IV_loss = 0.0
        model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train',  ncols=100)
        for batch_data in pbar:
            all_step_cnt += 1
            text_batch_a, text_batch_b, label_batch = batch_data
            text_batch_a = text_batch_a.to(device)
            text_batch_b = text_batch_b.to(device)
            label_batch = label_batch.to(device)
            scores = model(text_batch_a, text_batch_b)
            loss = criterion(scores, label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {} IV_loss {} ".format(epoch_loss / current_step, IV_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {}  IV_loss {} ".format(current_step, epoch_loss / current_step,  IV_loss / current_step))
        epoch_loss = epoch_loss / current_step
        epoch_IV_loss = IV_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f} IV_loss {:.4f}'.format(time_str, epoch, epoch_loss, epoch_IV_loss ))
        logging.info('train epoch {} loss: {:.4f} IV_loss {:.4f}'.format(epoch, epoch_loss, epoch_IV_loss ))
        model.eval()

        current_val_metric_value = evaluation(valid_dataloader, model, epoch, 'valid')
        is_save = early_stop.step(current_val_metric_value, epoch)

        if is_save:
            save_checkpoint(model, optimizer, epoch, "sentence_bert")
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
        current_test_metric_value = evaluation(test_dataloader, model, epoch, 'test')
        early_stop_test.step(current_test_metric_value, epoch)

def evaluation(valid_dataloader, model, epoch, type='valid'):
    with torch.no_grad():
        correct = 0
        total = 0
        current_step = 0
        all_true_label = []
        all_prediction_label = []
        pbar = tqdm(valid_dataloader, desc="Iteration", postfix=type)
        if args.save_embedding:
            data_dic = []
            for batch_data in pbar:
                text_batch_a, text_batch_b, label_batch = batch_data
                text_batch_a = text_batch_a.to(device)
                text_batch_b = text_batch_b.to(device)
                label_batch = label_batch.to(device)
                user_emb_A, user_emb_B = model(text_batch_a, text_batch_b)
                data_dic.append({"user_emb_A": user_emb_A.cpu().tolist(), "user_emb_B": user_emb_B.cpu().tolist(),
                                 "label": label_batch.cpu().tolist()})
                if args.rationale:
                    with open("sentence_bert_embedding_rationale.json", "w") as f:
                        json.dump(data_dic, f, ensure_ascii=False)
                else:
                    with open("sentence_bert_embedding.json", "w") as f:
                        json.dump(data_dic, f, ensure_ascii=False)
        else:
            for batch_data in pbar:
                text_batch_a, text_batch_b, label_batch = batch_data
                text_batch_a = text_batch_a.to(device)
                text_batch_b = text_batch_b.to(device)
                label_batch = label_batch.to(device)
                scores = model(text_batch_a, text_batch_b)
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
    data = load_data(data_predictor_json)
    train_data = data_split(data, 'train', splite_ratio=0.8)
    valid_data = data_split(data, 'valid', splite_ratio=0.8)
    test_data = data_split(data, 'test', splite_ratio=0.8)
    train_data_loader = build_pretrain_dataloader(train_data, args.batch_size_train, shuffle=True, data_type=args.data_type)
    valid_data_loader = build_pretrain_dataloader(valid_data, args.batch_size_test, shuffle=False, data_type=args.data_type)
    test_data_loader = build_pretrain_dataloader(test_data, args.batch_size_test, shuffle=False, data_type=args.data_type)
    P_model = PredictorModel()
    all_data_loader = build_pretrain_dataloader(data, args.batch_size_test, shuffle=False, data_type=args.data_type)

    print(P_model)
    if args.train:
        P_model = P_model.to(device)
        train_valid(P_model, train_data_loader, valid_data_loader, test_data_loader)
    else:
        P_model = P_model.to(device)
        load_checkpoint(P_model, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_for_paper/transfer_weight/sentence_bert_base.pkl")
        with torch.no_grad():
            P_model.eval()
            evaluation(all_data_loader, P_model, 0, 'all')



