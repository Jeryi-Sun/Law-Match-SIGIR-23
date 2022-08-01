"""

 章 -> 节 optional -> 条

"""
import re
import json
xingfa_file_path = "/code/explanation_project/explanation_model/models_v2/data/中华人民共和国刑法.txt"
with open(xingfa_file_path, 'r') as f:
    xingfa = f.read()

xingshisusong_file_path = "/code/explanation_project/explanation_model/models_v2/data/中华人民共和国刑事诉讼法.txt"
with open(xingshisusong_file_path, 'r') as f:
    xingshisusong = f.read()

xingshisusong_exp_file_path = "/code/explanation_project/explanation_model/models_v2/data/最高人民法院关于适用中华人民共和国刑事诉讼法的解释.txt"
with open(xingshisusong_exp_file_path, 'r') as f:
    xingshisusong_exp = f.read()

jianxing_file_path = "/code/explanation_project/explanation_model/models_v2/data/最高人民法院关于办理减刑、假释案件具体应用法律的规定.txt"
with open(jianxing_file_path, 'r') as f:
    jianxing = f.read()

hetongfa_file_path = "/code/explanation_project/explanation_model/models_v2/data/中华人民共和国合同法.txt"
with open(hetongfa_file_path, 'r') as f:
    hetongfa = f.read()

minshisusongfa_file_path = "/code/explanation_project/explanation_model/models_v2/data/中华人民共和国民事诉讼法.txt"
with open(minshisusongfa_file_path, 'r') as f:
    minshisusongfa = f.read()

def process_tiao(tiao, law_name):
    if law_name=="jianxing":
        new_tiao = tiao
    else:
        new_tiao = tiao[1:]
    i = 0
    while i < len(new_tiao):
        j = 0
        while j < len(new_tiao[i]):
            new_tiao[i][j].pop(0)
            if len(new_tiao[i][j]) == 0:
                new_tiao[i].remove(new_tiao[i][j])
                j = j - 1
            j = j + 1
        i = i + 1
    return new_tiao


def produce_law(law_name):
    """
    先分章
    """
    text = eval(law_name)
    zhang = re.split("第.*?章\u3000", text)

    """
    再分节
    """

    jie = [re.split("第.*?节\u3000", sent) for sent in zhang]

    """
    再分条
    """

    tiao = [[re.split("第.*?条\u3000", sent) for sent in list_sent] for list_sent in jie]
    new_tiao = process_tiao(tiao, law_name)
    law_dic = {}
    all_len = 0
    all_law_part = re.findall("第.*?条\u3000", text)
    i = 0
    for item1 in new_tiao:
        for item2 in item1:
            all_len += len(item2)
            for l in item2:
                law_dic[all_law_part[i][:-1]] = l
                i = i + 1
    print(all_len)


    with open("/code/explanation_project/explanation_model/models_v2/data/{}_law_dic.json".format(law_name),
              'w') as f:
        json.dump(law_dic, f, ensure_ascii=False)

    with open("/code/explanation_project/explanation_model/models_v2/data/{}_law_list.json".format(law_name),
            'w') as f:
        json.dump(new_tiao, f, ensure_ascii=False)
    return new_tiao, law_dic

#produce_law('xingfa')
#produce_law('xingshisusong')
#produce_law("xingshisusong_exp")
#produce_law("jianxing")
# produce_law("hetongfa")
# produce_law("minshisusongfa")




