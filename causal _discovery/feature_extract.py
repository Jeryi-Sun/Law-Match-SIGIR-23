import json
import os
filePath = "/code/law_project/data/process2_data/criminal/obstruct_public/case_all.json"
features = set()
with open(filePath, 'r') as f:
    data = json.load(f)
    for kd in data.values():
        for d in kd:
            if "features" in d.keys():
                features = features | set(d["features"])
            if "focus" in d.keys():
                features = features | set(d["focus"])

print(len(features))

with open("/code/explanation_project/explanation_model/models_v2/data/features.txt", 'w') as f:
    for line in features:
        if "问题" in line:
            f.writelines(line[:-2])
        elif "争议" in line:
            f.writelines(line[:-2])
        else:
            f.writelines(line[:-2])
        f.write('\n')



