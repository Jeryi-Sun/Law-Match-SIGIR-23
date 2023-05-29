# Implementation of Law-Match
This is the official implementation of the paper "Law Article-Enhanced Legal Case Matching: a Causal Learning Approach" based on PyTorch.

## Overview

## Law-Match(Sentence-Bert)
## Law-Match(Bert-PLI)
## Law-Match(Lawformer)

## Law-Match(IOT-Match)

Parameters are set as default in the code.

## Reproduction
Check the following instructions for reproducing experiments.

### Dataset
The Dataset details is shown in dataset file

### Quick Start
#### 1. Download data


#### 2. Train and evaluate our model:

```bash
python models/IV4sentence_bert_v5.py  
python Bert_PLI_models/IV4Bert_PLI.py 
python IV4lawformer_models/IV4lawformer_v5.py
python IOT-Match/IV4IOT-Match.py
```

#### 3. Check training and evaluation process
### Requirements
```
torch>=1.9.1+cu111
transformers>=4.18.0
numpy>=1.20.1
jieba>=0.42.1
six>=1.15.0
rouge>=1.0.1
tqdm>=4.59.0
scikit-learn>=0.24.1
pandas>=1.2.4
matplotlib>=3.3.4
termcolor>=1.1.0
networkx>=2.5
requests>=2.25.1
filelock>=3.0.12
gensim>=3.8.3
scipy>=1.6.2
seaborn>=0.11.1
boto3>=1.26.18
botocore>=1.29.18
pip>=22.0.3
packaging>=20.9
Pillow>=8.2.0
ipython>=7.22.0
regex>=2021.4.4
tokenizers>=0.12.1
PyYAML>=5.4.1
sacremoses>=0.0.53
psutil>=5.8.0
h5py>=2.10.0
msgpack>=1.0.2
```

### Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.1
* torch version: 1.9.0
* OS: CentOS Linux release 7.4.1708 (Core)
* GPU: The NVIDIA 3090 GPU
* CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
