 # cpt-encoder

 - ### 将CPT转成bert形式使用

## 说明
刚刚刷到又出了一个又出了一种模型：CPT，看论文显示，
在很多中文任务上性能比mac bert还好，就迫不及待想把它用起来。

平时主要接触的NLU任务，使用官方代码做NLU的话不太方便。根据对源码的研究，发现
该模型在做nlu建模时主要用的encoder部分，也就是bert，因此我将这部分权重转为bert权重类型，方便
做nlu任务。

目前性能还未测试，第一个epoch看起来和roberta差不多。

## 加载方式

使用huggingface的transformers就可以加载，和BERT一样的方式。

## 转换代码

见 [convert_cpt_to_bert.py](convert_cpt_to_bert.py)

## 转好的权重地址

cpt-encoder-base: https://pan.baidu.com/s/1PqUAWNczX9vVcFtRHcE5cg 
提取码：2fo2

cpt-encoder-large: https://pan.baidu.com/s/1KwumkF1NRL6wX7aifnq4xA 
提取码：ke7o
## 官方地址

论文：[CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation](https://arxiv.org/pdf/2109.05729.pdf)

github：[CPT](https://github.com/fastnlp/CPT)

**Reference**  
1. [CPT](https://github.com/fastnlp/CPT)