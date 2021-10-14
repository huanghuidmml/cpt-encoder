# -*- coding: UTF-8 -*-
# author    : huanghui
# date      : 2021/10/14 20:54
# file name : convert_cpt_to_bert.py
# project   : cpt-encoder
import collections
import os.path
import shutil
import argparse
from transformers import BertConfig
from modeling_cpt import CPTConfig
import torch


def convert(cpt_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    cpt_config = CPTConfig.from_pretrained(cpt_path)
    config = BertConfig(
        vocab_size=cpt_config.vocab_size,
        hidden_size=cpt_config.d_model,
        num_hidden_layers=cpt_config.encoder_layers,
        num_attention_heads=cpt_config.encoder_attention_heads,
        intermediate_size=cpt_config.encoder_ffn_dim,
        hidden_dropout_prob=cpt_config.activation_dropout,
        attention_probs_dropout_prob=cpt_config.attention_dropout,
    )
    config.save_pretrained(save_path)
    state_dict = torch.load(os.path.join(cpt_path, "pytorch_model.bin"), map_location="cpu")
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("encoder", "bert")
        new_k = new_k.replace("bert.bert", "bert.encoder")
        if 'bert' in new_k and 'decoder' not in new_k:  # 去掉decoder权重
            new_state_dict[new_k] = v
    torch.save(new_state_dict, os.path.join(save_path, "pytorch_model.bin"))
    shutil.copyfile(
        os.path.join(cpt_path, "vocab.txt"),
        os.path.join(save_path, "vocab.txt")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpt_path", type=str, required=True, help="model path of cpt model")
    parser.add_argument(
        "--save_path", type=str, required=True, help="save path"
    )

    args = parser.parse_args()

    convert(args.cpt_path, args.save_path)


if __name__ == "__main__":
    main()
