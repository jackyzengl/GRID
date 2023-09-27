
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os
from .utils import get_tokenlizer
from .GCN_backbone import GCN

from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)

class GRID_bert(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()

        self.config = config

        # RG SG Encoder
        self.robot_encoder=GCN(self.config.rg_encoder_in_channels, 
                               self.config.rg_encoder_hidden_channels_1,
                               self.config.rg_encoder_hidden_channels_2,
                               self.config.rg_encoder_hidden_channels_3, 
                               self.config.rg_encoder_out_channels)
        self.sg_encoder=GCN(self.config.sg_encoder_in_channels, 
                            self.config.sg_encoder_hidden_channels_1,
                            self.config.sg_encoder_hidden_channels_2,
                            self.config.sg_encoder_hidden_channels_3, 
                            self.config.sg_encoder_out_channels)
        
        # Bert Text Encoder
        self.tokenizer = get_tokenlizer.get_tokenlizer(self.config.text_encoder_type)
        self.lm = get_tokenlizer.get_pretrained_language_model(self.config.text_encoder_type)
        for param in self.lm.parameters():
            param.requires_grad_(False)
        # self.lm.pooler.dense.weight.requires_grad_(False)
        # self.lm.pooler.dense.bias.requires_grad_(False)
        self.lm = BertModelWarper(bert_model=self.lm)

        self.feat_map = nn.Linear(self.lm.config.hidden_size, self.config.d_model, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # Decoder
        self.decoder = nn.Transformer(d_model=self.config.d_model, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)
        
        # Head
        # 把特征维度压缩为1个score
        action_hide_dim = int(math.sqrt(self.config.num_robot_node*self.config.d_model/self.config.num_action)) * self.config.num_action
        self.action_head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.config.num_robot_node*self.config.d_model, action_hide_dim),
            nn.BatchNorm1d(action_hide_dim), # 参考NLP的通常做法，使用layerNorm
            nn.LeakyReLU(),
            nn.Linear(action_hide_dim, self.config.num_action),
            nn.Sigmoid()
        )

        object_hide_dim = int(math.sqrt(self.config.d_model))
        self.object_head = nn.Sequential(
            nn.Linear(self.config.d_model, object_hide_dim),
            nn.BatchNorm1d(object_hide_dim), # 参考NLP的通常做法，使用layerNorm
            nn.LeakyReLU(),
            nn.Linear(object_hide_dim, 1),
            nn.Sigmoid()
        )

    def tokenize(self, instruct):
        tokenized = self.tokenizer(instruct, padding="longest", return_tensors="pt").to(robot_embedding.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        # 截断超过长度限制的部分
        if text_self_attention_masks.shape[1] > self.config.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.config.max_text_len, : self.config.max_text_len
            ]
            position_ids = position_ids[:, : self.config.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.config.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.config.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.config.max_text_len]

        return tokenized
    
    def forward(self,rg_graph_data,sg_graph_data,instruct):
        # Encode SG
        robot_embedding = self.robot_encoder(rg_graph_data)
        sg_embedding = self.sg_encoder(sg_graph_data)

        # Tokenize text
        tokenized = self.tokenize(instruct)

        # Extract text embeddings
        if self.config.sub_sentence_present: # TODO 未知作用
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.lm(**tokenized_for_encoder)  # bs, 195, d_model

        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask
        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195

        if encoded_text.shape[1] > self.config.max_text_len:
            encoded_text = encoded_text[:, : self.config.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.config.max_text_len] # 注意这里取反了，True代表是padding的对象，需要mask掉，符合后面Transformer的输入
            position_ids = position_ids[:, : self.config.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.config.max_text_len, : self.config.max_text_len
            ]

        instruct_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": ~text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }


        # 理解为翻译任务，把文本指令翻译成节点。所以instruct应当是Transformer的src，各节点是tgt（query）
        # (batch, sequence, embedding)
        concat_sg_embedding = torch.concat([robot_embedding, sg_embedding], dim = 1)
        out = self.decoder(src = instruct_dict["encoded_text"], 
                           tgt = concat_sg_embedding , 
                           src_key_padding_mask = instruct_dict["text_token_mask"]) #

        # output
        action = self.action_head(out[:, :self.config.num_robot_node, :])
        object = self.object_head(out[:, self.config.num_robot_node:, :].reshape(-1, self.config.d_model)).reshape((out.shape[0], out.shape[1] - self.config.num_robot_node))

        return action, object



