import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from InstructorEmbedding import INSTRUCTOR
from .GCN_backbone import GCN
from .BiCrossAttentionModule import BiCrossAttentionModule
import numpy as np 

class GRID_instructor(nn.Module):
    def __init__(
            self, config
    ):
        super().__init__()

        self.config = config

        # RG SG Encoder
        self.robot_encoder=GCN(self.config.rg_encoder_in_channels, 
                               self.config.rg_encoder_hidden_channels_1,
                               self.config.rg_encoder_out_channels)
        self.sg_encoder=GCN(self.config.sg_encoder_in_channels, 
                            self.config.sg_encoder_hidden_channels_1,
                            self.config.sg_encoder_out_channels)
        # Create projectors when lm is used to extract information
        graph_feature_num=self.config.sg_encoder_out_channels
        self.graph_projector = nn.Linear(graph_feature_num, self.config.d_model)
        
        # If language have not been preprocessed
        if not self.config.preprocessed_language:
            # Text Encoder
            self.lm = INSTRUCTOR(self.config.text_encoder_type)

            for param in self.lm.parameters():
                param.requires_grad_(False)

        # Create projectors when lm is used to extract information
        self.instruct_projector = nn.Linear(self.config.lm_word_embedding_dim, self.config.d_model)

        # BiCrossAttentionModule
        self.graph_mask_padder = nn.ConstantPad1d((self.config.num_robot_node,0),False)
        self.bi_cross = BiCrossAttentionModule(feature_dim=self.config.d_model, num_heads=self.config.bi_cross_nhead, num_layers=self.config.bi_cross_num_layers)

        # Decoder
        self.decoder = Transformer(d_model=self.config.d_model, 
                                      nhead=self.config.nhead, 
                                      num_encoder_layers=self.config.num_encoder_layers, 
                                      num_decoder_layers=self.config.num_decoder_layers, 
                                      dim_feedforward=self.config.dim_feedforward, 
                                      batch_first=self.config.batch_first)
        
        
        if self.config.batch_size > 1:
            # Head
            action_hide_dim = int(math.sqrt(self.config.num_robot_node*self.config.d_model/self.config.num_action)) * self.config.num_action
            self.action_head = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(self.config.num_robot_node * self.config.d_model, action_hide_dim),
                nn.BatchNorm1d(action_hide_dim),
                nn.LeakyReLU(),
                nn.Linear(action_hide_dim, self.config.num_action),
                # nn.Sigmoid()
            )

            object_hide_dim = int(math.sqrt(self.config.d_model))
            self.object_head = nn.Sequential(
                nn.Linear(self.config.d_model, object_hide_dim),
                nn.BatchNorm1d(object_hide_dim),
                nn.LeakyReLU(),
                nn.Linear(object_hide_dim, 1),
                # nn.Sigmoid()
            )
        else:
            # Head
            action_hide_dim = int(math.sqrt(self.config.num_robot_node*self.config.d_model/self.config.num_action)) * self.config.num_action
            self.action_head = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(self.config.num_robot_node * self.config.d_model, action_hide_dim),
                nn.LeakyReLU(),
                nn.Linear(action_hide_dim, self.config.num_action),
                # nn.Sigmoid()
            )

            object_hide_dim = int(math.sqrt(self.config.d_model))
            self.object_head = nn.Sequential(
                nn.Linear(self.config.d_model, object_hide_dim),
                nn.LeakyReLU(),
                nn.Linear(object_hide_dim, 1),
                # nn.Sigmoid()
            )

    def lm_encode_gnn(self, rg_graph_data, sg_graph_data, sg_node_mask):
        '''
        Description : 
        param        {*} self: 
        param        {*} rg_graph_data: a batch of robot graph data
        param        {*} sg_graph_data: a batch of scene graph data
        return       {*} rg_node_text_embedding, sg_node_text_embedding  
        '''
        
        # Encode text in RG SG
        if self.config.process_node_feature_method != 'tokenize':
            
            rg_graph_txt_input = np.array(rg_graph_data[0]).transpose((1,0)) # convert to list since encoder only support list type
            rg_node_text_embedding = self.lm.encode(batch_size=1, 
                                                    sentences=rg_graph_txt_input, 
                                                    output_value='sentence_embedding',
                                                    convert_to_tensor=True,
                                                    padding_size=self.config.max_node_feature_number)
                                                    #print(rg_node_text_embedding.shape)#(batchsize,3,768)

            sg_graph_txt_input = np.array(sg_graph_data[0]).transpose((1,0)) # convert to list since encoder only support list type
            sg_node_text_embedding = self.lm.encode(batch_size=1,
                                            sentences=sg_graph_txt_input, 
                                            output_value='sentence_embedding',
                                            convert_to_tensor=True,
                                            padding_size=self.config.max_node_feature_number)
        
        else:
            rg_node_text_embedding = self.lm.encode_tokens(batch_size=self.config.batch_size, 
                                                            tokens=rg_graph_data[0], 
                                                            output_value='sentence_embedding',
                                                            convert_to_tensor=True,
                                                            padding_size=self.config.max_node_feature_number,
                                                            show_progress_bar=False)
            sg_node_text_embedding = self.lm.encode_tokens(batch_size=self.config.batch_size, 
                                                        tokens=sg_graph_data[0], 
                                                        output_value='sentence_embedding',
                                                        convert_to_tensor=True,
                                                        padding_size=self.config.max_node_feature_number,
                                                        show_progress_bar=False)
        # mask the embedding by multiplying the broadcasted mask
        sg_node_text_embedding = sg_node_mask.unsqueeze(-1).expand(sg_node_text_embedding.shape) * sg_node_text_embedding

        return rg_node_text_embedding, sg_node_text_embedding  

    def encode_raw_instruct(self, instruct):
        text_padding_length = 15
        text_embedding = self.lm.encode(batch_size=self.config.batch_size,
                                        sentences=instruct, 
                                        output_value=None, # get both embeddings and token
                                        padding_size=text_padding_length)
        # Extract token embeddings and attention masks from text_embedding
        list_token_embeddings = [item['token_embeddings'] for item in text_embedding]
        list_attention_mask = [item['attention_mask'] for item in text_embedding]
        list_sentence_embeddings = [item['sentence_embedding'] for item in text_embedding]

        # Concatenate the list of token embeddings along the first dimension
        text_token_embeddings = torch.stack(list_token_embeddings).squeeze(dim=1)#torch.Size([120, 15, 1024])
        text_sentence_embedding = torch.stack(list_token_embeddings).squeeze(dim=1)
        
        # Change dimension from 1024 to d_model
        text_token_embeddings = self.instruct_projector(text_token_embeddings)#torch.Size([120, 15, 256])

        # Invert the attention masks
        text_attention_mask =  ~torch.stack(list_attention_mask).bool().squeeze(dim=1)


        return text_token_embeddings, text_attention_mask, text_sentence_embedding

    def encode_tokenized_instruct(self, instruct):
        text_embedding = self.lm.encode_tokens(batch_size=self.config.batch_size,
                                                        tokens=instruct, # inputs must be the padded tokens
                                                        output_value=None, # get both embeddings and token
                                                        output_list=False, # output dictionary
                                                        ) # do not need to pad since the tokens are padded already

        text_token_embeddings = text_embedding['token_embeddings'].squeeze(1) # torch.Size([batch_size(120), num_node(1), max_sequence_length(100), token_embeddings(1024)])
        sentence_embedding = text_embedding['sentence_embedding'].squeeze(1) # (batch_size, num_node(1)， 768)

        
        # Change dimension from 1024 to d_model
        text_token_embeddings = self.instruct_projector(text_token_embeddings)#torch.Size([120, 100, 256])

        # Invert the attention masks
        text_attention_mask = ~text_embedding['attention_mask'].bool().squeeze(1)
        
        return text_token_embeddings, text_attention_mask,  sentence_embedding

    def forward(self, batch):
        # If language have not been preprocessed
        if not self.config.preprocessed_language:   
            # Unpack data from batch
            rg_graph_data, sg_graph_data, instruct, _, _, obj_mask = batch         
            # Use Encode() to tokenize+embed Raw Texts if the was not tokenized in dataloader
            if self.config.process_node_feature_method != 'tokenize':
                text_token_embeddings, text_attention_mask, sentence_embedding = self.encode_raw_instruct(instruct)

            # The text is already tokenized in dataloader, we only need to embed the tokens
            else:
                text_token_embeddings, text_attention_mask, sentence_embedding = self.encode_tokenized_instruct(instruct)

            # LM encode Graphs Nodes
            if self.config.lm_encoded_gnn_flag:
                # graph feed into lm
                rg_node_text_embedding, sg_node_text_embedding = self.lm_encode_gnn(rg_graph_data, sg_graph_data, obj_mask)

                # robot_embedding = self.robot_encoder((rg_node_text_embedding, rg_graph_data[1]))
                # sg_embedding = self.sg_encoder((sg_node_text_embedding, sg_graph_data[1]))

                # GCN encode graph
                robot_embedding = self.robot_encoder((rg_node_text_embedding, rg_graph_data[1]), sentence_embedding)
                sg_embedding = self.sg_encoder((sg_node_text_embedding, sg_graph_data[1]), sentence_embedding)
            else:
                # GCN encode graph
                robot_embedding = self.robot_encoder(rg_graph_data)
                sg_embedding = self.sg_encoder(sg_graph_data)

        else: # language have been preprocessed
            # Unzip data from batch
            input = batch['input']

            # Feed the lm encoded node features, edge index and sentence_embeddings into GCN
            robot_embedding = self.robot_encoder((input['robot_graph']['sentence_embedding'], input['robot_graph']['edge_index']), input['instruct']['sentence_embedding'])
            sg_embedding = self.sg_encoder((input['scene_graph']['sentence_embedding'], input['scene_graph']['edge_index']),  input['instruct']['sentence_embedding'])
            obj_mask = input['scene_graph']['node_index_mask']

            # Change dimension from 1024 to d_model
            text_token_embeddings = self.instruct_projector(input['instruct']['token_embeddings'])#torch.Size([120, 100, 256])
            text_attention_mask = input['instruct']['attention_mask']


        # Concate scene graph and robot graph
        concat_sg_embedding = torch.concat([robot_embedding, sg_embedding], dim = 1)
        # Change dimension from 1024 to d_model
        concat_sg_embedding = self.graph_projector(concat_sg_embedding)

        # Use BiCrossAttentionModule to enhance feature
        graph_mask = self.graph_mask_padder(obj_mask)
        concat_sg_embedding, text_token_embeddings = self.bi_cross(x1=concat_sg_embedding, x2=text_token_embeddings, x1_mask=graph_mask, x2_mask=text_attention_mask)

        # Decode and fuse
        # (batch, sequence, embedding)
        out = self.decoder(src = text_token_embeddings, tgt = concat_sg_embedding, src_key_padding_mask=text_attention_mask)

        # Head output
        action = self.action_head(out[:, :self.config.num_robot_node, :])
        object = self.object_head(out[:, self.config.num_robot_node:, :].reshape(-1, self.config.d_model)).reshape((out.shape[0], out.shape[1] - self.config.num_robot_node))

        return action, object



