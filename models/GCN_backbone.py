import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import SAGPooling

def swish(x):
    return x * F.sigmoid(x)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels_1, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels_1) 
        self.conv2 = GATConv(hidden_channels_1, out_channels)
        
        self.fc = nn.Linear(out_channels, out_channels)
        
    def forward(self, graph_data, instr_feature = None):#batch in data
        batch_x=graph_data[0]
        batch_edge_index=graph_data[1]
        batch_size = batch_x.shape[0]
        batch_results = []

        for i in range(batch_size):
            x = batch_x[i]
            edge_index = batch_edge_index[i]
            if instr_feature != None:
                instr_tensor = instr_feature[i].repeat(x.shape[0], 1)
                x=torch.cat((x,instr_tensor), dim=1) 

            # Apply the convolution layers
            x = self.conv1(x, edge_index)
            x = swish(x)
            x = F.dropout(x, p=0.2, training=self.training) 

            x = self.conv2(x, edge_index)  
            x = swish(x)
            x = F.dropout(x, p=0.2, training=self.training)


            x = self.fc(x) 
            
            batch_results.append(x)
        return torch.stack(batch_results)