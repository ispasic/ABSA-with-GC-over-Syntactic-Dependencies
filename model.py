import torch
import torch.nn.functional as F
import torch_scatter

class graph_conv(torch.nn.Module):
    def __init__(self, embedding_size, hidden_dim, hidden_dim2, dropout):
        super(graph_conv, self).__init__()
        
        self.l1 = torch.nn.Linear(2 * embedding_size, hidden_dim, bias=False)
        self.l2 = torch.nn.Linear(hidden_dim*2, hidden_dim, bias=True)
        self.l3 = torch.nn.Linear(hidden_dim*2, hidden_dim2, bias=True)
        self.l4 = torch.nn.Linear(hidden_dim2, 2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, edges, vertices, target_idx):
        x = vertices
        x = torch.cat([x] + [torch_scatter.scatter_add(x[edges[:,1]], edges[:,0], dim=0, dim_size=vertices.size(0))], dim=1)

        x = self.l1(x)

        identity = x
        x = F.relu(self.l2(torch.cat([x] + [torch_scatter.scatter_add(x[edges[:,1]], edges[:,0], dim=0, dim_size=vertices.size(0))], dim=1)))
        x = x / (torch.norm(x, p=2, dim=1).unsqueeze(0).t() + 0.000001) 
        x += identity # residual connection

        x = self.dropout(F.relu(self.l3(torch.cat([x] + [torch_scatter.scatter_add(x[edges[:,1]], edges[:,0], dim=0, dim_size=vertices.size(0))], dim=1))))
        x = x / (torch.norm(x, p=2, dim=1).unsqueeze(0).t() + 0.000001)

        x_target = x[target_idx]
        x = torch.squeeze(x, dim=1)
        
        x_target = self.l4(x_target)
        
        x_target = torch.unsqueeze(x_target, dim=0)

        return x_target
    