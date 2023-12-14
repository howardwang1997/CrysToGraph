# pooling.py

from torch import nn

from torch_geometric.utils import scatter
from torch_scatter.composite import scatter_softmax

class GlobalAttentionPooling(nn.Module):
    def __init__(self, channels):
        """
        modified from DGL.nn.pytorch.glob.GlobalAttention
        channels: int, dim of node features
        """
        super().__init__()
#         self.attn_vec = nn.Parameter(torch.tensor(1, channels))
        self.weight_gate = nn.Linear(512, 1)
        self.transform = nn.Linear(channels, 512)
        self.softplus = nn.Softplus()
        
    def forward(self, x, batch, batch_size=None, return_attention=False):
        """
        x: torch.Tensor, [N, channels], node features of the batch of graph
        batch: torch.Tensor, [N], torch_geometric batch index
        batch_size: int, batch size
        return_attention: bool, return attention value if True
        """
        if x.dim() == 1:
            x = x.view(1, -1)
            
        batch_size = int(batch.max().item() + 1) if batch_size is None else batch_size
        batch = batch.to(x.device)
        
        nx = self.softplus(self.transform(x))
        attn = self.weight_gate(nx)
        attn = scatter_softmax(attn, index=batch, dim=-2)
        output = attn * x

        mean = scatter(x, batch, dim=-2, dim_size=batch_size, reduce='mean')
        output = scatter(output, batch, dim=-2, dim_size=batch_size, reduce='sum') + mean
        
        if return_attention:
            return output, attn
        else:
            return output
