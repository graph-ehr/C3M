import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class StudentModel(nn.Module):

    def __init__(self, ehr_dim=16, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ehr_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 16))
        self.classfier = nn.Linear(16, num_classes)

    def forward(self, ehr):
        h = self.fc(ehr)
        return (self.classfier(h), h)

class TeacherTransformerModel(nn.Module):

    def __init__(self, ehr_dim: int=16, gene_dim: int=300, gene_hidden: int=16, combined_hidden: int=16, num_classes: int=2, dropout: float=0.2, num_masks: int=2, gene_dropout: float=0.1):
        super().__init__()
        self.gene_fc = FTTransformerMixAgg(num_genes=gene_dim, output_dim=gene_hidden, num_masks=num_masks, dropout=gene_dropout)
        self.combine_fc = nn.Sequential(nn.Linear(ehr_dim + gene_hidden, combined_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(combined_hidden, num_classes)

    def forward(self, ehr: torch.Tensor, gene: torch.Tensor):
        h_gene, rec_logits, mask_scores = self.gene_fc(gene)
        h = torch.cat([ehr, h_gene], dim=1)
        h = self.combine_fc(h)
        logits = self.classifier(h)
        return (logits, h, rec_logits, mask_scores)

class FTTransformerMixAgg(nn.Module):

    def __init__(self, d_model: int=32, nhead: int=2, num_layers: int=2, dim_feedforward: int=64, output_dim: int=16, num_masks: int=2, dropout: float=0.1, num_genes: int=300):
        super().__init__()
        if num_genes <= 0:
            raise ValueError('num_genes must be positive.')
        self.num_genes = num_genes
        self.gene_state_embedding = nn.Embedding(2 * num_genes, d_model)
        self.num_masks = num_masks
        self.mask_tokens = nn.Parameter(torch.zeros(num_masks, 1, d_model))
        nn.init.xavier_uniform_(self.mask_tokens)
        self.mask_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruct = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.num_genes:
            raise ValueError(f'Expected gene input [batch, {self.num_genes}], got {tuple(x.shape)}')
        B, N = x.shape
        device = x.device
        if not torch.all((x == 0) | (x == 1)):
            raise ValueError('Gene states must be binary 0/1; missing states must be handled before encoding.')
        gene_offsets = 2 * torch.arange(N, device=device)
        gene_state_ids = gene_offsets.unsqueeze(0) + x.long()
        gene_emb = self.gene_state_embedding(gene_state_ids)
        rec_views = []
        score_views = []
        state_views = []
        for k in range(self.num_masks):
            mask_tok = self.mask_tokens[k].expand(B, -1, -1)
            tokens = torch.cat([mask_tok, gene_emb], dim=1)
            _, attn_w = self.mask_attn(query=tokens[:, :1, :], key=tokens[:, 1:, :], value=tokens[:, 1:, :])
            mask_scores = attn_w.squeeze(1)
            m = mask_scores.unsqueeze(-1)
            mask_tok_exp = mask_tok.expand(-1, N, -1)
            masked_genes = (1 - m) * gene_emb + m * mask_tok_exp
            seq = torch.cat([mask_tok, masked_genes], dim=1)
            H = self.transformer(seq)
            rec_views.append(self.reconstruct(H[:, 1:, :]))
            state_views.append(H[:, 0, :])
            score_views.append(mask_scores)
        rec_stack = torch.stack(rec_views, dim=0)
        rec_logits = rec_stack.mean(dim=0)
        score_stack = torch.stack(score_views, dim=0)
        mask_scores = score_stack.mean(dim=0)
        state_stack = torch.stack(state_views, dim=0)
        mask_states = state_stack.mean(dim=0)
        out = self.head(self.norm(mask_states))
        return (out, rec_logits, mask_scores)

class EHRGNN_motor(torch.nn.Module):

    def __init__(self, init_emb, num_nodes, in_channels, hidden_channels, out_channels, dropout):
        super(EHRGNN_motor, self).__init__()
        eid_num = init_emb.shape[0]
        eid_dim = init_emb.shape[-1]
        self.init_emb = init_emb
        self.map = nn.Linear(eid_dim, in_channels)
        self.s_embedding = nn.Embedding(num_nodes - eid_num, in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.convs = [self.conv1, self.conv2]
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.s_embedding.reset_parameters()

    def forward(self, adj_t):
        eid_x = self.map(self.init_emb)
        snp_x = self.s_embedding(torch.arange(0, self.s_embedding.weight.shape[0], device=adj_t.device))
        x = torch.cat([eid_x, snp_x], axis=0)
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
