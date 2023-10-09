import torch
import torch.nn as nn


class Selector(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = AttentionModel(*args, **kwargs)


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class AttentionModel(nn.Module):
    def __init__(self, feat_size, nhead, layer_num, residual=0, concat=0, **kwargs):
        super(AttentionModel, self).__init__()
        self.concat = concat
        if self.concat==0:
            d_model = feat_size['ins_embed_size'] + feat_size['image_feat_size'] + feat_size['text_feat_size']
        else:
            d_model = feat_size['ins_embed_size']
        self.multihead_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead) for _ in range(layer_num)
        ])
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(layer_num)
        ])
        self.linear = nn.Linear(d_model, 1) 
        self.num_layers = layer_num
        self.residual = residual

    def forward(self, instruction_logits, image_feats, text_feats):
        if self.concat==0:
            x = torch.cat([image_feats, text_feats, instruction_logits], -1)
        else:
            if len(image_feats.shape) == 2:
                image_feats = image_feats.unsqueeze(1)
                text_feats = text_feats.unsqueeze(1)
                instruction_logits = instruction_logits.unsqueeze(1)
                x = torch.cat((image_feats, text_feats, instruction_logits), dim=1)
            else:
                x = torch.cat((image_feats, text_feats, instruction_logits), dim=1)
        for layer in range(self.num_layers):
            attn_output, _ = self.multihead_attns[layer](x, x, x)  # Self-attention
            x = x + attn_output if self.residual==0 else x
            x = self.linears[layer](x) if self.num_layers>1 else x

        return {"scores": self.linear(x)}


