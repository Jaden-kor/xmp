import torch
from torch import nn, einsum
from einops import repeat, pack, unpack
from performer_pytorch import Performer, CrossAttention

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, CrossAttention(dim=lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, CrossAttention(dim=sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# multi-scale Performer

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Performer(dim = sm_dim, ff_dropout = dropout, attn_dropout = dropout, **sm_enc_params),
                Performer(dim = lg_dim, ff_dropout = dropout, attn_dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

class SeqEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        cnn_kernel_size,
        stride,
        emb_dim,
        seq_len,
        token_size,
        dropout = 0.
    ):
        super().__init__()
        assert (seq_len % token_size) == 0, 'Seq_len must be divisible by the patch size.'
        num_embedding = seq_len // token_size
        maxpool_size = token_size//stride

        self.to_byte_n_gram_embedding = nn.Sequential(
            nn.Conv1d(emb_dim, dim, kernel_size = cnn_kernel_size, stride = stride, padding = (cnn_kernel_size-1) // 2),
            nn.ReLU(),
            nn.MaxPool1d(maxpool_size)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_embedding + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.to_byte_n_gram_embedding(seq)
        x = x.permute(0,2,1)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x), ps

# XMP class

class XMP(nn.Module):
    def __init__(
        self,
        *,
        seq_len,
        emb_dim,
        num_classes,
        sm_cnn_kernel_size,
        lg_cnn_kernel_size,
        stride = 1,
        sm_dim,
        lg_dim,
        sm_token_size,
        sm_enc_depth = 2,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_token_size,
        lg_enc_depth = 3,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()

        self.sm_seq_embedder = SeqEmbedder(dim = sm_dim, cnn_kernel_size = sm_cnn_kernel_size, stride = stride, emb_dim = emb_dim, seq_len = seq_len, token_size = sm_token_size, dropout = emb_dropout)
        self.lg_seq_embedder = SeqEmbedder(dim = lg_dim, cnn_kernel_size = lg_cnn_kernel_size, stride = stride, emb_dim = emb_dim, seq_len = seq_len, token_size = lg_token_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

        self.embedding = nn.Embedding(256, emb_dim)

    def forward(self, series):

        series = self.embedding(series)
        series = series.permute(0,2,1)

        sm_tokens, sm_ps = self.sm_seq_embedder(series)
        lg_tokens, lg_ps = self.lg_seq_embedder(series)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, _ = unpack(sm_tokens, sm_ps, 'b * d')
        lg_cls, _ = unpack(lg_tokens, lg_ps, 'b * d')

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits
