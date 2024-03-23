import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .llama import LLaMATransformer
import time
from pathlib import Path

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTLLama(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., model_name = "llm",
                 requires_grad = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


        # LLaMA
        self.model_name = model_name
        llama_default_config = {"dim": 4096, "multiple_of": 256,
                                "n_heads": 32, "n_layers": 32, "norm_eps": 1.0e-6,
                                "vocab_size": -1, "first_layer": 31}
        self.llama = LLaMATransformer(llama_default_config)

        # load llama checkpoint for the encoder layer
        model_name = "llama"
        llama_path = "./llama/pyllama_data/7B"
        if 'llama' in model_name:
            print("Loading LLaMA checkpoints")
            start_time = time.time()
            checkpoints = sorted(Path(llama_path).glob("*.pth"))
            ckpt_path = checkpoints[0]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
            print(f"Loaded in {time.time() - start_time:.2f} seconds")

        for param in self.llama.parameters():
            param.requires_grad = requires_grad
        self.llama_dim_mapper1 = nn.Linear(dim, 4096, bias=False)
        self.llama_dim_mapper2 = nn.Linear(4096, dim, bias=False)
        self.llama_dim_dim_mapper = nn.Linear(dim, dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.forward_features(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


    def forward_features(self, x):

        if self.model_name == "llm_in": # residual within nn
            x = self.llama_dim_mapper1(x)
            x = self.llama(x) + x
            x = self.llama_dim_mapper2(x)
        elif self.model_name == "llm_out": # residual out of nn
            tmp = x
            x = self.llama_dim_mapper1(x)
            x = self.llama(x)
            x = self.llama_dim_mapper2(x)
            x += tmp
        elif self.model_name == "llm_hybrid": # residual in and out of nn
            tmp = x
            x = self.llama_dim_mapper1(x)
            x = self.llama(x) + x
            x = self.llama_dim_mapper2(x)
            x += tmp
        elif self.model_name == "llm_out_nn": # residual in and out of nn
            tmp = x
            x = self.llama_dim_mapper1(x)
            x = self.llama(x)
            x = self.llama_dim_mapper2(x)
            x += self.leakyrelu(self.llama_dim_dim_mapper(tmp))
        elif self.model_name == "llm_mlp":  # mlp without llm
            x = self.llama_dim_mapper1(x)
            x = self.llama_dim_mapper2(x)
        else: # no residual
            x = self.llama_dim_mapper1(x)
            x = self.llama(x)
            x = self.llama_dim_mapper2(x)
        return x