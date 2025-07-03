import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim=6, embed_dim=256, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, n_stack, input_dim)
        x = self.input_proj(x)  # → (batch, n_stack, embed_dim)
        attn_out, _ = self.attn(x, x, x)  # → (batch, n_stack, embed_dim)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x  # (batch, n_stack, embed_dim)

class AttentionQNetwork(nn.Module):
    def __init__(self, stack_input_dim=6, order_input_dim=24, embed_dim=256, n_heads=4, hidden_dim=128):
        super().__init__()
        # print("[DEBUG][QNetwork init] stack_input_dim:", stack_input_dim,
        #       "order_input_dim:", order_input_dim)
        self.encoder = SelfAttentionEncoder(input_dim=stack_input_dim, embed_dim=embed_dim, n_heads=n_heads)
        self.order_proj = nn.Linear(order_input_dim, embed_dim)      # 24维
        self.policy_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.wait_mlp = nn.Sequential(
            nn.Linear(order_input_dim, hidden_dim),  # 24维
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, order_vec, stack_vecs):
        """
        order_vec: (batch, order_input_dim=24)
        stack_vecs: (batch, n_stack, stack_input_dim=6)
        returns: action_scores: (batch, n_stack + 1)
        """
        batch_size, n_stack, _ = stack_vecs.shape
        stack_embed = self.encoder(stack_vecs)  # (batch, n_stack, embed_dim)
        order_embed = self.order_proj(order_vec).unsqueeze(1).expand(-1, n_stack, -1)  # (batch, n_stack, embed_dim)

        concat = torch.cat([stack_embed, order_embed], dim=-1)  # (batch, n_stack, 2 * embed_dim)
        stack_scores = self.policy_mlp(concat).squeeze(-1)  # (batch, n_stack)

        wait_score = self.wait_mlp(order_vec).squeeze(-1)  # (batch,)

        action_scores = torch.cat([stack_scores, wait_score.unsqueeze(1)], dim=1)  # (batch, n_stack + 1)
        return action_scores
