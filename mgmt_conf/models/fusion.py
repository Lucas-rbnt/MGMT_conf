# Standard libraries
from typing import Dict, Optional, Tuple

import torch

# Third-party libraries
from einops import rearrange, reduce
from torch import einsum, nn

__all__ = ["MultiHeadAttention", "EmbraceWithAttentionNet"]


class MultiHeadAttention(nn.Module):
    """
    Explicitly copied from https://boring-guy.sh/posts/masking-rl/ with the author approbation.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(
            dim, inner_dim * 3, bias=False
        )  # Wq,Wk,Wv for each vector, thats why *3
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.heads

        # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qkv = self.to_qkv(x)

        # split into multi head attentions
        q, k, v = rearrange(qkv, "b n (h qkv d) -> b h n qkv d", h=h, qkv=3).unbind(
            dim=-2
        )

        # Batch matrix multiplication by QK^t and scaling
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # follow the softmax,q,d,v equation in the paper
        # softmax along row axis of the attention card
        attn = dots.softmax(dim=-1)

        # product of v times whatever inside softmax
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # concat heads into one matrix
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn


class EmbraceWithAttentionNet(nn.Module):
    """
    This class is an adapted version of https://github.com/idearibosome/embracenet.
    """

    def __init__(
        self,
        device: torch.device,
        modalities_size: Dict[str, int],
        embracement_size: int = 256,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.embracement_size = embracement_size
        self.use_attention = use_attention
        self.modalities_size = modalities_size
        if self.use_attention:
            self.attention = MultiHeadAttention(
                dim=self.embracement_size, heads=8, dim_head=64
            ).to(device)

        for modality in self.modalities_size.keys():
            setattr(
                self,
                f"docking_{modality}",
                nn.Linear(self.modalities_size[modality], self.embracement_size),
            )

    def forward(
        self,
        input: torch.tensor,
        availabilities: Optional[torch.tensor] = None,
        selection_probabilities: Optional[torch.tensor] = None,
    ) -> torch.tensor:

        num_modalities = len(input)
        batch_size = list(input.values())[0].size(0)

        docking_output_list = []
        for key in sorted(input):
            x = getattr(self, f"docking_{key}")(input[key])
            x = nn.functional.relu(x)
            docking_output_list.append(x)
            docking_output = torch.stack(docking_output_list, dim=1)

        # check availabilities
        if availabilities is None:
            availabilities = torch.ones(
                batch_size, num_modalities, dtype=torch.float, device=self.device
            )
        else:
            availabilities = availabilities.float()

        # adjust selection probabilities
        if selection_probabilities is None:
            selection_probabilities = torch.ones(
                batch_size, num_modalities, dtype=torch.float, device=self.device
            )
        selection_probabilities = torch.mul(selection_probabilities, availabilities)

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)
        if self.use_attention:
            docking_output, _ = self.attention(docking_output)
        # embrace
        modality_indices = torch.multinomial(
            selection_probabilities, num_samples=self.embracement_size, replacement=True
        )  # [batch_size, embracement_size]
        modality_toggles = nn.functional.one_hot(
            modality_indices, num_classes=num_modalities
        ).float()  # [batch_size, embracement_size, num_modalities]

        embracement_output_stack = torch.mul(
            docking_output.permute(0, 2, 1), modality_toggles
        )
        embracement_output = torch.sum(
            embracement_output_stack, dim=-1
        )  # [batch_size, embracement_size]

        return embracement_output
