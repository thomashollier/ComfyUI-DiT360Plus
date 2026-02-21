"""PersonalizeAnything attention processor for DiT360 inpainting/outpainting.

Based on: https://github.com/Insta360-Research-Team/DiT360/blob/main/pa_src/attn_processor.py
"""

from typing import Callable, Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


def set_flux_transformer_attn_processor(
    transformer,
    set_attn_proc_func: Callable,
    set_attn_module_names: Optional[list] = None,
) -> None:
    """Replace attention processors on a FLUX transformer."""
    do_set_processor = lambda name, module_names: (
        any(name.startswith(mn) for mn in module_names)
        if module_names is not None
        else True
    )

    attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        dim_head = transformer.config.attention_head_dim
        num_heads = transformer.config.num_attention_heads
        if name.endswith("attn.processor"):
            attn_procs[name] = (
                set_attn_proc_func(name, dim_head, num_heads, attn_processor)
                if do_set_processor(name, set_attn_module_names)
                else attn_processor
            )

    transformer.set_attn_processor(attn_procs)


class PersonalizeAnythingAttnProcessor:
    """Custom attention processor that preserves source content via token replacement.

    During denoising, at timesteps > tau, tokens in the edit batch at masked positions
    are replaced with the corresponding tokens from the source batch. This preserves
    the source content in the masked regions while allowing new generation elsewhere.

    Args:
        name: Attention layer name.
        mask: Boolean mask of shape (img_dims,). True = preserve from source.
        device: Target device.
        tau: Timestep threshold (0-1). Higher = more preservation.
        img_dims: Number of image tokens (latent_h * (latent_w + 2) for circular padding).
    """

    def __init__(self, name, mask, device, tau=0.5, img_dims=4096):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+ for scaled_dot_product_attention.")

        self.name = name
        self.mask = mask.view(img_dims).bool().to(device)
        self.device = device
        self.tau = tau
        self.img_dims = img_dims

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep=None,
    ) -> torch.FloatTensor:
        batch_size = (
            hidden_states.shape[0]
            if encoder_hidden_states is None
            else encoder_hidden_states.shape[0]
        )

        if timestep is not None:
            t_flag = timestep > self.tau
        else:
            t_flag = False

        r_q = t_flag
        r_k = t_flag
        r_v = t_flag

        # Extract source concept features from batch[0]
        if encoder_hidden_states is not None:
            concept_feature_ = hidden_states[0, self.mask, :]
        else:
            concept_feature_ = hidden_states[0, 512:, :][self.mask, :]

        # Token replacement: copy source tokens into edit batch at mask positions
        if r_k or r_q or r_v:
            r_hidden_states = hidden_states.clone()
            if encoder_hidden_states is not None:
                r_hidden_states[1, self.mask, :] = concept_feature_
            else:
                text_hs = hidden_states[1, :512, :]
                image_hs = hidden_states[1, 512:, :].clone()
                image_hs[self.mask, :] = concept_feature_
                r_hidden_states[1] = torch.cat([text_hs, image_hs], dim=0)

        # Compute Q, K, V
        key = attn.to_k(r_hidden_states if r_k else hidden_states)
        value = attn.to_v(r_hidden_states if r_v else hidden_states)
        query = attn.to_q(r_hidden_states if r_q else hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Joint attention (double transformer blocks)
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)

        # Apply RoPE
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

            # Trim to original image dimensions
            hidden_states = hidden_states[:, : self.img_dims, :]

            return hidden_states, encoder_hidden_states_out
        else:
            dims = self.img_dims + 512
            hidden_states = hidden_states[:, :dims, :]
            return hidden_states
