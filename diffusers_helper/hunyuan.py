import torch
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask  # Not used in this snippet

_LATENT_RGB_FACTORS = torch.tensor([
    [-0.0395, -0.0331, 0.0445], [0.0696, 0.0795, 0.0518], [0.0135, -0.0945, -0.0282],
    [0.0108, -0.0250, -0.0765], [-0.0209, 0.0032, 0.0224], [-0.0804, -0.0254, -0.0639],
    [-0.0991, 0.0271, -0.0669], [-0.0646, -0.0422, -0.0400], [-0.0696, -0.0595, -0.0894],
    [-0.0799, -0.0208, -0.0375], [0.1166, 0.1627, 0.0962], [0.1165, 0.0432, 0.0407],
    [-0.2315, -0.1920, -0.1355], [-0.0270, 0.0401, -0.0821], [-0.0616, -0.0997, -0.0727],
    [0.0249, -0.0469, -0.1703]
], dtype=torch.float32).transpose(0, 1)[:, :, None, None, None]

_LATENT_RGB_BIAS = torch.tensor([0.0259, -0.0192, -0.0761], dtype=torch.float32)


@torch.no_grad()
def encode_prompt_conds(prompt: str, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)
    prompt = [prompt]

    # LLAMA
    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = llama_inputs.input_ids.to(text_encoder.device)
    attention_mask = llama_inputs.attention_mask.to(text_encoder.device)

    outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-3]

    attn_len = int(attention_mask.sum())
    vec = hidden_states[:, crop_start:attn_len]
    attn_mask = attention_mask[:, crop_start:attn_len]

    assert torch.all(attn_mask.bool())

    # CLIP
    clip_inputs = tokenizer_2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    clip_ids = clip_inputs.input_ids.to(text_encoder_2.device)
    clip_pooler = text_encoder_2(clip_ids, output_hidden_states=False).pooler_output

    return vec, clip_pooler


@torch.no_grad()
def vae_decode_fake(latents: torch.Tensor) -> torch.Tensor:
    weight = _LATENT_RGB_FACTORS.to(device=latents.device, dtype=latents.dtype)
    bias = _LATENT_RGB_BIAS.to(device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias)
    return images.clamp(0.0, 1.0)


@torch.no_grad()
def vae_decode(latents: torch.Tensor, vae, image_mode=False) -> torch.Tensor:
    latents = latents.to(vae.device, vae.dtype) / vae.config.scaling_factor

    if not image_mode:
        return vae.decode(latents).sample

    frames = [vae.decode(l.unsqueeze(2)).sample for l in latents.unbind(2)]
    return torch.cat(frames, dim=2)


@torch.no_grad()
def vae_encode(image: torch.Tensor, vae) -> torch.Tensor:
    image = image.to(vae.device, vae.dtype)
    latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    return latents
