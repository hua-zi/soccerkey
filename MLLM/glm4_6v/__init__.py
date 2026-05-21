"""GLM-4.6V model loading and multimodal inference helpers."""

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration


def model_init(model_path=None, modal='video', **kwargs):
    model_path = "zai-org/GLM-4.6V-Flash" if model_path is None else model_path
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def mm_infer(messages, model, processor, modal='video', video_id=None, save_attentions=False, frame_idx=None, nframes=None, **kwargs):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    output_text = output_text.split("</answer>")[0].strip()
    return output_text


def clean_cache(model):
    """Clean cache for glm4.6v model."""
    torch.cuda.empty_cache()
