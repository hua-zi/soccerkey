import sys

from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
# from qwen_vl_utils import process_vision_info
from .v0_0_14_vision_process import process_vision_info_frame_idx
import torch

def model_init(model_path=None, modal='video', **kwargs):
    model_path = "Qwen/Qwen3-VL-7B-Instruct" if model_path is None else model_path
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    processor = Qwen3VLProcessor.from_pretrained(model_path)
    return model, processor

def mm_infer(messages, model, processor, modal='video', video_id=None, save_attentions=False, frame_idx=None, nframes=None, **kwargs):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info_frame_idx(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True, frame_idx=frame_idx, nframes=nframes)

    video_inputs, video_metadatas = zip(*video_inputs)  # 
    video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)   # [torch.Size([6, 3, 320, 448])], 
    
    # if modal == 'video' and video_inputs[0].shape[0] - 1 > 64:      # 只保留64帧，且去掉第一帧（因为第一帧通常是全黑的）
    #     indices = torch.linspace(0, video_inputs[0].shape[0] - 1, 64).long()
    #     video_inputs[0] = video_inputs[0][indices]
    
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs
    ).to(model.device)

    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, 
                                       max_new_tokens=max_new_tokens,
                                       temperature=0.01)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text

def clean_cache(model):
    """Clean cache for qwen3-vl model"""
    torch.cuda.empty_cache()
