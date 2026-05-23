"""Run InternVL3.5 inference on SVBench through lmdeploy's OpenAI API."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from openai import OpenAI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('./')
sys.path.append('./MLLM')
from utils import setup_seed, disable_torch_init


DEFAULT_DATA_ROOT = "/home/zhonghua/softwares/data/sv_bench"

tasks = {
    "action_classification": ("action_classification.json",),
    "commentary_generation": ("commentary_generation.json",),
    "fouls_penalties": ("fouls_penalties.json",),
    "offside_judgement": ("offside_judgement.json",),
    "player_identification": ("player_identification.json",),
    "space_identification": ("space_identification.json",),
    "time_allocation": ("time_allocation.json",),
    "causal_inference": ("causal_inference.json",),
}


@dataclass
class RuntimeConfig:
    fps: float = 1.0
    resize_for_memory: bool = True
    restore_dir: str = "restore"


def collate_fn(batch):
    video_paths = [x['video_paths'] for x in batch]
    prompts = [x['prompt'] for x in batch]
    questions = [x['question'] for x in batch]
    question_ids = [x['question_id'] for x in batch]
    answers = [x['answer'] for x in batch]
    return video_paths, prompts, questions, question_ids, answers


class SVBenchDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.instruction = """
            You are a football expert. You are given a question Q and multiple answer options labeled O1, O2, O3, O4, ....
            Select the single option that best answers the question.
            Output only the content of the chosen option, not the option label (e.g., do not output "O1").
            Do not include any other text or explanation.
        """.strip()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = item['data']
        question = data['Q']
        options = get_options(data)

        options_string = ''
        for option_key, option_value in options:
            options_string += f"({option_key}) {option_value}\n"

        prompt = f"Question: {question}\nOptions:\n{options_string}"

        return {
            'video_paths': get_video_paths(data, item['prefix']),
            'prompt': prompt,
            'question': question,
            'question_id': data['video_name'],
            'answer': data['openA'],
        }


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def get_video_paths(data, base_dir):
    if data.get("video_path"):
        return [resolve_path(data["video_path"], base_dir)]

    indexed_keys = []
    for key in data.keys():
        if key.startswith("video_path") and key[len("video_path"):].isdigit():
            indexed_keys.append((int(key[len("video_path"):]), key))
    indexed_keys.sort()

    video_paths = [resolve_path(data[key], base_dir) for _, key in indexed_keys if data.get(key)]
    if not video_paths:
        raise ValueError(f"No video path found for sample {data.get('video_name')}")
    return video_paths


def get_options(data):
    options = []
    for key, value in data.items():
        if key.startswith("O") and key[1:].isdigit():
            options.append((key, value))
    options.sort(key=lambda x: int(x[0][1:]))
    return options


def load_svbench_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    if isinstance(json_data, dict) and "QA_pairs" in json_data:
        return json_data["QA_pairs"]
    return json_data


def build_svbench_eval(args, task_name, done_ids=None):
    done_ids = done_ids or set()
    task = tasks[task_name]
    question_dir = args.question_file
    if os.path.isdir(os.path.join(question_dir, "json")):
        question_dir = os.path.join(question_dir, "json")

    json_file = os.path.join(question_dir, task[0])
    vis_folder = os.path.dirname(json_file)
    json_data = load_svbench_json(json_file)

    data_list = []
    for data in json_data:
        if data.get('video_name') in done_ids:
            continue
        data_list.append({
            'task_type': task_name,
            'prefix': vis_folder,
            'data_type': 'video',
            'bound': None,
            'data': data,
        })

    dataset = SVBenchDataset(data_list)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def load_done_ids(output_file):
    done_ids = set()
    if not output_file or not os.path.exists(output_file):
        return done_ids

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('id'):
                done_ids.add(item['id'])
    return done_ids


def clean_option_prefix(output):
    return re.sub(r'^\s*\(O\d+\)\s*', '', output).strip()


def sample_frame_indices(total_frames: int, video_fps: float, sample_fps: float) -> List[int]:
    if total_frames <= 0:
        return []

    if sample_fps and sample_fps > 0 and video_fps and video_fps > 0:
        step = max(int(round(video_fps / sample_fps)), 1)
        indices = list(range(0, total_frames, step))
    else:
        indices = list(range(total_frames))

    return [min(max(i, 0), total_frames - 1) for i in indices]


def sampling_cache_tag(cfg: RuntimeConfig) -> str:
    fps_tag = str(cfg.fps).replace(".", "_")
    return f"fps{fps_tag}"


def video_cache_dir(video_path: str, cfg: RuntimeConfig) -> str:
    abs_path = os.path.abspath(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    path_hash = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:10]
    return os.path.abspath(os.path.join(cfg.restore_dir, sampling_cache_tag(cfg), f"{video_name}_{path_hash}"))


def load_cached_frames(save_dir: str) -> Tuple[str, List[float]] | None:
    manifest_path = os.path.join(save_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    frame_files = manifest.get("frame_files") or []
    timestamps = manifest.get("timestamps") or []
    if len(frame_files) != len(timestamps) or not frame_files:
        return None

    if all(os.path.exists(os.path.join(save_dir, frame_file)) for frame_file in frame_files):
        return save_dir, timestamps
    return None


def save_video_frames(video_path: str, sample_id: str, video_idx: int, cfg: RuntimeConfig) -> Tuple[str, List[float]]:
    save_dir = video_cache_dir(video_path, cfg)
    cached = load_cached_frames(save_dir)
    if cached is not None:
        return cached

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        vr = VideoReader(video_path, ctx=cpu(0))

    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    frame_indices = sample_frame_indices(total_frames, video_fps, cfg.fps)
    if not frame_indices:
        raise ValueError(f"No frames sampled from video: {video_path}")

    os.makedirs(save_dir, exist_ok=True)

    frames = vr.get_batch(frame_indices).asnumpy()
    frame_files = []
    for i, frame in enumerate(frames):
        frame_file = f"{i:06d}.jpg"
        img = Image.fromarray(frame.astype("uint8")).convert("RGB")
        img.save(os.path.join(save_dir, frame_file))
        frame_files.append(frame_file)

    timestamps = [idx / video_fps if video_fps > 0 else float(idx) for idx in frame_indices]
    manifest = {
        "video_path": os.path.abspath(video_path),
        "fps": cfg.fps,
        "video_fps": video_fps,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "frame_files": frame_files,
    }
    with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return save_dir, timestamps


def build_openai_content(video_paths: Sequence[str], prompt: str, sample_id: str, cfg: RuntimeConfig) -> List[Dict]:
    text_parts = []
    image_items = []

    for video_idx, video_path in enumerate(video_paths):
        frames_dir, timestamps = save_video_frames(video_path, sample_id, video_idx, cfg)
        video_title = "Original video" if len(video_paths) == 1 else f"Original video {video_idx + 1}"
        text_parts.append(f"{video_title}:")

        for frame_idx, timestamp in enumerate(timestamps):
            text_parts.append(f"Frame{frame_idx} ({timestamp:.1f}s): {IMAGE_TOKEN}")
            image_url: Dict = {
                "max_dynamic_patch": 1,
                "url": os.path.abspath(os.path.join(frames_dir, f"{frame_idx:06d}.jpg")),
            }
            if cfg.resize_for_memory:
                image_url["max_pixels"] = 360 * 420
            image_items.append({"type": "image_url", "image_url": image_url})

    text_parts.append(prompt)
    content: List[Dict] = [{"type": "text", "text": "\n".join(text_parts)}]
    content.extend(image_items)
    return content


def ask_mllm(
    client: OpenAI,
    model_id: str,
    video_paths: Sequence[str],
    prompt: str,
    sample_id: str,
    cfg: RuntimeConfig,
    max_tokens: int,
) -> str:
    instruction = """
        You are a football expert. You are given a question Q and multiple answer options labeled O1, O2, O3, O4, ....
        Select the single option that best answers the question.
        Output only the content of the chosen option, not the option label (e.g., do not output "O1").
        Do not include any other text or explanation.
    """.strip()

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": build_openai_content(video_paths, prompt, sample_id, cfg)},
    ]
    out = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return out.choices[0].message.content or ""


def get_model_id(model_path: str | None) -> str:
    if model_path:
        return os.path.abspath(os.path.expanduser(model_path))
    raise RuntimeError("Please pass --model-path. It is also used as the OpenAI-compatible model id.")


def run_inference(args):
    setup_seed(43)
    disable_torch_init()

    answer_file = os.path.expanduser(args.output_file)
    done_ids = load_done_ids(answer_file) if args.resume else set()
    if done_ids:
        print(f'Resume enabled, skip {len(done_ids)} finished samples.')

    val_loader = build_svbench_eval(args, args.task_name, done_ids=done_ids)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model_id = get_model_id(args.model_path)
    print(f'OpenAI-compatible model: {model_id}')

    cfg = RuntimeConfig(
        fps=args.fps,
        resize_for_memory=args.resize_for_memory,
        restore_dir=args.restore_dir,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "a" if args.resume else "w", encoding='utf-8')

    # NOTE: only support batch size 1 for now
    for i, (video_paths, prompts, questions, question_ids, answers) in enumerate(tqdm(val_loader)):
        video_paths  = video_paths[0]
        prompt       = prompts[0]
        question     = questions[0]
        question_id  = question_ids[0]
        answer       = answers[0]

        output = ask_mllm(
            client=client,
            model_id=model_id,
            video_paths=video_paths,
            prompt=prompt,
            sample_id=question_id,
            cfg=cfg,
            max_tokens=max(len(answer.split(' ')) * 2, 32),
        )
        output = clean_option_prefix(output)
        
        ans_file.write(json.dumps({'id': question_id, 'question': question, 'answer': answer, 'pred': output}) + '\n')
        ans_file.flush()
        # if i==2:
        #     break
    ans_file.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Model path. Also used as the OpenAI-compatible model id.', required=True)
    parser.add_argument('--base-url', '--base_url', dest='base_url', type=str, default="http://0.0.0.0:23333/v1")
    parser.add_argument('--api-key', type=str, default="EMPTY")
    parser.add_argument('--video-folder', help='SVBench root directory.', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--question-file', help='Directory containing SVBench json files.', default=None)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', default=None)
    parser.add_argument('--output-file', help='Path to save the model results JSONL.', required=True)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action='store_true', help='Append to output file and skip finished sample ids.')
    parser.add_argument("--task-name", type=str, required=True, default='action_classification', choices=list(tasks.keys()))
    parser.add_argument("--keyframe-mode", type=str, required=False, default='Uniform', choices=['Uniform'])
    parser.add_argument("--fps", type=float, required=False, default=1.0)
    parser.add_argument("--resize-for-memory", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--restore-dir", type=str, default="restore")

    args = parser.parse_args()
    if args.question_file is None:
        args.question_file = os.path.join(args.video_folder, "json")

    print(f'task-name:{args.task_name}  keyframe-mode:{args.keyframe_mode}  fps:{args.fps}')
    run_inference(args)
