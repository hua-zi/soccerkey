"""Run GLM-4.6V inference on SVBench with 1 FPS video sampling."""

import argparse
import json
import os
import re
import sys

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('./')
sys.path.append('./MLLM')
from glm4_6v import model_init, mm_infer, clean_cache
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


def collate_fn(batch):
    aud_vid  = [x['messages'] for x in batch]
    qus  = [x['question'] for x in batch]
    qid  = [x['question_id'] for x in batch]
    ans  = [x['answer'] for x in batch]
    return aud_vid, qus, qid, ans


class SVBenchDataset(Dataset):

    def __init__(self, data_list, fps=1.0):
        self.data_list = data_list
        self.fps = fps
        self.instruction = """
            You are a football expert. You are given a question Q and multiple answer options labeled O1, O2, O3, O4, ....
            Select the single option that best answers the question.
            Output only the content of the chosen option, not the option label (e.g., do not output “O1”).
            Do not include any other text or explanation.
        """.strip()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = item['data']
        video_paths = get_video_paths(data, item['prefix'])
        video_name = data['video_name']
        question = data['Q']
        options = get_options(data)
        answer = data['openA']
        question_id = video_name

        options_string = ''
        for option_key, option_value in options:
            options_string += f"({option_key}) {option_value}\n"

        instruct = f'Question: {question}\nOptions:\n{options_string}'
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {
                "role": "user",
                "content": [
                    *[build_video_content(video_path, self.fps) for video_path in video_paths],
                    {"type": "text", "text": instruct},
                ],
            },
        ]

        return {
            'messages': messages,
            'question': question,
            'question_id': question_id,
            'answer': answer
        }


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def build_video_content(video_path, fps):
    return {
        "type": "video",
        "video": video_path,
        "fps": fps,
    }


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
    data_list = []
    # for task_name, task in tasks.items():
    task = tasks[task_name]
    question_dir = args.question_file
    if os.path.isdir(os.path.join(question_dir, "json")):
        question_dir = os.path.join(question_dir, "json")
    json_file = os.path.join(question_dir, task[0])
    vis_folder = os.path.dirname(json_file)
    json_data = load_svbench_json(json_file)
    for i, data in enumerate(json_data):
        # if i == 3:
        #     break
        if data.get('video_name') in done_ids:
            continue
        data_list.append({
            'task_type': task_name,
            'prefix': vis_folder,
            'data_type': 'video',
            'bound': None,
            'data': data
        })
    dataset = SVBenchDataset(data_list, fps=args.fps)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    return dataloader


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


def load_include_frame_idx(args, keyframe_mode, nframes=32):
    if keyframe_mode == 'Uniform':
        return None
    elif keyframe_mode == 'ASK':
        file_path = f'./frame_idx/longvideobench/AKS/frame_idx_{nframes}.json'
    elif keyframe_mode == 'KFC':
        file_path = f'./frame_idx/longvideobench/KFC/frame_idx_{nframes}_15_{args.threshold}.json'
    elif keyframe_mode == 'FOCUS':
        file_path = f'./frame_idx/longvideobench/FOCUS/frame_idx_{nframes}.json'
    elif keyframe_mode == 'KFCblip2':
        file_path = f'./frame_idx/longvideobench/KFCblip2/frame_idx_{nframes}_15_{args.threshold}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    include_frame_idx_dict = {}
    for item in data:
        video_id = item['id']
        frame_idx = item['frame_idx']
        include_frame_idx_dict[video_id] = frame_idx
    return include_frame_idx_dict


def run_inference(args):
    setup_seed(43)
    disable_torch_init()

    answer_file = os.path.expanduser(args.output_file)
    done_ids = load_done_ids(answer_file) if args.resume else set()
    if done_ids:
        print(f'Resume enabled, skip {len(done_ids)} finished samples.')
    
    # 数据和模型
    val_loader = build_svbench_eval(args, args.task_name, done_ids=done_ids)
    model, processor = model_init(args.model_path)

    # 抽帧
    include_frame_idx = load_include_frame_idx(args, keyframe_mode=args.keyframe_mode, nframes=args.nframes)

    # save answer
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "a" if args.resume else "w", encoding='utf-8')
    
    # NOTE: only support batch size 1 for now
    for i, (messages, questions, question_ids, answers) in enumerate(tqdm(val_loader)):
        # if i < 7:
        #     continue
        messages     = messages[0]
        question     = questions[0]
        question_id  = question_ids[0]  # qjY9kmveQAk_0
        answer       = answers[0]   # 'Ate the medicine.'

        output = mm_infer(
            messages, # dict_keys(['video', 'audio']):torch.Size([8, 3, 384, 384]),torch.Size([1, 2998, 128])
            model=model,
            processor=processor,
            modal='video',
            do_sample=False,
            # save_attentions=True,
            video_id=question_id,
            max_new_tokens = max(len(answer.split(' ')) * 2, 32),
            frame_idx=include_frame_idx[question_id] if include_frame_idx is not None else None,
            nframes=len(include_frame_idx[question_id]) if include_frame_idx is not None else args.nframes,
        )
        output = clean_option_prefix(output)
        clean_cache(model)
        # import ipdb; ipdb.set_trace()
        ans_file.write(json.dumps({'id': question_id, 'question': question, 'answer': answer, 'pred': output}) + '\n')
        ans_file.flush()
        # if i == 2:
        #     break
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='SVBench root directory.', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--question-file', help='Directory containing SVBench json files.', default=None)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', default=None)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action='store_true', help='Append to output file and skip finished sample ids.')

    parser.add_argument("--task-name", type=str, required=True, default='action_classification', choices=list(tasks.keys()))
    parser.add_argument("--keyframe-mode", type=str, required=False, default='Uniform', choices=['ASK', 'FOCUS', 'KFC', 'KFCblip2', 'Uniform'])
    parser.add_argument("--nframes", type=int, required=False, default=32)
    parser.add_argument("--fps", type=float, required=False, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.01)

    args = parser.parse_args()
    if args.question_file is None:
        args.question_file = os.path.join(args.video_folder, "json")

    print(f'task-name:{args.task_name}  keyframe-mode:{args.keyframe_mode}  fps:{args.fps}')
    run_inference(args)
