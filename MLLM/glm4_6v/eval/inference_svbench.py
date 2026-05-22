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
    messages = [x['messages'] for x in batch]
    questions = [x['question'] for x in batch]
    question_ids = [x['question_id'] for x in batch]
    answers = [x['answer'] for x in batch]
    return messages, questions, question_ids, answers


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
        question = data['Q']
        options = get_options(data)

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
            'question_id': data['video_name'],
            'answer': data['openA'],
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


def build_svbench_eval(args, task_name):
    task = tasks[task_name]
    question_dir = args.question_file
    if os.path.isdir(os.path.join(question_dir, "json")):
        question_dir = os.path.join(question_dir, "json")

    json_file = os.path.join(question_dir, task[0])
    vis_folder = os.path.dirname(json_file)
    json_data = load_svbench_json(json_file)

    data_list = []
    for data in json_data:
        data_list.append({
            'task_type': task_name,
            'prefix': vis_folder,
            'data_type': 'video',
            'bound': None,
            'data': data,
        })

    dataset = SVBenchDataset(data_list, fps=args.fps)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)


def clean_option_prefix(output):
    return re.sub(r'^\s*\(O\d+\)\s*', '', output).strip()

def run_inference(args):
    setup_seed(43)
    disable_torch_init()

    val_loader = build_svbench_eval(args, args.task_name)
    model, processor = model_init(args.model_path)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    for i, (messages, questions, question_ids, answers) in enumerate(tqdm(val_loader)):
        # import ipdb; ipdb.set_trace()
        messages = messages[0]
        question = questions[0]
        question_id = question_ids[0]
        answer = answers[0]

        output = mm_infer(
            messages,
            model=model,
            processor=processor,
            modal='video',
            do_sample=False,
            video_id=question_id,
            max_new_tokens=max(len(answer.split(' ')) * 2, 32),
        )
        output = clean_option_prefix(output)
        clean_cache(model)

        ans_file.write(json.dumps({'id': question_id, 'question': question, 'answer': answer, 'pred': output}) + '\n')
        if i == 2:
            break
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
    parser.add_argument("--task-name", type=str, required=True, default='action_classification', choices=list(tasks.keys()))
    parser.add_argument("--fps", type=float, required=False, default=1.0)

    args = parser.parse_args()
    if args.question_file is None:
        args.question_file = os.path.join(args.video_folder, "json")

    print(f'task-name:{args.task_name}  fps:{args.fps}')
    run_inference(args)
