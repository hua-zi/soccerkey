"""Backward-compatible entry point for GLM-4.6V SVBench inference."""

try:
    from .inference_svbench import run_inference
except ImportError:
    from inference_svbench import run_inference


if __name__ == "__main__":
    try:
        from .inference_svbench import argparse, os, tasks, DEFAULT_DATA_ROOT, str2bool
    except ImportError:
        from inference_svbench import argparse, os, tasks, DEFAULT_DATA_ROOT, str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--base-url', '--base_url', dest='base_url', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api-key', type=str, default="EMPTY")
    parser.add_argument('--video-folder', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--question-file', default=None)
    parser.add_argument('--output-file', required=True)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--task-name", type=str, required=True, choices=list(tasks.keys()))
    parser.add_argument("--keyframe-mode", type=str, required=False, default='Uniform', choices=['Uniform'])
    parser.add_argument("--fps", type=float, required=False, default=1.0)
    parser.add_argument("--resize-for-memory", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--restore-dir", type=str, default="restore/glm4_6v")

    args = parser.parse_args()
    if args.question_file is None:
        args.question_file = os.path.join(args.video_folder, "json")

    print(f'task-name:{args.task_name}  keyframe-mode:{args.keyframe_mode}  fps:{args.fps}')
    run_inference(args)
