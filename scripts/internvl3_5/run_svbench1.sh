EVAL_DATA_DIR=/home/zhonghua/softwares/data/sv_bench
OUTPUT_DIR=eval_output
CKPT=softlink/hf_model/OpenGVLab/InternVL3_5-8B
BASE_URL=http://0.0.0.0:24333/v1

# conda activate soccer
# CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server "${CKPT}" --server-port 23333

logname=internvl3_5_log.txt
fps=1.0
keyframe_mode=Uniform
nframes=32

mkdir -p "${OUTPUT_DIR}"

echo '-----------' >> "${OUTPUT_DIR}/${logname}"
# for task_name in action_classification commentary_generation fouls_penalties offside_judgement player_identification space_identification time_allocation causal_inference; do
for task_name in player_identification space_identification time_allocation causal_inference; do
# for task_name in causal_inference; do
    echo "${task_name}"

    output_folder=${OUTPUT_DIR}/internvl3_5/SVBench/${task_name}/answers/${keyframe_mode}
    output_file=${output_folder}/merge.jsonl
    mkdir -p "${output_folder}"

    TRANSFORMERS_OFFLINE=1 python3 -m MLLM.internvl3_5.eval.inference_svbench \
        --model-path "${CKPT}" \
        --base-url "${BASE_URL}" \
        --task-name "${task_name}" \
        --video-folder "${EVAL_DATA_DIR}" \
        --question-file "${EVAL_DATA_DIR}/json" \
        --output-file "${output_folder}/1_0.jsonl" \
        --keyframe-mode "${keyframe_mode}" \
        --fps "${fps}" \
        --nframes "${nframes}" \
        --num-workers 0 \
        # --resume

    > "${output_file}"
    cat "${output_folder}/1_0.jsonl" >> "${output_file}"
    echo -n "task_name:${task_name} keyframe_mode:${keyframe_mode} fps:${fps} nframes:${nframes} " >> "${OUTPUT_DIR}/${logname}"
    python3 MLLM/internvl3_5/eval/eval_rouge.py \
        --pred-path "${output_file}" \
        --output-dir "${output_folder}/" \
        --log-path "${OUTPUT_DIR}/${logname}"
    echo '' >> "${OUTPUT_DIR}/${logname}"
done

# CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server 'softlink/hf_model/OpenGVLab/InternVL3_5-8B' --server-port 24333 --tp 1 --cache-max-entry-count 0.4