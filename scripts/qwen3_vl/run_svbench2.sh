EVAL_DATA_DIR=/home/zhonghua/softwares/data/sv_bench
OUTPUT_DIR=eval_output
CKPT=softlink/hf_model/Qwen/Qwen3-VL-8B-Instruct

# conda activate soccer
# CUDA_VISIBLE_DEVICES=0

logname=qwen3_vl_log2.txt
fps=1.0
keyframe_mode=Uniform

mkdir -p "${OUTPUT_DIR}"

echo '-----------' >> "${OUTPUT_DIR}/${logname}"
# for task_name in action_classification commentary_generation fouls_penalties offside_judgement player_identification space_identification time_allocation; do
for task_name in causal_inference; do
    echo "${task_name}"

    output_folder=${OUTPUT_DIR}/qwen3_vl/SVBench/${task_name}/answers/${keyframe_mode}
    output_file=${output_folder}/merge.jsonl
    mkdir -p "${output_folder}"

    TRANSFORMERS_OFFLINE=1 python3 -m MLLM.qwen3_vl.eval.inference_svbench \
        --model-path "${CKPT}" \
        --task-name "${task_name}" \
        --video-folder "${EVAL_DATA_DIR}" \
        --question-file "${EVAL_DATA_DIR}/json" \
        --output-file "${output_folder}/1_0.jsonl" \
        --keyframe-mode "${keyframe_mode}" \
        --fps "${fps}"

    > "${output_file}"
    cat "${output_folder}/1_0.jsonl" >> "${output_file}"
    echo -n "task_name:${task_name} keyframe_mode:${keyframe_mode} fps:${fps} " >> "${OUTPUT_DIR}/${logname}"
    python3 MLLM/qwen3_vl/eval/eval_rouge.py \
        --pred-path "${output_file}" \
        --output-dir "${output_folder}/" \
        --log-path "${OUTPUT_DIR}/${logname}"
    echo '' >> "${OUTPUT_DIR}/${logname}"
done
