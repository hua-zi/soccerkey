EVAL_DATA_DIR=/home/zhonghua/softwares/data/sv_bench
OUTPUT_DIR=eval_output
MODEL_PATH=$(realpath "softlink/hf_model/zai-org/GLM-4.6V-Flash")
BASE_URL=http://localhost:23333/v1
RESTORE_DIR=/home/zhonghua/hua_ws/code/soccerkey/restore

# conda activate soccer
# CUDA_VISIBLE_DEVICES=0 transformers serve "${MODEL_PATH}" --host 0.0.0.0 --port 8000 --trust-remote-code

logname=glm4_6v_log.txt
fps=1.0
method=Uniform

mkdir -p "${OUTPUT_DIR}" "${RESTORE_DIR}"

echo '-----------' >> "${OUTPUT_DIR}/${logname}"
# for task_name in action_classification commentary_generation fouls_penalties offside_judgement player_identification space_identification time_allocation causal_inference; do
for task_name in causal_inference; do
    echo "${task_name}"

    output_folder=${OUTPUT_DIR}/glm4_6v/SVBench/${task_name}/answers/${method}
    output_file=${output_folder}/merge.jsonl
    mkdir -p "${output_folder}"

    TRANSFORMERS_OFFLINE=1 python3 -m MLLM.glm4_6v.eval.inference_svbench \
        --model-path "${MODEL_PATH}" \
        --base-url "${BASE_URL}" \
        --task-name "${task_name}" \
        --video-folder "${EVAL_DATA_DIR}" \
        --question-file "${EVAL_DATA_DIR}/json" \
        --output-file "${output_folder}/1_0.jsonl" \
        --keyframe-mode "${method}" \
        --fps "${fps}" \
        --restore-dir "${RESTORE_DIR}" \
        --num-workers 0

    > "${output_file}"
    cat "${output_folder}/1_0.jsonl" >> "${output_file}"
    echo -n "task_name:${task_name} method:${method} fps:${fps} restore_dir:${RESTORE_DIR} " >> "${OUTPUT_DIR}/${logname}"
    python3 MLLM/glm4_6v/eval/eval_rouge.py \
        --pred-path "${output_file}" \
        --output-dir "${output_folder}/" \
        --log-path "${OUTPUT_DIR}/${logname}"
    echo '' >> "${OUTPUT_DIR}/${logname}"
done
