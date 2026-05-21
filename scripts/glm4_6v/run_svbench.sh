EVAL_DATA_DIR=/home/zhonghua/softwares/data/sv_bench
OUTPUT_DIR=eval_output
CKPT=softlink/hf_model/zai-org/GLM-4.6V-Flash

# conda activate soccer
# CUDA_VISIBLE_DEVICES=0

logname=glm4_6v_log.txt
fps=1.0
method=Uniform

mkdir -p "${OUTPUT_DIR}"

echo '-----------' >> "${OUTPUT_DIR}/${logname}"
# for task_name in action_classification commentary_generation fouls_penalties offside_judgement player_identification space_identification time_allocation; do
for task_name in action_classification; do
    echo "${task_name}"

    output_folder=${OUTPUT_DIR}/glm4_6v/SVBench/${task_name}/answers/${method}
    output_file=${output_folder}/merge.jsonl
    mkdir -p "${output_folder}"

    TRANSFORMERS_OFFLINE=1 python3 -m MLLM.glm4_6v.eval.inference_svbench \
        --model-path "${CKPT}" \
        --task-name "${task_name}" \
        --video-folder "${EVAL_DATA_DIR}" \
        --question-file "${EVAL_DATA_DIR}/json" \
        --output-file "${output_folder}/1_0.jsonl" \
        --fps "${fps}"

    > "${output_file}"
    cat "${output_folder}/1_0.jsonl" >> "${output_file}"
    echo -n "task_name:${task_name} method:${method} fps:${fps} " >> "${OUTPUT_DIR}/${logname}"
    python3 MLLM/glm4_6v/eval/eval_rouge.py \
        --pred-path "${output_file}" \
        --output-dir "${output_folder}/" \
        --log-path "${OUTPUT_DIR}/${logname}"
    echo '' >> "${OUTPUT_DIR}/${logname}"
done
