# soccerkey

```
# 进入代码目录
cd soccersight

# 启动 MLLM
lmdeploy serve api_server 'softlink/hf_model/Qwen/Qwen2.5-VL-7B-Instruct' --server-port 23333 --tp 1 --cache-max-entry-count 0.4
# lmdeploy serve api_server 'softlink/hf_model/Qwen/Qwen3-VL-8B-Instruct' --server-port 23333 --tp 1 --cache-max-entry-count 0.4
# CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server 'softlink/hf_model/OpenGVLab/InternVL3_5-8B' --server-port 23333 --tp 1 --cache-max-entry-count 0.4
# lmdeploy serve api_server 'softlink/hf_model/zai-org/GLM-4.6V-Flash' --server-port 23333 --tp 1 --cache-max-entry-count 0.4
# lmdeploy serve api_server 'softlink/hf_model/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct' --server-port 23333 --tp 1 --cache-max-entry-count 0.4

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server 'softlink/hf_model/unsloth/Llama-3.2-11B-Vision-Instruct' --server-port 23333 --tp 1 --cache-max-entry-count 0.4

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server 'softlink/hf_model/zai-org/GLM-4.1V-9B-Thinking' --server-port 23333 --tp 1 --cache-max-entry-count 0.4
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server 'softlink/hf_model/zai-org/glm-4v-9b' --server-port 23333 --tp 1 --cache-max-entry-count 0.4 --trust_remote_code

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server 'softlink/hf_model/Qwen/Qwen3-VL-8B-Instruct' --server-port 23333 --tp 1 --cache-max-entry-count 0.4

CUDA_VISIBLE_DEVICES=0 transformers serve --force-model softlink/hf_model/unsloth/Llama-3.2-11B-Vision-Instruct --port 23333 --trust-remote-code

transformers serve --port 23333 --device cuda --dtype auto --trust-remote-code

# 再开一个终端：测试 和 指标计算
bash script/run_svbench.sh
```