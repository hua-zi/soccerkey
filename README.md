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

CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server 'softlink/hf_model/LLM-Research/Llama-3.2-11B-Vision' --server-port 23333 --tp 1 --cache-max-entry-count 0.4

CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server 'softlink/hf_model/Qwen/Qwen3-VL-8B-Instruct' --server-port 23333 --tp 1 --cache-max-entry-count 0.4


# 再开一个终端：测试 和 指标计算
bash script/run_svbench.sh
```