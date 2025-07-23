set -x

# 1台机器，8块A100

# 这里num_nodes都设置为1 -- 我只有1台机器
# 需要调整：nums_gpus_per_node & vLLm的并行度
# 法一：手动拆分，明确各个角色的卡的数量
# 法二：Hybrid Engine自动拆分所有模型的卡的数量【此处--colocate_all_models】
#        这里Reference, Actor, Reward, Critic, vLLM全部放到8张卡上，自己进行动态调度

# 【rollout number】--n_samples_per_prompt：每个 prompt 要采样的样本数（也就是每个 prompt 会生成多少条独立的轨迹）；
# 【一个batch（更新一次）生成的rollout data】rollout_batch_size * n_samples_per_prompt = 128*8 = 1024

# BUG 
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.7 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --agent_func_path /openrlhf/examples/python/agent_func.py \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --kl_estimator k2 \
