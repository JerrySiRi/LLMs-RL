set -x

# 开启调试模式，打印【每一条执行的命令，便于排查】
# 这里定义了一个checkSuccess函数，用于检查上一个命令的执行状态
# 如果上一个命令失败（$? = 上一个命令），则打印错误信息并退出脚本
checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

mkdir -p ./checkpoint/llama-2-8b-csft
RM_OUTPUT=./checkpoint/llama-2-8b-csft/rm.jsonl

# step 1: reward model推理打分，生成训练标签【最后用deepspeed做inference】
# 多行字符串赋值命令，定义了 get_rewards_commands，它等价于一条实际要运行的 openrlhf.cli.batch_inference 命令：
# read 标准输入中读取一行文本，并赋值给变量
# -r 原样读取
# -d '' ：读取到 EOF（End of File）为止，d是delimiter， ' '是空格作为分隔符
read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference \
    --eval_task rm \
    --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
    --bf16 \
    --max_len 4096 \
    --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --input_key chosen \
    --apply_chat_template \
    --max_samples 128000 \
    --zero_stage 0 \
    --post_processor csft \
    --normalize_reward
    --micro_batch_size 4 \
    --output_path $RM_OUTPUT
EOF

# step 2: 使用奖励模型的输出数据进行有监督微调【CSFT】【最后用deepspeed做sft训练】
# 使用 reward model 打分之后的输出，作为训练数据，对 SFT 模型进行训练（也就是 CSFT = conditionally supervised fine-tuning）。
# DEF：
# CSFT（Conditionally Supervised Fine-Tuning）是一种监督微调方法，
# 其中模型被训练去预测在特定语境【被reward model打分后的对话 - chosen or rejected】下更优的回答，而非简单模仿人类数据。
read -r -d '' sft_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 4096 \
    --dataset $RM_OUTPUT \
    --dataset_probs 1.0 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
    --save_path ./checkpoint/llama-3-8b-csft \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 5e-6 \
    --gradient_checkpointing
EOF

if [ ! -e $RM_OUTPUT ]; then
    deepspeed --module $get_rewards_commands
    checkSuccess "RM"
fi
deepspeed --module $sft_commands