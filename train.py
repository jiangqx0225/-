import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

# 1. 数据加载
# 假设 JSON 文件名为 data.json
with open('./ruozhiba/data/ruozhiba-post-annual.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将数据转换为 Dataset 对象
dataset = Dataset.from_list(data)

# 2. 模型和分词器加载
model_name = "qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 处理填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 数据预处理函数
def preprocess_function(examples):
    # 这里我们使用 content 字段作为输入
    contents = examples["content"]
    # 对内容进行分词处理
    model_inputs = tokenizer(contents, truncation=True, padding="max_length", max_length=512)
    # 对于因果语言模型，标签通常和输入 ID 相同
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# 对数据集进行预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. 训练参数设置
training_args = TrainingArguments(
    output_dir='./results',  # 训练结果保存的目录
    num_train_epochs=3,  # 训练轮数
    per_device_train_batch_size=2,  # 每个设备的训练批次大小
    save_steps=10_000,  # 每多少步保存一次模型
    save_total_limit=2,  # 最多保存的模型数量
    prediction_loss_only=True,  # 只计算预测损失
    learning_rate=2e-5  # 学习率
)

# 5. 初始化 Trainer 并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# 开始训练
trainer.train()

# 6. 保存模型和分词器
model_save_path = 'trained_qwen_model'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")
