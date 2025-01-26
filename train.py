from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

ds = load_dataset("fka/awesome-chatgpt-prompts")

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 处理填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 数据预处理函数
def preprocess_function(examples):
    acts = examples["act"]
    prompts = examples["prompt"]
    inputs = []
    for act, prompt in zip(acts, prompts):
        # 拼接 act 和 prompt，这里可以根据实际需求调整拼接方式
        input_text = f"{act}: {prompt}"
        inputs.append(input_text)
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

    # 对于因果语言模型，标签通常和输入 ID 相同
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# 对数据集进行预处理
tokenized_datasets = ds.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=2e-5
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

# 开始训练
trainer.train()

# 保存训练后的模型
trainer.save_model('trained_qwen_model')
