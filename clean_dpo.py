import json
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Transform dataset to instruct-output format.")
parser.add_argument("--data", type=str, required=True, help="Path to the original dataset file.")
parser.add_argument("--prompt", type=str, default="prompt", help="Instruction for the dataset.")
parser.add_argument("--chosen", type=str, default="chosen", help="Value in the original data to use as chosen.")
parser.add_argument("--rejected", type=str, default="rejected", help="Value in the original data to use as reject.")

args = parser.parse_args()

# 读取原始数据集
with open(args.data_path, "r", encoding='utf-8') as f:
    original_data = json.load(f)

# 转换为 instruct-output 格式
transformed_data = []

for item in original_data:
    transformed_item = {
        "prompt": item[args.prompt],
        "chosen": item[args.chosen],
        "rejected": item[args.rejected]
    }
    transformed_data.append(transformed_item)

# 保存转换后的数据
output_path = f"./{args.data_path}_cleaned_data.json"
with open(output_path, "w", encoding="utf-8") as outf:
    json.dump(transformed_data, outf, ensure_ascii=False, indent=4)

print(f"Transformed data saved to {output_path}")
