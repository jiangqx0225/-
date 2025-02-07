import json


instruction = ""  # Define your instruction
file_path = "" # Define file path
key = ""
value = ""

# Original dataset in prompt-completion format
with open(file_path, "r", encoding='utf-8') as f:
  original_data = json.load(f)

# Transform to instruct-output format
transformed_data = []

for item in original_data:
    transformed_item = {
        "instruction": instruction,
        "input": item[key],
        "output": item[value]
    }
    transformed_data.append(transformed_item)

# Print the transformed data
with open("./cleaned_data.json" , "w", encoding="utf-8") as outf:
  json.dumps(transformed_data, outf, ensure_ascii=False, indent=4)
