import argparse
import os

def run(args):
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from unsloth import is_bfloat16_supported
    PatchDPOTrainer()
    import torch
    from transformers import TrainingArguments
    from trl import DPOTrainer
    import json
    from datasets import Dataset
    
    
    with open(args.dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        loftq_config=args.loftq_config,
    )
    
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio = arg.warmup_ratio,
            # max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim = "adamw_8bit",
            seed=args.seed,
            output_dir=args.output_dir,
        ),
        beta = 0.1,
        train_dataset = dataset,
        # eval_dataset = YOUR_DATASET_HERE,
        tokenizer = tokenizer,
        max_length = 1024,
        max_prompt_length = 512,
    )
    dpo_trainer.train()
    
    if args.save_model:
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                print(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=quantization_method,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")


if __name__ == "__main__":

    # Define argument parser
    parser = argparse.ArgumentParser(description="🦥 Fine-tune your llm faster using unsloth!")

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Model name to load")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Use 4bit quantization to reduce memory usage")
    model_group.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned", help="Huggingface dataset to use for training")

    lora_group = parser.add_argument_group("🧠 LoRA Options", "These options are used to configure the LoRA model.")
    lora_group.add_argument('--r', type=int, default=16, help="Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_dropout', type=float, default=0, help="LoRA dropout rate, default is 0.0 which is optimized.")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility, default is 3407.")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")

   
    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=4, help="Batch size per device during training, default is 4.")
    training_group.add_argument('--num_train_epochs', type=int, default=3, help="Number of train epochs, default is 3.")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of gradient accumulation steps, default is 8.")
    training_group.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio, default is 0.1.")
    # training_group.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps.")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate, default is 2e-4.")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer type.")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay, default is 0.01.")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type, default is 'linear'.")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility, default is 3407.")
    

    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="tensorboard",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps, default is 1")

    # Saving and pushing arguments
    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model, default is 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+",
        help="Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf ")

    push_group = parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")

    args = parser.parse_args()
    run(args)
