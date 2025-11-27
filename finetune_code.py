from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Configuration
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = "./tmf_training_data.jsonl"
output_dir = "./tinyllama_oss_finetuned"

# 1. Load the dataset
dataset = load_dataset("json", data_files=dataset_path)

# 2. Load base model and tokenizer (CPU-compatible settings)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # Don't use auto device mapping on CPU
    torch_dtype=torch.float32  # Use full precision for CPU
)

# 3. Configure LoRA (CPU-compatible settings)
lora_config = LoraConfig(
    r=8,  # Reduced rank for CPU training
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("Trainable parameters after LoRA:")
model.print_trainable_parameters()

# 4. Define formatting function for chat template
def format_chat_template(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    # Apply chat template and add EOS token
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Ensure it ends with EOS for training to learn when to stop
    return {"text": formatted_text + tokenizer.eos_token}

# 5. Apply formatting and tokenize the dataset
formatted_dataset = dataset['train'].map(format_chat_template, remove_columns=["instruction", "response"])

def tokenize_function(examples):
    # Tokenize the formatted text
    # max_length should be sufficient for your examples (e.g., 256 or 512)
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 6. Define training arguments (CPU-compatible settings)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Smaller batch size for CPU
    gradient_accumulation_steps=8,  # Increased to compensate for smaller batch size
    gradient_checkpointing=False,  # Disabled for CPU training
    optim="adamw_torch",  # Standard AdamW optimizer
    logging_steps=1,
    learning_rate=2e-4,
    fp16=False,  # Disable mixed precision training on CPU
    save_strategy="epoch",
    push_to_hub=False,
    report_to="none",
)

# 7. Create Trainer and fine-tune
from transformers import DataCollatorForLanguageModeling

# Create a proper data collator that handles tokenized inputs
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets,
    args=training_args,
    data_collator=data_collator,
)

print("\n--- Starting Fine-tuning ---")
trainer.train()

# 8. Save the fine-tuned adapter weights
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nFine-tuned model adapters saved to {output_dir}")

# Optional: Merge LoRA adapters into the base model for a standalone model
# This is useful if you want to deploy the model without needing PEFT installed
print("\nAttempting to merge LoRA adapters with base model...")
try:
    from peft import PeftModel
    
    # Load base model with same settings as training
    base_model_full = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,  # Match training settings
        torch_dtype=torch.float32
    )
    
    # Load and merge adapters
    merged_model = PeftModel.from_pretrained(
        base_model_full, 
        output_dir,
        is_trainable=False  # We only want to merge, not train further
    )
    merged_model = merged_model.merge_and_unload()
    
    # Save merged model
    merged_model_path = "./merged_tinyllama_oss_finetuned"
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"\nMerged fine-tuned model saved to {merged_model_path}")
except Exception as e:
    print("\nWarning: Could not merge LoRA adapters with base model.")
    print(f"Error: {str(e)}")
    print("\nYou can still use the model with adapters from:", output_dir)
    print("See post_finetune_inference.py for both approaches (merged and adapter loading)")
print(f"\nMerged fine-tuned model saved to {merged_model_path}")