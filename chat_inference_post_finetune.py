import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Path to your fine-tuned model (use the merged one if you ran that step)
finetuned_model_path = "C:/coding/LLM_training_demo/merged_tinyllama_oss_finetuned" # Or "./tinyllama_oss_finetuned" if not merged

# 1. Load the fine-tuned model and tokenizer
tokenizer_finetuned = AutoTokenizer.from_pretrained(finetuned_model_path)
if tokenizer_finetuned.pad_token is None:
    tokenizer_finetuned.pad_token = tokenizer_finetuned.eos_token

# If you merged the model:
model_finetuned = AutoModelForCausalLM.from_pretrained(
    finetuned_model_path,
    torch_dtype=torch.bfloat16, # Match dtype used for merging/training
    device_map="auto"
)

# If you did NOT merge, load the base model and then apply adapters
# model_base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#                                                   load_in_4bit=True, # Need to load in 4-bit again
#                                                   torch_dtype=torch.bfloat16,
#                                                   device_map="auto")
# model_finetuned = PeftModel.from_pretrained(model_base, finetuned_model_path)


# Create a text generation pipeline for the fine-tuned model
generator_finetuned = pipeline(
    "text-generation",
    model=model_finetuned,
    tokenizer=tokenizer_finetuned,
    device_map="auto"
)

def ask_finetuned_llama(question):
    # Use the same chat template as during training
    messages = [
        {"role": "user", "content": question},
    ]
    prompt = tokenizer_finetuned.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = generator_finetuned(
        prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer_finetuned.eos_token_id
    )
    response_start_index = outputs[0]['generated_text'].find("<|assistant|>") + len("<|assistant|>")
    return outputs[0]['generated_text'][response_start_index:].strip()

print("\n--- Fine-tuned TinyLlama 1.1B Response ---")
while True:
    question = input("Enter your question (type 'bye' to exit): ")
    if question.lower() == 'bye':
        print("Goodbye!")
        break
    
    response = ask_finetuned_llama(question)
    print(f"\nQuestion: {question}")
    print(f"Fine-tuned Response: {response}\n") 