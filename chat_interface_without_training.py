import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load the TinyLlama 1.1B Chat model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model safely depending on available hardware. On CPU-only environments
# we avoid forcing bfloat16 (which may not be supported) and pin the model to CPU.
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 on CUDA-capable GPUs if supported
        device_map="auto",
    )
    pipeline_device = 0
else:
    # CPU fallback: avoid setting torch_dtype and force CPU device to keep compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
    )
    pipeline_device = -1

# Ensure padding token is set for consistent generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
    # device is managed by Accelerate's device_map
)

def ask_llama(question):
    # TinyLlama-Chat expects a specific chat format
    messages = [
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = generator(
        prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    # Extract only the model's response part
    response_start_index = outputs[0]['generated_text'].find("<|assistant|>") + len("<|assistant|>")
    return outputs[0]['generated_text'][response_start_index:].strip()

print("--- Initial TinyLlama 1.1B Response ---")
while True:
    question = input("Enter your question (type 'bye' to exit): ")
    if question.lower() == 'bye':
        print("Goodbye!")
        break
    
    response = ask_llama(question)
    print(f"\nQuestion: {question}")
    print(f"Response: {response}\n")