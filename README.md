# CPU-Fine-Tuning-TinyLlama-with-LoRA
This script does the fine-tuning a small Language Model (TinyLlama-1.1B-Chat-v1.0) using the LoRA technique.
The model is loaded and LoRA config is set for r=8 and target_modules are se for the 4 attention layers. 
The data is converted to chat template and tokenized.
Model is trained, i.e the final weights are calculated and store.
The base model is loaded and merged with the new weights.
