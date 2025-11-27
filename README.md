# LLM_training_demo — TinyLlama fine-tune & inference demo

This small demo shows how to fine-tune TinyLlama with LoRA adapters and run inference.

Quick setup (PowerShell, Windows)

```powershell
# Create and activate a venv (use Python 3.11+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt

# (Optional) If you need CPU-only PyTorch: use the official CPU index
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade

# Run quick import smoke test
.venv\Scripts\python.exe smoke_test.py

# Run the example scripts (these may download model weights from Hugging Face and require network)
.venv\Scripts\python.exe initial_inference.py
.venv\Scripts\python.exe finetune_model.py
.venv\Scripts\python.exe post_finetune_inference.py
```

Notes:
- For 4-bit training (`load_in_4bit=True`) and bitsandbytes support, prefer running on Linux/GPU. On Windows, installing `bitsandbytes` can be problematic.
- `finetune_model.py` saves adapter weights to `./tinyllama_got_finetuned` and optionally saves a merged model to `./merged_tinyllama_got_finetuned`.
