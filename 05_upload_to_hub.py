"""
Agricultural Insight Distillation - Step 5: Upload to HuggingFace Hub
=====================================================================

This script uploads your fine-tuned model to HuggingFace Hub.

Author: Sarathi
"""

import json
from pathlib import Path
from datetime import datetime

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_USERNAME = "sarathi-balakrishnan"
MODEL_NAME = "phi3-agricultural-analyst"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

LOCAL_MODEL_PATH = "models/phi3-agricultural-analyst/final"

# =============================================================================
# MODEL CARD CONTENT
# =============================================================================

MODEL_CARD = """---
license: mit
language:
- en
tags:
- agriculture
- fine-tuned
- phi-3
- lora
- peft
- domain-specific
library_name: peft
base_model: microsoft/Phi-3-mini-4k-instruct
datasets:
- custom
pipeline_tag: text-generation
model-index:
- name: phi3-agricultural-analyst
  results:
  - task:
      type: text-generation
      name: Agricultural Analysis
    metrics:
    - type: content_coverage
      value: 63.3
      name: Content Coverage (%)
    - type: structure_quality
      value: 83.3
      name: Structure Quality (%)
---

# Phi-3 Agricultural Analyst

A domain-specific language model fine-tuned for agricultural analysis tasks. This model provides structured insights on farm operations, investment opportunities, risk assessment, and regional agricultural profiling.

## Project Scope

This release focuses on **8 major US agricultural counties** as a proof-of-concept:
- Fresno, CA (Tree nuts, Grapes)
- Kern, CA (Almonds, Pistachios)
- Lancaster, PA (Dairy, Corn)
- Sioux, IA (Corn, Soybeans)
- Yakima, WA (Apples, Hops)
- Imperial, CA (Alfalfa, Lettuce)
- Monterey, CA (Strawberries, Lettuce)
- Deaf Smith, TX (Cattle, Wheat)

**Upcoming releases** will include:
- Full US county coverage (3,000+ counties)
- Specialized models for crop yield prediction
- Harvest timing optimization
- Soil health analysis
- Weather impact assessment

## Performance Benchmarks

Evaluated against base Phi-3 Mini on agricultural analysis tasks:

### Content Coverage
```
Base Model    |████████████████████                    | 50.0%
Fine-Tuned    |█████████████████████████               | 63.3%  (+26.7%)
```

### Structure Quality
```
Base Model    |██████████████████████                  | 56.7%
Fine-Tuned    |█████████████████████████████████       | 83.3%  (+47.1%)
```

### Inference Speed
```
Base Model    |████████████████████████████████████████| 101.9s
Fine-Tuned    |████████████████████████                | 62.4s  (38.8% faster)
```

### Summary Table

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Content Coverage | 50.0% | 63.3% | +26.7% |
| Structure Quality | 56.7% | 83.3% | +47.1% |
| Avg Inference Time | 101.9s | 62.4s | 38.8% faster |

### Training Loss Curve
```
Epoch 1: ████████████████████████████████████████ Loss: 1.605 → 0.115
Epoch 2: ████████████                             Loss: 0.115 → 0.020
Epoch 3: ████                                     Loss: 0.020 → 0.019
```

The fine-tuned model shows significant improvement in generating domain-relevant content and producing well-structured analytical outputs.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | microsoft/Phi-3-mini-4k-instruct |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training Examples | 900 |
| Validation Examples | 100 |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| Precision | BF16 |
| Training Time | ~51 minutes |
| Final Eval Loss | 0.0186 |
| Token Accuracy | 99.3% |

Hardware: NVIDIA DGX Spark (Blackwell GPU, 128GB RAM)

## Use Cases

- Farm investment analysis
- Regional agricultural profiling
- Risk factor identification
- Technology adoption recommendations
- Irrigation and water sustainability assessment
- Crop expansion planning

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/phi3-agricultural-analyst")

# Create prompt
prompt = \"\"\"You are an expert agricultural analyst. Analyze the following query using the provided data context.

### Query
What are the key investment opportunities in Fresno County agriculture?

### Data Context
County: Fresno, CA
Operators: 5,847
Average Farm Size: 423 acres
Irrigation Coverage: 89%
Revenue per Acre: $2,340

### Analysis\"\"\"

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=500,
    temperature=0.7,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Output Format

The model generates structured analysis with:
- Summary of the agricultural region
- Key insights based on data patterns
- Investment/growth opportunities
- Risk factors and constraints
- Confidence level

## Limitations

- Currently covers 8 US counties (expansion planned)
- Training data is synthetically generated
- Should not replace professional agricultural consulting
- Performance may vary for regions outside training distribution

## Technical Notes

- Uses eager attention for Phi-3 compatibility
- Optimized for BF16 inference on modern GPUs
- LoRA adapters can be merged for deployment: `model.merge_and_unload()`

## License

MIT License - Free for commercial and research use.

## Contact

For questions about this model or collaboration on agricultural AI:
- HuggingFace: [@YOUR_USERNAME](https://huggingface.co/YOUR_USERNAME)
"""

# =============================================================================
# UPLOAD FUNCTIONS
# =============================================================================

def create_model_card():
    """Create README.md for the model."""
    readme_path = Path(LOCAL_MODEL_PATH) / "README.md"

    # Replace placeholder with actual username
    card_content = MODEL_CARD.replace("YOUR_USERNAME", HF_USERNAME)

    readme_path.write_text(card_content)
    print(f"Created model card: {readme_path}")
    return readme_path


def upload_to_hub():
    """Upload model to HuggingFace Hub."""

    print("=" * 60)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("=" * 60)

    api = HfApi()

    # Check if model exists locally
    model_path = Path(LOCAL_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {LOCAL_MODEL_PATH}. "
            "Run 02_finetune_lora.py first."
        )

    # Create model card
    print("\n[1/3] Creating model card...")
    create_model_card()

    # Create repo (if doesn't exist)
    print(f"\n[2/3] Creating/accessing repo: {REPO_ID}")
    try:
        create_repo(REPO_ID, exist_ok=True, repo_type="model")
        print(f"      Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"      Note: {e}")

    # Upload
    print(f"\n[3/3] Uploading model files...")
    upload_folder(
        folder_path=str(model_path),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Initial release - Agricultural analyst model for 8 US counties"
    )

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nModel is live at: https://huggingface.co/{REPO_ID}")


def setup_instructions():
    """Print setup instructions for HuggingFace Hub."""

    print("""
================================================================================
HUGGINGFACE HUB SETUP
================================================================================

1. Login to HuggingFace:

   huggingface-cli login

   (Enter your access token when prompted)

2. Update HF_USERNAME in this script with your username

3. Run:

   python 05_upload_to_hub.py

================================================================================
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_instructions()
    else:
        if HF_USERNAME == "YOUR_USERNAME":
            print("Please update HF_USERNAME in this script first!")
            print("Run with --setup for instructions.")
        else:
            upload_to_hub()
