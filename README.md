# Knowledge Distillation

> **🌾 Agricultural Insight Analyst Knowledge Distillation**: Transferring domain expertise from large language models into a compact, efficient model for agricultural analysis.

[![Model on HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/sarathi-balakrishnan/phi3-agricultural-analyst)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![Agricultural Analyst Demo](assets/demo_screenshot.png)

## 🎯 Project Overview

This project demonstrates **knowledge distillation** - the process of transferring capabilities from a large language model (teacher) to a smaller, more efficient model (student) for domain-specific tasks.

### What I Built

| Component | Description |
|-----------|-------------|
| **Training Pipeline** | Synthetic data generation + LoRA fine-tuning |
| **Domain Model** | Phi-3 Mini adapted for agricultural analysis |
| **Benchmark Suite** | Automated evaluation comparing base vs fine-tuned |
| **Live Demo** | Interactive Gradio app on HuggingFace Spaces |

### Key Results

| Metric | Base Phi-3 | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Content Coverage | 45% | 78% | **+73%** |
| Structure Quality | 35% | 82% | **+134%** |
| Domain Accuracy | 52% | 85% | **+63%** |

## 🧠 Technical Approach

### 1. Knowledge Distillation Strategy

Instead of trying to "memorize" 10M+ rows of agricultural data, I:

1. **Generated expert reasoning traces** using a large model (Claude/GPT-4)
2. **Created structured training examples** that teach analytical patterns
3. **Fine-tuned a small model** to replicate the expert reasoning style

```
┌─────────────────────────────────────────────────────────────┐
│                  DISTILLATION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐      ┌──────────────┐     ┌──────────┐  │
│   │ Agricultural │ ───▶ │   Expert     │ ──▶ │ Training │  │
│   │ Data (10M+)  │      │   Analysis   │     │   Data   │  │
│   └──────────────┘      │  (Teacher)   │     │  (200+)  │  │
│                         └──────────────┘     └──────────┘  │
│                                                     │       │
│                                                     ▼       │
│                         ┌──────────────────────────────┐   │
│                         │   LoRA Fine-Tuning           │   │
│                         │   (Phi-3 Mini + Adapters)    │   │
│                         └──────────────────────────────┘   │
│                                                     │       │
│                                                     ▼       │
│                         ┌──────────────────────────────┐   │
│                         │   Domain-Adapted Model       │   │
│                         │   (Agricultural Analyst)     │   │
│                         └──────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Parameter-Efficient Fine-Tuning (LoRA)

Rather than fine-tuning all 3.8B parameters, I used **LoRA (Low-Rank Adaptation)**:

```python
LoraConfig(
    r=16,                    # Rank of update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", 
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05
)
```

**Benefits:**
- Trainable parameters: **0.5%** of total (vs 100% for full fine-tuning)
- Training time: **~30 minutes** on single GPU
- Memory: **<16GB VRAM** required
- Easy to merge, share, and version

### 3. Structured Output Training

The model learns to generate consistently structured analysis:

```
### Query
Analyze investment potential for Fresno County agriculture.

### Analysis
**Summary**: Fresno County represents a high-value intensive 
agricultural region with 5,847 operators...

**Key Insights**:
- High-value intensive agriculture with 89% irrigation coverage
- Revenue per acre ($2,340) indicates premium crop focus
- Tree nut production requires long-term capital commitment

**Opportunities**:
- Precision irrigation technology investment
- Sensor-based monitoring for crop protection

**Risk Factors**:
- Drought conditions affecting water availability
- Labor cost pressures

**Confidence**: high
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes gradio
```

### Run the Pipeline

```bash
# 1. Generate training data
python 01_generate_training_data.py

# 2. Fine-tune the model
python 02_finetune_lora.py

# 3. Run benchmarks
python 03_benchmark.py

# 4. Launch demo
python 04_gradio_demo.py

# 5. Upload to HuggingFace
python 05_upload_to_hub.py --setup  # First time
python 05_upload_to_hub.py          # After configuration
```

## 📁 Project Structure

```
agricultural-analyst/
├── 01_generate_training_data.py   # Synthetic data generation
├── 02_finetune_lora.py           # LoRA fine-tuning script
├── 03_benchmark.py               # Model evaluation
├── 04_gradio_demo.py             # Interactive demo
├── 05_upload_to_hub.py           # HuggingFace deployment
├── data/
│   ├── train.json                # Training examples
│   └── eval.json                 # Evaluation examples
├── models/
│   └── phi3-agricultural-analyst/
│       └── final/                # Fine-tuned model weights
├── benchmark_results.json        # Evaluation metrics
└── README.md
```

## 📊 Evaluation Methodology

The benchmark suite evaluates models on:

1. **Content Coverage** - Does the response include expected domain elements?
2. **Structure Quality** - Does it follow the analyst output format?
3. **Domain Accuracy** - Are agricultural insights contextually appropriate?

```python
# Example evaluation criteria
TEST_CASES = [
    {
        "query": "Analyze investment potential of Fresno County",
        "expected_elements": [
            "irrigation", 
            "almonds OR tree nuts",
            "water", 
            "risk", 
            "opportunity"
        ]
    }
]
```

## 🔧 Configuration Options

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA scaling factor |
| `learning_rate` | 2e-4 | Training learning rate |
| `num_epochs` | 3 | Training epochs |
| `batch_size` | 2 | Per-device batch size |

### Supported Models

The pipeline can be adapted for:
- `microsoft/Phi-3-mini-4k-instruct` (default)
- `microsoft/Phi-3-mini-128k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`

## 🎓 Skills Demonstrated

This project showcases:

- **Knowledge Distillation** - Transferring capabilities from large to small models
- **Parameter-Efficient Fine-Tuning** - LoRA/PEFT techniques
- **Domain Adaptation** - Specializing general models for specific tasks
- **Structured Output Generation** - Training models for consistent formats
- **Model Evaluation** - Designing benchmarks and metrics
- **MLOps** - Model deployment to HuggingFace Hub

## 📈 Future Improvements

- [ ] Add RAG integration for real-time data retrieval
- [ ] Expand to more agricultural regions
- [ ] Implement A/B testing framework
- [ ] Add model quantization for edge deployment
- [ ] Create evaluation dataset with human annotations

## 📝 License

MIT License - feel free to use, modify, and distribute.

## 👤 Author

**[Sarathi Balakrishnan]**
- LinkedIn: https://www.linkedin.com/in/sarathib/

---

*Built with 🌾 and ☕ | Feedback welcome via issues or pull requests*
