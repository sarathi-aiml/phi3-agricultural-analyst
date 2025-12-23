"""
Agricultural Insight Analyst - Interactive Demo
================================================

This Gradio app provides an interactive demo of your fine-tuned model.
Deploy this to HuggingFace Spaces for a live LinkedIn demo!

Author: [Your Name]
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path

# =============================================================================
# SAMPLE DATA FOR DEMO
# =============================================================================

DEMO_COUNTIES = {
    "Fresno, CA": {
        "operators": 5847, "avg_farm_acres": 423,
        "irrigation_pct": 89, "revenue_per_acre": 2340,
        "top_crops": "Almonds, Grapes, Cotton"
    },
    "Kern, CA": {
        "operators": 3201, "avg_farm_acres": 512,
        "irrigation_pct": 94, "revenue_per_acre": 2890,
        "top_crops": "Almonds, Pistachios, Citrus"
    },
    "Lancaster, PA": {
        "operators": 5123, "avg_farm_acres": 89,
        "irrigation_pct": 12, "revenue_per_acre": 1450,
        "top_crops": "Corn, Soybeans, Dairy"
    },
    "Sioux, IA": {
        "operators": 2891, "avg_farm_acres": 342,
        "irrigation_pct": 8, "revenue_per_acre": 890,
        "top_crops": "Corn, Soybeans, Hogs"
    },
    "Yakima, WA": {
        "operators": 3456, "avg_farm_acres": 156,
        "irrigation_pct": 87, "revenue_per_acre": 4200,
        "top_crops": "Apples, Hops, Cherries"
    },
    "Imperial, CA": {
        "operators": 1234, "avg_farm_acres": 623,
        "irrigation_pct": 99, "revenue_per_acre": 1890,
        "top_crops": "Alfalfa, Lettuce, Wheat"
    },
    "Monterey, CA": {
        "operators": 1543, "avg_farm_acres": 234,
        "irrigation_pct": 92, "revenue_per_acre": 8900,
        "top_crops": "Lettuce, Strawberries, Broccoli"
    },
    "Deaf Smith, TX": {
        "operators": 987, "avg_farm_acres": 1245,
        "irrigation_pct": 67, "revenue_per_acre": 560,
        "top_crops": "Cattle, Wheat, Corn"
    },
}

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load the fine-tuned model - optimized for DGX Spark."""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "models/phi3-agricultural-analyst/final"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Check if fine-tuned model exists
    if Path(adapter_path).exists():
        print("Loading fine-tuned model with BF16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # BF16 for Blackwell GPU
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Eager attention for Phi-3 compatibility
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model_type = "Fine-tuned Agricultural Analyst (DGX Spark)"
    else:
        print("Fine-tuned model not found, loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # BF16 for Blackwell GPU
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Eager attention for Phi-3 compatibility
        )
        model_type = "Base Phi-3 (not fine-tuned)"

    return model, tokenizer, model_type


# Global model loading
print("Initializing model...")
MODEL, TOKENIZER, MODEL_TYPE = load_model()
print(f"Model loaded: {MODEL_TYPE}")

# =============================================================================
# INFERENCE
# =============================================================================

def analyze(county: str, question: str, temperature: float = 0.7):
    """Run analysis on selected county."""

    if county not in DEMO_COUNTIES:
        return "Please select a valid county."

    if not question.strip():
        return "Please enter a question."

    data = DEMO_COUNTIES[county]
    county_name, state = county.split(", ")

    # Format prompt
    prompt = f"""You are an expert agricultural analyst. Provide a comprehensive and detailed analysis for the following query using the provided data context. Give thorough explanations, specific recommendations, and actionable insights.

### Query
{question}

### Data Context
County: {county_name}, {state}
Operators: {data['operators']:,}
Average Farm Size: {data['avg_farm_acres']} acres
Irrigation Coverage: {data['irrigation_pct']}%
Revenue per Acre: ${data['revenue_per_acre']:,}
Top Crops: {data['top_crops']}

### Detailed Analysis"""

    # Run inference with more tokens for detailed response
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    with torch.no_grad():
        outputs = MODEL.generate(
            inputs["input_ids"],
            max_new_tokens=1000,  # Increased for longer, more detailed responses
            temperature=temperature,
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id,
            use_cache=False  # Fix for DynamicCache compatibility
        )

    response = TOKENIZER.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return response


def update_county_info(county: str):
    """Update county information display."""
    if county not in DEMO_COUNTIES:
        return ""

    data = DEMO_COUNTIES[county]
    return f"""**County Profile**
- Operators: {data['operators']:,}
- Avg Farm Size: {data['avg_farm_acres']} acres
- Irrigation: {data['irrigation_pct']}%
- Revenue/Acre: ${data['revenue_per_acre']:,}
- Top Crops: {data['top_crops']}"""

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(title="Agricultural Insight Analyst") as demo:

        # Header
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>Agricultural Insight Analyst</h1>
            <p>AI-powered agricultural analysis using distilled domain expertise</p>
        </div>
        """)

        # Model info badge
        gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="background: linear-gradient(90deg, #10b981, #059669); color: white; padding: 5px 15px; border-radius: 15px; font-size: 14px;">Model: {MODEL_TYPE}</span>
        </div>
        """)

        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### Select County")
                county_dropdown = gr.Dropdown(
                    choices=list(DEMO_COUNTIES.keys()),
                    value="Fresno, CA",
                    label="County"
                )

                county_info = gr.Markdown(update_county_info("Fresno, CA"))

                gr.Markdown("### Ask Your Question")
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask any question about agricultural analysis for this county...",
                    lines=3
                )

                temperature = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                    label="Temperature (creativity)"
                )

                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                # Sample questions for reference
                gr.Markdown("""
                **Sample Questions:**
                - What are the investment opportunities?
                - Analyze the water sustainability risks
                - What technology should farmers adopt?
                - Compare small vs large farm opportunities
                - What crops should expand here?
                - Assess the labor market conditions
                """)

            # Right column - Output (using Markdown for proper rendering)
            with gr.Column(scale=2):
                gr.Markdown("### Analysis Results")
                output = gr.Markdown(
                    value="*Your analysis will appear here...*",
                    label="Agricultural Insight"
                )

        # About section
        gr.Markdown("""
        ---
        ### About This Project

        This demo showcases a **distilled agricultural analyst model** built by fine-tuning
        Microsoft's Phi-3 Mini on expert-generated agricultural analysis data.

        **Technical Approach:**
        - Base Model: Phi-3 Mini (3.8B parameters)
        - Fine-tuning: LoRA (Low-Rank Adaptation) for efficient training
        - Training Data: Synthetic expert analysis generated via knowledge distillation
        - Task: Domain-specific agricultural insight generation

        **Key Skills Demonstrated:**
        - Knowledge distillation from large to small models
        - Parameter-efficient fine-tuning (PEFT/LoRA)
        - Domain adaptation for specialized tasks
        - Structured output generation
        """)

        # Event handlers
        county_dropdown.change(
            fn=update_county_info,
            inputs=county_dropdown,
            outputs=county_info
        )

        analyze_btn.click(
            fn=analyze,
            inputs=[county_dropdown, question_input, temperature],
            outputs=output
        )

        # Also trigger on Enter key in question input
        question_input.submit(
            fn=analyze,
            inputs=[county_dropdown, question_input, temperature],
            outputs=output
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Disable share for faster local testing
    )
