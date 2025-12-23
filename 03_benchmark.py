"""
Agricultural Insight Distillation - Step 3: Benchmark & Compare
===============================================================

This script compares the base Phi-3 model against your fine-tuned version
on agricultural analysis tasks. This creates the EVIDENCE you need for
LinkedIn - showing measurable improvement.

Author: [Your Name]
"""

import json
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import pandas as pd

# =============================================================================
# TEST CASES - Designed to show improvement
# =============================================================================

TEST_CASES = [
    {
        "id": "test_01",
        "query": "Analyze the agricultural investment potential of Fresno County, California.",
        "context": {
            "county": "Fresno", "state": "CA",
            "operators": 5847, "avg_farm_acres": 423,
            "irrigation_pct": 89, "revenue_per_acre": 2340
        },
        "expected_elements": [
            "irrigation", "almonds OR tree nuts OR permanent crops",
            "revenue", "water", "risk", "opportunity"
        ]
    },
    {
        "id": "test_02", 
        "query": "What are the key risk factors for farming in Imperial County, California?",
        "context": {
            "county": "Imperial", "state": "CA",
            "operators": 1234, "avg_farm_acres": 623,
            "irrigation_pct": 99, "revenue_per_acre": 1890
        },
        "expected_elements": [
            "water", "Colorado River OR allocation",
            "heat OR temperature", "risk"
        ]
    },
    {
        "id": "test_03",
        "query": "Compare small farm opportunities in Lancaster County, Pennsylvania.",
        "context": {
            "county": "Lancaster", "state": "PA",
            "operators": 5123, "avg_farm_acres": 89,
            "irrigation_pct": 12, "revenue_per_acre": 1450
        },
        "expected_elements": [
            "small farm OR family farm", "dairy",
            "direct market OR farmers market", "urban"
        ]
    },
    {
        "id": "test_04",
        "query": "Assess technology investment priorities for Kern County agriculture.",
        "context": {
            "county": "Kern", "state": "CA",
            "operators": 3201, "avg_farm_acres": 512,
            "irrigation_pct": 94, "revenue_per_acre": 2890
        },
        "expected_elements": [
            "precision OR technology OR automation",
            "irrigation OR water efficiency",
            "almond OR pistachio", "investment"
        ]
    },
    {
        "id": "test_05",
        "query": "Evaluate sustainability challenges for Deaf Smith County, Texas ranching.",
        "context": {
            "county": "Deaf Smith", "state": "TX",
            "operators": 987, "avg_farm_acres": 1245,
            "irrigation_pct": 67, "revenue_per_acre": 560
        },
        "expected_elements": [
            "Ogallala OR aquifer OR groundwater",
            "cattle OR ranching", "sustainability OR depletion",
            "large scale"
        ]
    },
]

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def format_prompt(test_case: dict) -> str:
    """Format test case into model prompt."""
    ctx = test_case["context"]
    return f"""You are an expert agricultural analyst. Analyze the following query using the provided data context.

### Query
{test_case['query']}

### Data Context
County: {ctx['county']}, {ctx['state']}
Operators: {ctx['operators']:,}
Average Farm Size: {ctx['avg_farm_acres']} acres
Irrigation Coverage: {ctx['irrigation_pct']}%
Revenue per Acre: ${ctx['revenue_per_acre']:,}

### Analysis"""


def score_response(response: str, expected_elements: List[str]) -> Dict:
    """Score response based on expected content elements."""
    response_lower = response.lower()
    
    scores = {}
    for element in expected_elements:
        # Handle OR conditions
        if " OR " in element:
            options = [opt.strip().lower() for opt in element.split(" OR ")]
            found = any(opt in response_lower for opt in options)
        else:
            found = element.lower() in response_lower
        scores[element] = 1.0 if found else 0.0
    
    return {
        "element_scores": scores,
        "coverage": sum(scores.values()) / len(scores),
        "elements_found": sum(scores.values()),
        "elements_total": len(scores)
    }


def evaluate_response_quality(response: str) -> Dict:
    """Evaluate structural quality of response."""
    
    # Check for structured output elements
    has_summary = "summary" in response.lower() or response.strip().startswith("**")
    has_insights = "insight" in response.lower() or "key" in response.lower()
    has_risks = "risk" in response.lower()
    has_recommendations = any(word in response.lower() for word in ["recommend", "opportunity", "suggest", "consider"])
    has_confidence = "confidence" in response.lower()
    
    # Check for bullet points / structure
    has_bullets = "-" in response or "•" in response or "*" in response
    
    # Length check (good responses should be substantive)
    word_count = len(response.split())
    appropriate_length = 100 < word_count < 500
    
    structure_score = sum([
        has_summary, has_insights, has_risks, 
        has_recommendations, has_bullets, appropriate_length
    ]) / 6.0
    
    return {
        "has_summary": has_summary,
        "has_insights": has_insights,
        "has_risks": has_risks,
        "has_recommendations": has_recommendations,
        "has_confidence": has_confidence,
        "has_structure": has_bullets,
        "word_count": word_count,
        "appropriate_length": appropriate_length,
        "structure_score": structure_score
    }


def run_inference(model, tokenizer, prompt: str, max_tokens: int = 300) -> tuple:
    """Run inference and return response with timing."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # Fix for DynamicCache compatibility
        )
    inference_time = time.time() - start_time
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return response, inference_time


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def load_base_model():
    """Load the base Phi-3 model - optimized for DGX Spark."""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # BF16 for Blackwell GPU
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Eager attention for Phi-3 compatibility
    )

    return model, tokenizer


def load_finetuned_model():
    """Load the fine-tuned model with LoRA weights - optimized for DGX Spark."""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "models/phi3-agricultural-analyst/final"

    print(f"Loading fine-tuned model from: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load base model with BF16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # BF16 for Blackwell GPU
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Eager attention for Phi-3 compatibility
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer


def run_benchmark():
    """Run full benchmark comparison."""
    
    print("=" * 70)
    print("AGRICULTURAL ANALYST MODEL BENCHMARK")
    print("Base Phi-3 vs Fine-Tuned (Distilled) Version")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test cases: {len(TEST_CASES)}")
    
    results = {
        "base": [],
        "finetuned": []
    }
    
    # Load models
    print("\n" + "-" * 70)
    print("LOADING MODELS")
    print("-" * 70)
    
    base_model, base_tokenizer = load_base_model()
    
    # Check if fine-tuned model exists
    finetuned_path = Path("models/phi3-agricultural-analyst/final")
    if finetuned_path.exists():
        ft_model, ft_tokenizer = load_finetuned_model()
        has_finetuned = True
    else:
        print("\n⚠️  Fine-tuned model not found. Running base model only.")
        print("   Run 02_finetune_lora.py first to train the model.")
        has_finetuned = False
    
    # Run evaluations
    print("\n" + "-" * 70)
    print("RUNNING EVALUATIONS")
    print("-" * 70)
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {test_case['id']}")
        print(f"Query: {test_case['query'][:60]}...")
        
        prompt = format_prompt(test_case)
        
        # Base model
        print("  → Running base model...", end=" ")
        base_response, base_time = run_inference(base_model, base_tokenizer, prompt)
        base_content_score = score_response(base_response, test_case["expected_elements"])
        base_quality_score = evaluate_response_quality(base_response)
        print(f"Done ({base_time:.2f}s)")
        
        results["base"].append({
            "test_id": test_case["id"],
            "response": base_response,
            "inference_time": base_time,
            "content_score": base_content_score,
            "quality_score": base_quality_score
        })
        
        # Fine-tuned model
        if has_finetuned:
            print("  → Running fine-tuned model...", end=" ")
            ft_response, ft_time = run_inference(ft_model, ft_tokenizer, prompt)
            ft_content_score = score_response(ft_response, test_case["expected_elements"])
            ft_quality_score = evaluate_response_quality(ft_response)
            print(f"Done ({ft_time:.2f}s)")
            
            results["finetuned"].append({
                "test_id": test_case["id"],
                "response": ft_response,
                "inference_time": ft_time,
                "content_score": ft_content_score,
                "quality_score": ft_quality_score
            })
    
    # Calculate aggregate scores
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    # Base model stats
    base_coverage = sum(r["content_score"]["coverage"] for r in results["base"]) / len(results["base"])
    base_structure = sum(r["quality_score"]["structure_score"] for r in results["base"]) / len(results["base"])
    base_avg_time = sum(r["inference_time"] for r in results["base"]) / len(results["base"])
    
    print(f"\n{'Metric':<30} {'Base Model':<15} {'Fine-Tuned':<15} {'Improvement':<15}")
    print("-" * 75)
    
    print(f"{'Content Coverage':<30} {base_coverage*100:>12.1f}%", end="")
    
    if has_finetuned:
        ft_coverage = sum(r["content_score"]["coverage"] for r in results["finetuned"]) / len(results["finetuned"])
        ft_structure = sum(r["quality_score"]["structure_score"] for r in results["finetuned"]) / len(results["finetuned"])
        ft_avg_time = sum(r["inference_time"] for r in results["finetuned"]) / len(results["finetuned"])
        
        coverage_improvement = ((ft_coverage - base_coverage) / base_coverage) * 100 if base_coverage > 0 else 0
        structure_improvement = ((ft_structure - base_structure) / base_structure) * 100 if base_structure > 0 else 0
        
        print(f"   {ft_coverage*100:>12.1f}%   {coverage_improvement:>+12.1f}%")
        print(f"{'Structure Quality':<30} {base_structure*100:>12.1f}%   {ft_structure*100:>12.1f}%   {structure_improvement:>+12.1f}%")
        print(f"{'Avg Inference Time':<30} {base_avg_time:>12.2f}s   {ft_avg_time:>12.2f}s")
    else:
        print("       N/A              N/A")
        print(f"{'Structure Quality':<30} {base_structure*100:>12.1f}%       N/A              N/A")
        print(f"{'Avg Inference Time':<30} {base_avg_time:>12.2f}s       N/A")
    
    # Save detailed results
    output_path = Path("benchmark_results.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": len(TEST_CASES),
        "summary": {
            "base_model": {
                "content_coverage": base_coverage,
                "structure_quality": base_structure,
                "avg_inference_time": base_avg_time
            }
        },
        "detailed_results": results
    }
    
    if has_finetuned:
        output_data["summary"]["finetuned_model"] = {
            "content_coverage": ft_coverage,
            "structure_quality": ft_structure,
            "avg_inference_time": ft_avg_time
        }
        output_data["summary"]["improvements"] = {
            "content_coverage_pct": coverage_improvement,
            "structure_quality_pct": structure_improvement
        }
    
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nDetailed results saved to: {output_path}")
    
    # Print sample comparison
    print("\n" + "=" * 70)
    print("SAMPLE RESPONSE COMPARISON (Test 1)")
    print("=" * 70)
    
    print("\n[BASE MODEL RESPONSE]")
    print("-" * 40)
    print(results["base"][0]["response"][:800])
    
    if has_finetuned:
        print("\n[FINE-TUNED MODEL RESPONSE]")
        print("-" * 40)
        print(results["finetuned"][0]["response"][:800])
    
    return output_data


if __name__ == "__main__":
    run_benchmark()
