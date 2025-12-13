#!/usr/bin/env python3
"""
LLM Fine-Tuning Script
Uses OpenAI API to fine-tune a model on trading data.
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not installed. Run: pip install openai")


def validate_training_data(file_path: str) -> dict:
    """Validate training data format"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    valid = 0
    invalid = 0
    total_tokens = 0
    
    for line in lines:
        try:
            data = json.loads(line)
            if 'messages' in data:
                valid += 1
                # Rough token estimate
                total_tokens += len(json.dumps(data)) // 4
            else:
                invalid += 1
        except:
            invalid += 1
    
    return {
        'total_examples': len(lines),
        'valid': valid,
        'invalid': invalid,
        'estimated_tokens': total_tokens,
        'estimated_cost': total_tokens / 1_000_000 * 3  # ~$3/1M tokens for gpt-4o-mini
    }


def upload_and_finetune(
    training_file: str,
    model: str = 'gpt-4o-mini-2024-07-18',
    suffix: str = 'gold-line-trader'
):
    """
    Upload training data and start fine-tuning job.
    
    Args:
        training_file: Path to JSONL training file
        model: Base model to fine-tune
        suffix: Suffix for the fine-tuned model name
    """
    if not HAS_OPENAI:
        print("❌ OpenAI not available")
        return None
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return None
    
    client = OpenAI(api_key=api_key)
    
    # Validate data first
    print("🔍 Validating training data...")
    stats = validate_training_data(training_file)
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Valid examples: {stats['valid']}")
    print(f"   Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"   Estimated cost: ${stats['estimated_cost']:.2f}")
    
    if stats['invalid'] > 0:
        print(f"   ⚠️ {stats['invalid']} invalid examples found")
    
    # Upload file
    print(f"\n📤 Uploading {training_file}...")
    with open(training_file, 'rb') as f:
        file_response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    file_id = file_response.id
    print(f"   ✅ Uploaded: {file_id}")
    
    # Create fine-tuning job
    print(f"\n🚀 Starting fine-tuning job...")
    print(f"   Base model: {model}")
    print(f"   Suffix: {suffix}")
    
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=suffix
    )
    
    job_id = job.id
    print(f"   ✅ Job created: {job_id}")
    print(f"   Status: {job.status}")
    
    # Save job info
    job_info = {
        'job_id': job_id,
        'file_id': file_id,
        'model': model,
        'suffix': suffix,
        'status': job.status,
        'created_at': str(job.created_at),
        'training_file': training_file,
        'stats': stats
    }
    
    with open('results/finetune_job.json', 'w') as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\n📋 Job info saved to results/finetune_job.json")
    print(f"\n⏳ Fine-tuning typically takes 10-30 minutes.")
    print(f"   Check status with: python finetune.py --status {job_id}")
    
    return job_id


def check_status(job_id: str):
    """Check status of fine-tuning job"""
    if not HAS_OPENAI:
        print("❌ OpenAI not available")
        return
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"\n📊 Fine-tuning Job Status")
    print(f"{'='*50}")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model: {job.model}")
    
    if job.fine_tuned_model:
        print(f"\n✅ Fine-tuned model ready: {job.fine_tuned_model}")
        print(f"\nTo use in analysis, update config/strategy_params.yaml:")
        print(f"  llm:")
        print(f"    model: \"{job.fine_tuned_model}\"")
    
    if job.status == 'failed':
        print(f"\n❌ Job failed: {job.error}")
    
    # Show events
    events = client.fine_tuning.jobs.list_events(job_id, limit=10)
    if events.data:
        print(f"\n📜 Recent Events:")
        for event in reversed(events.data):
            print(f"   [{event.created_at}] {event.message}")


def test_model(model_name: str, prompt: str = None):
    """Test the fine-tuned model"""
    if not HAS_OPENAI:
        print("❌ OpenAI not available")
        return
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    if prompt is None:
        prompt = """Symbol: BTCUSDT
Signal: LONG
Entry Price: $45000.00
CCI: 85.3
RSI: 62.1
MACD: 150.25
Price vs Gold Line: above
Volatility: 0.85%

Should I take this trade?"""
    
    print(f"\n🧪 Testing fine-tuned model: {model_name}")
    print(f"{'='*50}")
    print(f"Prompt:\n{prompt}")
    print(f"\n{'='*50}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a crypto trading signal filter. Analyze the signal context and decide whether to TAKE or SKIP the trade."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )
    
    print(f"Response:\n{response.choices[0].message.content}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Fine-Tuning for Trading')
    parser.add_argument('--train', type=str, help='Path to training JSONL file')
    parser.add_argument('--status', type=str, help='Check status of job ID')
    parser.add_argument('--test', type=str, help='Test fine-tuned model name')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', help='Base model')
    
    args = parser.parse_args()
    
    if args.train:
        upload_and_finetune(args.train, args.model)
    elif args.status:
        check_status(args.status)
    elif args.test:
        test_model(args.test)
    else:
        # Default: train with aggregated data
        training_file = 'results/training_data_all_assets.jsonl'
        if Path(training_file).exists():
            upload_and_finetune(training_file)
        else:
            print(f"❌ Training file not found: {training_file}")
            print("   Run: python run_multi_asset.py first")
