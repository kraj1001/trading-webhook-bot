#!/usr/bin/env python3
"""
Analyze Results
Run LLM analysis on backtest trades to discover patterns and get improvement suggestions.
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import os

import sys
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from llm.analyzer import LLMAnalyzer


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_latest_trades_file(results_dir: str = 'results') -> str:
    """Find the most recent trades file"""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None
    
    trades_files = list(results_path.glob('trades_*.json'))
    if not trades_files:
        return None
    
    # Sort by modification time
    latest = max(trades_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    parser = argparse.ArgumentParser(description='Analyze backtest results with LLM')
    parser.add_argument('--trades', type=str, help='Path to trades JSON file')
    parser.add_argument('--config', type=str, default='config/strategy_params.yaml', help='Config file path')
    parser.add_argument('--output', type=str, help='Output file for analysis')
    parser.add_argument('--generate-training', action='store_true', help='Generate LLM training data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 LLM Analysis of Backtest Results")
    print("=" * 60)
    
    # Find trades file
    trades_file = args.trades or find_latest_trades_file()
    
    if not trades_file or not Path(trades_file).exists():
        print("❌ No trades file found. Run a backtest first:")
        print("   python run_backtest.py --symbol BTCUSDT --days 90")
        return
    
    print(f"📄 Analyzing: {trades_file}")
    
    # Load trades summary
    with open(trades_file, 'r') as f:
        trades = json.load(f)
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    
    print(f"   Total trades: {len(trades)}")
    print(f"   Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"   Win rate: {len(wins)/len(trades)*100:.1f}%")
    
    # Load config
    config = load_config(args.config)
    
    # Check for API keys
    llm_provider = config.get('llm', {}).get('provider', 'openai')
    api_key = os.getenv('OPENAI_API_KEY') if llm_provider == 'openai' else os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        print(f"\n⚠️  No {llm_provider.upper()}_API_KEY found in environment")
        print("   Analysis will run in mock mode.")
        print(f"   To enable real analysis, add your API key to .env file")
    
    # Initialize analyzer
    print("\n🔍 Running LLM analysis...")
    analyzer = LLMAnalyzer(config)
    
    # Run analysis
    analysis = analyzer.analyze_trades(trades_file)
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    print(analysis)
    print("=" * 60)
    
    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('results') / f'analysis_{timestamp}.md'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# LLM Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Trades Analyzed:** {trades_file}\n")
        f.write(f"**Total Trades:** {len(trades)} | Win Rate: {len(wins)/len(trades)*100:.1f}%\n\n")
        f.write("---\n\n")
        f.write(analysis)
    
    print(f"\n💾 Analysis saved to: {output_path}")
    
    # Generate training data if requested
    if args.generate_training:
        print("\n📝 Generating LLM training data...")
        training_file = analyzer.generate_training_data(
            trades_file,
            output_file=str(Path('results') / 'training_data.jsonl')
        )
        print(f"✅ Training data saved to: {training_file}")
        print("\n   To fine-tune a model, use this data with:")
        print("   - OpenAI: openai api fine_tunes.create -t training_data.jsonl -m gpt-3.5-turbo")
        print("   - Or use with any JSONL-compatible fine-tuning framework")


if __name__ == '__main__':
    main()
