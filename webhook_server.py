#!/usr/bin/env python3
"""
TradingView Webhook Server
Receives alerts from TradingView and executes paper/live trades.

Setup:
1. Deploy this to Railway/Render/ngrok for free
2. Create TradingView alert with webhook URL
3. Trades execute automatically when alerts trigger
"""

import os
import json
import hmac
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('webhook_trades.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', 'your-secret-key-here')
PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'

# Trade state
STATE_FILE = Path('webhook_state.json')


def load_state():
    """Load trading state"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'capital': 10000.0,
        'position': None,
        'trades': []
    }


def save_state(state):
    """Save trading state"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def verify_signature(payload, signature):
    """Verify webhook signature (optional security)"""
    if not WEBHOOK_SECRET or WEBHOOK_SECRET == 'your-secret-key-here':
        return True  # Skip verification if no secret set
    
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def execute_paper_trade(action, symbol, price, quantity=None):
    """Execute paper trade"""
    state = load_state()
    
    position = state.get('position')
    capital = state['capital']
    
    if quantity is None:
        # Use 5% of capital
        trade_value = capital * 0.05
        quantity = trade_value / price
    
    result = {'success': False, 'message': ''}
    
    if action.upper() == 'BUY':
        # Close short if exists
        if position and position['direction'] == 'SHORT':
            pnl = (position['entry_price'] - price) * position['quantity']
            state['capital'] += pnl
            state['trades'].append({
                'type': 'CLOSE_SHORT',
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'pnl': pnl,
                'time': datetime.now().isoformat()
            })
            logger.info(f"Closed SHORT @ ${price:,.2f} | PnL: ${pnl:+,.2f}")
        
        # Open long
        state['position'] = {
            'direction': 'LONG',
            'symbol': symbol,
            'entry_price': price,
            'quantity': quantity,
            'entry_time': datetime.now().isoformat()
        }
        result = {'success': True, 'message': f'Opened LONG {symbol} @ ${price:,.2f}'}
        logger.info(f"📈 Opened LONG @ ${price:,.2f} | Qty: {quantity:.6f}")
    
    elif action.upper() == 'SELL':
        # Close long if exists
        if position and position['direction'] == 'LONG':
            pnl = (price - position['entry_price']) * position['quantity']
            state['capital'] += pnl
            state['trades'].append({
                'type': 'CLOSE_LONG',
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'pnl': pnl,
                'time': datetime.now().isoformat()
            })
            logger.info(f"Closed LONG @ ${price:,.2f} | PnL: ${pnl:+,.2f}")
        
        # Open short
        state['position'] = {
            'direction': 'SHORT',
            'symbol': symbol,
            'entry_price': price,
            'quantity': quantity,
            'entry_time': datetime.now().isoformat()
        }
        result = {'success': True, 'message': f'Opened SHORT {symbol} @ ${price:,.2f}'}
        logger.info(f"📉 Opened SHORT @ ${price:,.2f} | Qty: {quantity:.6f}")
    
    elif action.upper() == 'CLOSE':
        if position:
            if position['direction'] == 'LONG':
                pnl = (price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - price) * position['quantity']
            
            state['capital'] += pnl
            state['trades'].append({
                'type': f"CLOSE_{position['direction']}",
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'pnl': pnl,
                'time': datetime.now().isoformat()
            })
            state['position'] = None
            result = {'success': True, 'message': f'Closed position @ ${price:,.2f} | PnL: ${pnl:+,.2f}'}
            logger.info(f"Closed position @ ${price:,.2f} | PnL: ${pnl:+,.2f}")
    
    save_state(state)
    return result


@app.route('/')
def home():
    """Home page with status"""
    state = load_state()
    return jsonify({
        'status': 'running',
        'mode': 'paper' if PAPER_MODE else 'live',
        'capital': state['capital'],
        'position': state['position'],
        'total_trades': len(state['trades'])
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Receive TradingView webhook alerts.
    
    Expected JSON payload:
    {
        "action": "BUY" or "SELL" or "CLOSE",
        "symbol": "BTCUSDT",
        "price": 90000.00,
        "message": "MACD Bullish Crossover"
    }
    
    Or simple text: "BUY BTCUSDT 90000"
    """
    try:
        # Get payload
        if request.is_json:
            data = request.json
        else:
            # Parse simple text format: "BUY BTCUSDT 90000"
            text = request.data.decode('utf-8').strip()
            parts = text.split()
            data = {
                'action': parts[0] if len(parts) > 0 else '',
                'symbol': parts[1] if len(parts) > 1 else 'BTCUSDT',
                'price': float(parts[2]) if len(parts) > 2 else 0
            }
        
        action = data.get('action', '').upper()
        symbol = data.get('symbol', 'BTCUSDT')
        price = float(data.get('price', 0))
        message = data.get('message', '')
        
        logger.info(f"📨 Webhook received: {action} {symbol} @ ${price:,.2f} | {message}")
        
        if not action or not price:
            return jsonify({'error': 'Missing action or price'}), 400
        
        if PAPER_MODE:
            result = execute_paper_trade(action, symbol, price)
        else:
            # TODO: Implement live trading via Bybit API
            result = {'success': False, 'message': 'Live trading not implemented'}
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get trading statistics"""
    state = load_state()
    trades = state['trades']
    
    if not trades:
        return jsonify({'message': 'No trades yet', 'capital': state['capital']})
    
    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    return jsonify({
        'capital': state['capital'],
        'initial_capital': 10000.0,
        'total_pnl': total_pnl,
        'roi_pct': ((state['capital'] / 10000.0) - 1) * 100,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'current_position': state['position']
    })


@app.route('/trades')
def get_trades():
    """Get all trades"""
    state = load_state()
    return jsonify(state['trades'])


@app.route('/reset', methods=['POST'])
def reset():
    """Reset paper trading state"""
    state = {
        'capital': 10000.0,
        'position': None,
        'trades': []
    }
    save_state(state)
    logger.info("🔄 Paper trading state reset")
    return jsonify({'success': True, 'message': 'State reset'})


def print_tradingview_setup():
    """Print TradingView alert setup instructions"""
    print("\n" + "=" * 70)
    print("📺 TRADINGVIEW ALERT SETUP")
    print("=" * 70)
    print("""
1. Go to TradingView and open BTCUSDT chart
2. Add MACD indicator (Settings: 8, 17, 9)
3. Right-click chart → Add Alert
4. Set Condition: MACD crosses above Signal Line
5. Alert Action: Webhook URL
6. Webhook URL: YOUR_SERVER_URL/webhook
7. Message (JSON format):

   {"action": "BUY", "symbol": "BTCUSDT", "price": {{close}}, "message": "MACD Bullish"}

8. Create another alert for SELL:
   Condition: MACD crosses below Signal Line
   Message: {"action": "SELL", "symbol": "BTCUSDT", "price": {{close}}, "message": "MACD Bearish"}

Or simple format (works too):
   BUY BTCUSDT {{close}}
   SELL BTCUSDT {{close}}
""")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TradingView Webhook Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--setup', action='store_true', help='Show TradingView setup')
    
    args = parser.parse_args()
    
    if args.setup:
        print_tradingview_setup()
    else:
        print("\n" + "=" * 70)
        print("🚀 TradingView Webhook Server")
        print("=" * 70)
        print(f"Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
        print(f"Webhook URL: http://YOUR_IP:{args.port}/webhook")
        print(f"Stats: http://YOUR_IP:{args.port}/stats")
        print("=" * 70)
        print("\nFor TradingView setup, run: python webhook_server.py --setup\n")
        
        app.run(host=args.host, port=args.port, debug=False)
