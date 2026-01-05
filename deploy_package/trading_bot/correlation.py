"""
Multi-Asset Correlation Module
Prevents over-exposure to correlated assets
"""
import logging
from typing import Dict, List, Tuple
import ccxt

logger = logging.getLogger(__name__)

# Pre-computed correlation matrix for common crypto pairs
# These are approximate correlations based on historical data
# Values range from -1 (inverse) to 1 (perfect correlation)
CORRELATION_MATRIX = {
    "DOGEUSDT": {
        "XRPUSDT": 0.75,   # Both altcoins, move together
        "AVAXUSDT": 0.65,  # Moderate correlation
        "BTCUSDT": 0.60,
        "ETHUSDT": 0.55,
    },
    "XRPUSDT": {
        "DOGEUSDT": 0.75,
        "AVAXUSDT": 0.70,
        "BTCUSDT": 0.65,
        "ETHUSDT": 0.60,
    },
    "AVAXUSDT": {
        "DOGEUSDT": 0.65,
        "XRPUSDT": 0.70,
        "BTCUSDT": 0.70,
        "ETHUSDT": 0.75,
    },
}

# Strategy to symbol mapping
STRATEGY_SYMBOLS = {
    "ScalpingHybrid_DOGE": "DOGEUSDT",
    "ScalpingHybrid_AVAX": "AVAXUSDT",
    "LLM_v4_LowDD": "XRPUSDT",
    "LLM_v3_Tight": "XRPUSDT",
}

# Default correlation threshold (0.7 = 70% correlation)
CORRELATION_THRESHOLD = 0.70


def get_symbol_for_strategy(strategy_name: str) -> str:
    """Get the trading symbol for a strategy"""
    return STRATEGY_SYMBOLS.get(strategy_name, "")


def get_correlation(symbol1: str, symbol2: str) -> float:
    """
    Get correlation between two symbols.
    Returns 0 if no data available, 1.0 if same symbol.
    """
    # Same symbol = perfect correlation
    if symbol1 == symbol2:
        return 1.0
    
    # Clean symbols
    s1 = symbol1.replace("/", "")
    s2 = symbol2.replace("/", "")
    
    # Look up in matrix
    if s1 in CORRELATION_MATRIX and s2 in CORRELATION_MATRIX[s1]:
        return CORRELATION_MATRIX[s1][s2]
    
    if s2 in CORRELATION_MATRIX and s1 in CORRELATION_MATRIX[s2]:
        return CORRELATION_MATRIX[s2][s1]
    
    # Default: assume moderate correlation for unknown pairs
    return 0.50


def is_correlated(
    new_symbol: str, 
    open_strategies: List[str], 
    threshold: float = CORRELATION_THRESHOLD
) -> Tuple[bool, str]:
    """
    Check if a new trade would be too correlated with existing open trades.
    
    Args:
        new_symbol: Symbol to check (e.g., "XRPUSDT")
        open_strategies: List of strategy names with open trades
        threshold: Correlation threshold (default 0.70)
    
    Returns:
        Tuple of (is_correlated, reason_string)
    """
    if not open_strategies:
        return False, ""
    
    new_sym = new_symbol.replace("/", "")
    
    for strategy in open_strategies:
        existing_symbol = get_symbol_for_strategy(strategy)
        if not existing_symbol:
            continue
            
        correlation = get_correlation(new_sym, existing_symbol)
        
        if correlation >= threshold:
            reason = f"Correlated with {strategy} ({existing_symbol}): {correlation:.0%}"
            logger.warning(f"⚠️ {new_sym} skipped - {reason}")
            return True, reason
    
    return False, ""


def get_correlation_matrix_for_symbols(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Get correlation matrix for a list of symbols.
    Returns dict of dict with correlation values.
    """
    result = {}
    for s1 in symbols:
        result[s1] = {}
        for s2 in symbols:
            result[s1][s2] = get_correlation(s1, s2)
    return result
