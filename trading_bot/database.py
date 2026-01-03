"""
Trade Database Storage
Stores all trades for analysis and dashboard
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

from .config import DATABASE_URL

logger = logging.getLogger(__name__)
Base = declarative_base()


class TradeRecord(Base):
    """Database model for trades"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    strategy = Column(String(50))
    symbol = Column(String(20))
    side = Column(String(10))  # long or short
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, nullable=True)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float)
    pnl_usd = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    commission = Column(Float, default=0)
    exit_reason = Column(String(50), nullable=True)
    is_open = Column(Boolean, default=True)
    is_paper = Column(Boolean, default=True)
    
    # Additional analytics fields
    max_drawdown = Column(Float, nullable=True)
    max_runup = Column(Float, nullable=True)
    bars_held = Column(Integer, nullable=True)
    
    # Indicator values at entry (for LLM analysis)
    entry_rsi = Column(Float, nullable=True)
    entry_adx = Column(Float, nullable=True)
    entry_macd = Column(Float, nullable=True)
    
    # LLM Analysis
    llm_analysis = Column(String(2000), nullable=True)
    llm_recommendation = Column(String(500), nullable=True)


class StrategyLog(Base):
    """Log of strategy execution checks"""
    __tablename__ = "strategy_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    strategy = Column(String(50))
    symbol = Column(String(20))
    status = Column(String(20))  # CHECKING, SIGNAL, NO_SIGNAL, ERROR
    message = Column(String(500))
    price = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)


class TrainingDataLog(Base):
    """LLM training data for fine-tuning momentum expert"""
    __tablename__ = "training_data_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    symbol = Column(String(20))
    strategy = Column(String(50))
    analysis_type = Column(String(30))  # entry_signal, no_signal
    signal_generated = Column(Boolean)
    
    # Indicators at analysis time
    price = Column(Float)
    rsi = Column(Float)
    adx = Column(Float)
    ema_15 = Column(Float, nullable=True)
    ema_30 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    
    # LLM generated content for training
    instruction = Column(String(1000))  # Input for fine-tuning
    output = Column(String(2000))       # Output for fine-tuning
    
    # Metadata
    confidence = Column(String(20), nullable=True)
    raw_indicators_json = Column(String(2000), nullable=True)


class TradeDatabase:
    """Database interface for trade storage"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or DATABASE_URL
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Trade database initialized: {self.db_url}")
    
    def save_trade(self, trade: TradeRecord) -> int:
        """Save a new trade"""
        session = self.Session()
        try:
            session.add(trade)
            session.commit()
            trade_id = trade.id
            logger.info(f"Saved trade #{trade_id}: {trade.strategy} {trade.side} {trade.symbol}")
            return trade_id
        finally:
            session.close()
    
    def close_trade(
        self, 
        trade_id: int, 
        exit_price: float, 
        exit_reason: str,
        pnl_usd: float,
        pnl_pct: float
    ):
        """Close an open trade"""
        session = self.Session()
        try:
            trade = session.query(TradeRecord).filter_by(id=trade_id).first()
            if trade:
                trade.exit_time = datetime.now()
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl_usd = pnl_usd
                trade.pnl_pct = pnl_pct
                trade.is_open = False
                session.commit()
                logger.info(f"Closed trade #{trade_id}: {exit_reason}, PnL: {pnl_pct:.2f}%")
        finally:
            session.close()
    
    def get_open_trades(self, strategy: str = None) -> List[TradeRecord]:
        """Get all open trades"""
        session = self.Session()
        try:
            query = session.query(TradeRecord).filter_by(is_open=True)
            if strategy:
                query = query.filter_by(strategy=strategy)
            return query.all()
        finally:
            session.close()
    
    def get_trades(
        self, 
        strategy: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100
    ) -> List[TradeRecord]:
        """Get trades with optional filters"""
        session = self.Session()
        try:
            query = session.query(TradeRecord)
            
            if strategy:
                query = query.filter_by(strategy=strategy)
            if start_date:
                query = query.filter(TradeRecord.entry_time >= start_date)
            if end_date:
                query = query.filter(TradeRecord.entry_time <= end_date)
            
            return query.order_by(TradeRecord.entry_time.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_strategy_stats(self, strategy: str) -> dict:
        """Get statistics for a strategy"""
        session = self.Session()
        try:
            trades = session.query(TradeRecord).filter_by(
                strategy=strategy, is_open=False
            ).all()
            
            if not trades:
                return {}
            
            wins = [t for t in trades if t.pnl_pct and t.pnl_pct > 0]
            losses = [t for t in trades if t.pnl_pct and t.pnl_pct <= 0]
            
            total_pnl = sum(t.pnl_usd or 0 for t in trades)
            total_pnl_pct = sum(t.pnl_pct or 0 for t in trades)
            
            return {
                "strategy": strategy,
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) * 100 if trades else 0,
                "total_pnl_usd": total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "avg_win_pct": sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
                "avg_loss_pct": sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
            }
        finally:
            session.close()

    def log_strategy_check(self, strategy: str, symbol: str, status: str, message: str, price: float = None, rsi: float = None, adx: float = None):
        """Log a strategy check event"""
        session = self.Session()
        try:
            log = StrategyLog(
                strategy=strategy,
                symbol=symbol,
                status=status,
                message=message,
                price=price,
                rsi=rsi,
                adx=adx,
                timestamp=datetime.now()
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log strategy check: {e}")
        finally:
            session.close()

    def get_latest_logs(self, limit: int = 50) -> List[StrategyLog]:
        """Get latest strategy logs"""
        session = self.Session()
        try:
            return session.query(StrategyLog).order_by(StrategyLog.timestamp.desc()).limit(limit).all()
        finally:
            session.close()

    def update_trade_analysis(self, trade_id: int, analysis: str, recommendation: str = None):
        """Update trade with LLM analysis"""
        session = self.Session()
        try:
            trade = session.query(TradeRecord).filter_by(id=trade_id).first()
            if trade:
                trade.llm_analysis = analysis
                trade.llm_recommendation = recommendation
                session.commit()
                logger.info(f"Updated trade #{trade_id} with LLM analysis")
        except Exception as e:
            logger.error(f"Failed to save LLM analysis: {e}")
        finally:
            session.close()

    def save_training_data(
        self,
        symbol: str,
        strategy: str,
        analysis_type: str,
        signal_generated: bool,
        price: float,
        rsi: float,
        adx: float,
        instruction: str,
        output: str,
        ema_15: float = None,
        ema_30: float = None,
        macd: float = None,
        confidence: str = None,
        raw_indicators_json: str = None
    ) -> int:
        """Save LLM training data record"""
        session = self.Session()
        try:
            record = TrainingDataLog(
                symbol=symbol,
                strategy=strategy,
                analysis_type=analysis_type,
                signal_generated=signal_generated,
                price=price,
                rsi=rsi,
                adx=adx,
                ema_15=ema_15,
                ema_30=ema_30,
                macd=macd,
                instruction=instruction,
                output=output,
                confidence=confidence,
                raw_indicators_json=raw_indicators_json,
                timestamp=datetime.now()
            )
            session.add(record)
            session.commit()
            record_id = record.id
            logger.info(f"Saved training data #{record_id}: {analysis_type} for {symbol}")
            return record_id
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return -1
        finally:
            session.close()

    def get_training_data(self, limit: int = 1000) -> List[dict]:
        """Get training data for JSONL export"""
        session = self.Session()
        try:
            records = session.query(TrainingDataLog).order_by(
                TrainingDataLog.timestamp.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "symbol": r.symbol,
                    "strategy": r.strategy,
                    "analysis_type": r.analysis_type,
                    "signal_generated": r.signal_generated,
                    "instruction": r.instruction,
                    "output": r.output,
                    "price": r.price,
                    "rsi": r.rsi,
                    "adx": r.adx
                }
                for r in records
            ]
        finally:
            session.close()
