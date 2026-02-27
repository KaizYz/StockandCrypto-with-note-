"""初始化 Notes 数据库并写入 seed 数据。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notes.app import create_app, seed_default_boards
from src.notes.models import Like, Note, TradePlan, TradingJournal, User, db


def _ensure_user(username: str, email: str, password: str) -> User:
    user = User.query.filter_by(username=username).first()
    if user:
        return user
    user = User(username=username, email=email, bio=f"{username} profile")
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


def _seed_notes_and_plans(owner: User, guest: User) -> None:
    if Note.query.count() > 0:
        return

    note = Note(
        user_id=owner.id,
        title="BTC Breakout Plan",
        content="关注 4H 突破结构，若回踩支撑成功考虑入场。",
        is_public=True,
        note_type="PLAN",
    )
    note.set_tags(["crypto", "btc", "breakout"])
    db.session.add(note)
    db.session.flush()

    plan = TradePlan(
        user_id=owner.id,
        note_id=note.id,
        symbol="BTCUSDT",
        title="BTC Breakout Plan",
        analysis=note.content,
        entry_zone=104000.0,
        stop_loss=101500.0,
        take_profit=109000.0,
        risk_reward_ratio=2.0,
        confidence_level=4,
        status="ACTIVE",
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.session.add(plan)
    db.session.flush()
    db.session.add(Like(user_id=guest.id, target_type="TRADE_PLAN", target_id=plan.id))

    j_note = Note(
        user_id=owner.id,
        title="AAPL Swing Journal",
        content="交易前判断上升通道有效，按计划执行止损。",
        is_public=False,
        is_trading_journal=True,
        note_type="JOURNAL",
    )
    j_note.set_tags(["stock", "aapl", "journal"])
    db.session.add(j_note)
    db.session.flush()

    journal = TradingJournal(
        user_id=owner.id,
        note_id=j_note.id,
        symbol="AAPL",
        direction="LONG",
        entry_price=181.25,
        exit_price=186.80,
        position_size=100.0,
        stop_loss=178.0,
        take_profit=188.0,
        risk_reward_ratio=2.08,
        pnl=555.0,
        pnl_percent=3.06,
        status="CLOSED",
        entry_time=datetime.utcnow() - timedelta(days=2),
        exit_time=datetime.utcnow() - timedelta(days=1),
        trade_reason="趋势延续 + 回踩均线企稳。",
    )
    db.session.add(journal)
    db.session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Init Notes DB")
    parser.add_argument("--reset", action="store_true", help="drop all tables before init")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        if args.reset:
            db.drop_all()
        db.create_all()

        demo = _ensure_user("demo", "demo@example.com", "demo123456")
        guest = _ensure_user("guest", "guest@example.com", "guest123456")
        seed_default_boards(created_by=demo.id)
        _seed_notes_and_plans(owner=demo, guest=guest)

    print("Notes DB initialized.")
    print("Demo login: username=demo password=demo123456")
    print("Guest login: username=guest password=guest123456")


if __name__ == "__main__":
    main()
