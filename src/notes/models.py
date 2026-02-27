"""StockandCrypto Notes 模块数据库模型。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def _now() -> datetime:
    return datetime.utcnow()


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class User(db.Model):
    """用户。"""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    avatar_url = db.Column(db.Text)
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=_now, nullable=False)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now, nullable=False)

    notes = db.relationship("Note", backref="author", lazy="dynamic", cascade="all, delete-orphan")
    journals = db.relationship("TradingJournal", backref="author", lazy="dynamic", cascade="all, delete-orphan")
    plans = db.relationship("TradePlan", backref="author", lazy="dynamic", cascade="all, delete-orphan")

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "created_at": _to_iso(self.created_at),
            "updated_at": _to_iso(self.updated_at),
        }


class Note(db.Model):
    """笔记。"""

    __tablename__ = "notes"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False, default="")
    tags = db.Column(db.Text, default="[]", nullable=False)  # JSON array
    is_public = db.Column(db.Boolean, default=False, nullable=False)
    is_trading_journal = db.Column(db.Boolean, default=False, nullable=False)
    note_type = db.Column(db.String(20), default="NOTE", nullable=False)  # NOTE/JOURNAL/PLAN
    created_at = db.Column(db.DateTime, default=_now, nullable=False)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now, nullable=False)

    journals = db.relationship("TradingJournal", backref="note", lazy="dynamic", cascade="all, delete-orphan")
    trade_plans = db.relationship("TradePlan", backref="note", lazy="dynamic", cascade="all, delete-orphan")

    def set_tags(self, tags_value: list[str]) -> None:
        clean_tags = [str(tag).strip() for tag in tags_value if str(tag).strip()]
        self.tags = json.dumps(clean_tags, ensure_ascii=False)

    def get_tags(self) -> list[str]:
        try:
            value = json.loads(self.tags or "[]")
            if isinstance(value, list):
                return [str(x) for x in value]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return []

    def like_count(self) -> int:
        return Like.query.filter_by(target_type="NOTE", target_id=self.id).count()

    def to_dict(self, viewer_id: int | None = None) -> dict[str, Any]:
        liked_by_viewer = False
        if viewer_id is not None:
            liked_by_viewer = (
                Like.query.filter_by(
                    user_id=viewer_id,
                    target_type="NOTE",
                    target_id=self.id,
                ).first()
                is not None
            )
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.author.username if self.author else None,
            "title": self.title,
            "content": self.content,
            "tags": self.get_tags(),
            "is_public": self.is_public,
            "is_trading_journal": self.is_trading_journal,
            "note_type": self.note_type,
            "like_count": self.like_count(),
            "liked_by_viewer": liked_by_viewer,
            "created_at": _to_iso(self.created_at),
            "updated_at": _to_iso(self.updated_at),
        }


class TradingJournal(db.Model):
    """交易日记。"""

    __tablename__ = "trading_journals"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    note_id = db.Column(db.Integer, db.ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    direction = db.Column(db.String(10), nullable=False, default="LONG")  # LONG/SHORT
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    position_size = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    risk_reward_ratio = db.Column(db.Float)
    pnl = db.Column(db.Float)
    pnl_percent = db.Column(db.Float)
    status = db.Column(db.String(20), default="OPEN", nullable=False)  # OPEN/CLOSED/CANCELLED
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime)
    trade_reason = db.Column(db.Text)
    screenshot_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=_now, nullable=False)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "note_id": self.note_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "status": self.status,
            "entry_time": _to_iso(self.entry_time),
            "exit_time": _to_iso(self.exit_time),
            "trade_reason": self.trade_reason,
            "screenshot_url": self.screenshot_url,
            "created_at": _to_iso(self.created_at),
            "updated_at": _to_iso(self.updated_at),
        }


class TradePlan(db.Model):
    """交易计划。"""

    __tablename__ = "trade_plans"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    note_id = db.Column(db.Integer, db.ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    analysis = db.Column(db.Text, nullable=False)
    entry_zone = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    risk_reward_ratio = db.Column(db.Float)
    confidence_level = db.Column(db.Integer)
    status = db.Column(db.String(20), default="ACTIVE", nullable=False)
    views = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, default=_now, nullable=False)
    expires_at = db.Column(db.DateTime)

    def like_count(self) -> int:
        return Like.query.filter_by(target_type="TRADE_PLAN", target_id=self.id).count()

    def to_dict(self, viewer_id: int | None = None) -> dict[str, Any]:
        liked_by_viewer = False
        if viewer_id is not None:
            liked_by_viewer = (
                Like.query.filter_by(
                    user_id=viewer_id,
                    target_type="TRADE_PLAN",
                    target_id=self.id,
                ).first()
                is not None
            )
        return {
            "id": self.id,
            "user_id": self.user_id,
            "note_id": self.note_id,
            "username": self.author.username if self.author else None,
            "symbol": self.symbol,
            "title": self.title,
            "analysis": self.analysis,
            "entry_zone": self.entry_zone,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "confidence_level": self.confidence_level,
            "status": self.status,
            "views": self.views,
            "like_count": self.like_count(),
            "liked_by_viewer": liked_by_viewer,
            "is_public": bool(self.note.is_public) if self.note else False,
            "created_at": _to_iso(self.created_at),
            "expires_at": _to_iso(self.expires_at),
        }


class Like(db.Model):
    """点赞记录。支持 NOTE / TRADE_PLAN。"""

    __tablename__ = "likes"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    target_type = db.Column(db.String(20), nullable=False)  # NOTE / TRADE_PLAN
    target_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=_now, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("user_id", "target_type", "target_id", name="uq_likes_target"),
    )


class ChatBoard(db.Model):
    """聊天版块。"""

    __tablename__ = "chat_boards"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    topic = db.Column(db.String(50))
    created_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"))
    created_at = db.Column(db.DateTime, default=_now, nullable=False)

    members = db.relationship("ChatMember", backref="board", lazy="dynamic", cascade="all, delete-orphan")
    messages = db.relationship("ChatMessage", backref="board", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "topic": self.topic,
            "created_by": self.created_by,
            "member_count": self.members.count(),
            "message_count": self.messages.count(),
            "created_at": _to_iso(self.created_at),
        }


class ChatMember(db.Model):
    """版块成员。"""

    __tablename__ = "chat_members"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    board_id = db.Column(db.Integer, db.ForeignKey("chat_boards.id", ondelete="CASCADE"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = db.Column(db.String(20), default="MEMBER", nullable=False)
    joined_at = db.Column(db.DateTime, default=_now, nullable=False)

    user = db.relationship("User", backref=db.backref("chat_memberships", lazy="dynamic"))

    __table_args__ = (
        db.UniqueConstraint("board_id", "user_id", name="uq_chat_board_member"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "board_id": self.board_id,
            "user_id": self.user_id,
            "username": self.user.username if self.user else None,
            "role": self.role,
            "joined_at": _to_iso(self.joined_at),
        }


class ChatMessage(db.Model):
    """聊天消息。"""

    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    board_id = db.Column(db.Integer, db.ForeignKey("chat_boards.id", ondelete="CASCADE"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default="TEXT", nullable=False)
    created_at = db.Column(db.DateTime, default=_now, nullable=False)

    user = db.relationship("User", backref=db.backref("chat_messages", lazy="dynamic"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "board_id": self.board_id,
            "user_id": self.user_id,
            "username": self.user.username if self.user else None,
            "content": self.content,
            "message_type": self.message_type,
            "created_at": _to_iso(self.created_at),
        }


DEFAULT_CHAT_BOARDS: list[dict[str, str]] = [
    {"name": "Crypto Trading", "description": "BTC/ETH/SOL 等加密资产讨论", "topic": "crypto"},
    {"name": "Stock Trading", "description": "A股、美股与指数交易讨论", "topic": "stock"},
    {"name": "Forex Trading", "description": "外汇行情与交易策略", "topic": "forex"},
]
