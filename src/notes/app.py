"""StockandCrypto Notes 后端 API。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    decode_token,
    get_jwt_identity,
    jwt_required,
)

try:
    from .models import (
        ChatBoard,
        ChatMember,
        ChatMessage,
        DEFAULT_CHAT_BOARDS,
        Like,
        Note,
        TradePlan,
        TradingJournal,
        User,
        db,
    )
except ImportError:  # pragma: no cover
    from models import (  # type: ignore
        ChatBoard,
        ChatMember,
        ChatMessage,
        DEFAULT_CHAT_BOARDS,
        Like,
        Note,
        TradePlan,
        TradingJournal,
        User,
        db,
    )


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_db_uri() -> str:
    db_dir = _root_dir() / "data" / "notes"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_dir / 'notes.db'}"


def _json_body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(tag).strip() for tag in raw if str(tag).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(tag).strip() for tag in parsed if str(tag).strip()]
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        return [x.strip() for x in text.split(",") if x.strip()]
    return []


def _safe_page_size(raw: Any, default: int = 20, max_size: int = 100) -> int:
    value = _as_int(raw)
    if value is None:
        return default
    return max(1, min(value, max_size))


def _safe_page(raw: Any, default: int = 1) -> int:
    value = _as_int(raw)
    if value is None:
        return default
    return max(1, value)


def _current_user_id_optional() -> int | None:
    auth = request.headers.get("Authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        decoded = decode_token(token)
        sub = decoded.get("sub")
        return int(sub) if sub is not None else None
    except Exception:
        return None


def _is_note_visible(note: Note, viewer_id: int | None) -> bool:
    return note.is_public or (viewer_id is not None and note.user_id == viewer_id)


def _compute_rr(entry: float | None, stop_loss: float | None, take_profit: float | None) -> float | None:
    if entry is None or stop_loss is None or take_profit is None:
        return None
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0:
        return None
    return round(reward / risk, 4)


def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    """创建 Notes API 应用。"""

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("NOTES_DATABASE_URL", _default_db_uri())
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JWT_SECRET_KEY"] = os.getenv("NOTES_JWT_SECRET", "stockandcrypto-notes-dev-secret")

    if config_overrides:
        app.config.update(config_overrides)

    db.init_app(app)
    JWTManager(app)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "service": "notes-api"})

    @app.route("/api/auth/register", methods=["POST"])
    def auth_register():
        data = _json_body()
        username = str(data.get("username", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        if len(username) < 3:
            return _error("username 至少 3 个字符")
        if "@" not in email:
            return _error("email 格式错误")
        if len(password) < 6:
            return _error("password 至少 6 个字符")
        if User.query.filter_by(username=username).first():
            return _error("username 已存在", 409)
        if User.query.filter_by(email=email).first():
            return _error("email 已存在", 409)

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return jsonify({"ok": True, "user": user.to_dict()}), 201

    @app.route("/api/auth/login", methods=["POST"])
    def auth_login():
        data = _json_body()
        username = str(data.get("username", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))
        if not password:
            return _error("password 不能为空")

        user = None
        if username:
            user = User.query.filter_by(username=username).first()
        elif email:
            user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            return _error("用户名或密码错误", 401)

        token = create_access_token(identity=str(user.id))
        return jsonify({"ok": True, "token": token, "user": user.to_dict()})

    @app.route("/api/auth/me", methods=["GET"])
    @jwt_required()
    def auth_me():
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        if user is None:
            return _error("用户不存在", 404)
        return jsonify({"ok": True, "user": user.to_dict()})

    @app.route("/api/auth/profile", methods=["PUT"])
    @jwt_required()
    def auth_profile():
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        if user is None:
            return _error("用户不存在", 404)
        data = _json_body()
        if "avatar_url" in data:
            user.avatar_url = str(data.get("avatar_url") or "").strip() or None
        if "bio" in data:
            user.bio = str(data.get("bio") or "").strip() or None
        db.session.commit()
        return jsonify({"ok": True, "user": user.to_dict()})

    @app.route("/api/notes", methods=["POST"])
    @jwt_required()
    def create_note():
        user_id = int(get_jwt_identity())
        data = _json_body()
        title = str(data.get("title", "")).strip()
        content = str(data.get("content", "")).strip()
        if not title:
            return _error("title 不能为空")
        note = Note(
            user_id=user_id,
            title=title,
            content=content,
            is_public=_as_bool(data.get("is_public"), False),
            is_trading_journal=_as_bool(data.get("is_trading_journal"), False),
            note_type=str(data.get("note_type", "NOTE")).upper(),
        )
        note.set_tags(_parse_tags(data.get("tags")))
        db.session.add(note)
        db.session.commit()
        return jsonify({"ok": True, "note": note.to_dict(viewer_id=user_id)}), 201

    @app.route("/api/notes", methods=["GET"])
    @app.route("/api/notes/search", methods=["GET"])
    def list_notes():
        viewer_id = _current_user_id_optional()
        query = Note.query

        mine = _as_bool(request.args.get("mine"), False)
        user_id = _as_int(request.args.get("user_id"))
        tag = str(request.args.get("tag", "")).strip()
        search = str(request.args.get("q", "")).strip()
        note_type = str(request.args.get("note_type", "")).strip().upper()

        if mine:
            if viewer_id is None:
                return _error("需要登录", 401)
            query = query.filter(Note.user_id == viewer_id)
        elif user_id is not None:
            if viewer_id == user_id:
                query = query.filter(Note.user_id == user_id)
            else:
                query = query.filter(Note.user_id == user_id, Note.is_public.is_(True))
        elif viewer_id is not None:
            query = query.filter(db.or_(Note.is_public.is_(True), Note.user_id == viewer_id))
        else:
            query = query.filter(Note.is_public.is_(True))

        if tag:
            query = query.filter(Note.tags.like(f'%"{tag}"%'))
        if search:
            like = f"%{search}%"
            query = query.filter(db.or_(Note.title.ilike(like), Note.content.ilike(like), Note.tags.ilike(like)))
        if note_type:
            query = query.filter(Note.note_type == note_type)

        query = query.order_by(Note.updated_at.desc())
        page = _safe_page(request.args.get("page"))
        page_size = _safe_page_size(request.args.get("page_size"))
        pagination = query.paginate(page=page, per_page=page_size, error_out=False)
        return jsonify(
            {
                "ok": True,
                "items": [item.to_dict(viewer_id=viewer_id) for item in pagination.items],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": pagination.total,
                    "pages": pagination.pages,
                },
            }
        )

    @app.route("/api/notes/<int:note_id>", methods=["GET"])
    def get_note(note_id: int):
        viewer_id = _current_user_id_optional()
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        if not _is_note_visible(note, viewer_id):
            return _error("没有权限访问该 note", 403)

        payload = note.to_dict(viewer_id=viewer_id)
        journal = TradingJournal.query.filter_by(note_id=note.id).first()
        if journal:
            payload["journal"] = journal.to_dict()
        payload["trade_plans"] = [plan.to_dict(viewer_id=viewer_id) for plan in note.trade_plans.order_by(TradePlan.created_at.desc()).all()]
        return jsonify({"ok": True, "note": payload})

    @app.route("/api/notes/<int:note_id>", methods=["PUT"])
    @jwt_required()
    def update_note(note_id: int):
        user_id = int(get_jwt_identity())
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        if note.user_id != user_id:
            return _error("无权修改该 note", 403)
        data = _json_body()
        if "title" in data:
            title = str(data.get("title") or "").strip()
            if not title:
                return _error("title 不能为空")
            note.title = title
        if "content" in data:
            note.content = str(data.get("content") or "")
        if "tags" in data:
            note.set_tags(_parse_tags(data.get("tags")))
        if "is_public" in data:
            note.is_public = _as_bool(data.get("is_public"))
        if "note_type" in data:
            note.note_type = str(data.get("note_type") or "NOTE").upper()
        db.session.commit()
        return jsonify({"ok": True, "note": note.to_dict(viewer_id=user_id)})

    @app.route("/api/notes/<int:note_id>", methods=["DELETE"])
    @jwt_required()
    def delete_note(note_id: int):
        user_id = int(get_jwt_identity())
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        if note.user_id != user_id:
            return _error("无权删除该 note", 403)
        Like.query.filter_by(target_type="NOTE", target_id=note.id).delete()
        db.session.delete(note)
        db.session.commit()
        return jsonify({"ok": True})

    @app.route("/api/notes/<int:note_id>/publish", methods=["POST"])
    @jwt_required()
    def publish_note(note_id: int):
        user_id = int(get_jwt_identity())
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        if note.user_id != user_id:
            return _error("无权修改可见性", 403)
        note.is_public = True
        db.session.commit()
        return jsonify({"ok": True, "note": note.to_dict(viewer_id=user_id)})

    @app.route("/api/notes/<int:note_id>/unpublish", methods=["POST"])
    @jwt_required()
    def unpublish_note(note_id: int):
        user_id = int(get_jwt_identity())
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        if note.user_id != user_id:
            return _error("无权修改可见性", 403)
        note.is_public = False
        db.session.commit()
        return jsonify({"ok": True, "note": note.to_dict(viewer_id=user_id)})

    @app.route("/api/community/notes", methods=["GET"])
    @app.route("/api/notes/public", methods=["GET"])
    def list_public_notes():
        viewer_id = _current_user_id_optional()
        query = Note.query.filter(Note.is_public.is_(True)).order_by(Note.updated_at.desc())
        page = _safe_page(request.args.get("page"))
        page_size = _safe_page_size(request.args.get("page_size"))
        pagination = query.paginate(page=page, per_page=page_size, error_out=False)
        return jsonify(
            {
                "ok": True,
                "items": [item.to_dict(viewer_id=viewer_id) for item in pagination.items],
                "pagination": {"page": page, "page_size": page_size, "total": pagination.total, "pages": pagination.pages},
            }
        )

    @app.route("/api/notes/journal", methods=["POST"])
    @jwt_required()
    def create_journal():
        user_id = int(get_jwt_identity())
        data = _json_body()

        title = str(data.get("title", "")).strip() or f"{data.get('symbol', 'TRADE')} 交易日记"
        content = str(data.get("content", "")).strip()
        symbol = str(data.get("symbol", "")).strip().upper()
        entry_price = _as_float(data.get("entry_price"))
        if not symbol:
            return _error("symbol 不能为空")
        if entry_price is None:
            return _error("entry_price 必填且为数字")

        note = Note(
            user_id=user_id,
            title=title,
            content=content,
            is_public=_as_bool(data.get("is_public"), False),
            is_trading_journal=True,
            note_type="JOURNAL",
        )
        note.set_tags(_parse_tags(data.get("tags")))
        db.session.add(note)
        db.session.flush()

        stop_loss = _as_float(data.get("stop_loss"))
        take_profit = _as_float(data.get("take_profit"))
        journal = TradingJournal(
            user_id=user_id,
            note_id=note.id,
            symbol=symbol,
            direction=str(data.get("direction", "LONG")).upper(),
            entry_price=entry_price,
            exit_price=_as_float(data.get("exit_price")),
            position_size=_as_float(data.get("position_size")),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=_as_float(data.get("risk_reward_ratio")) or _compute_rr(entry_price, stop_loss, take_profit),
            pnl=_as_float(data.get("pnl")),
            pnl_percent=_as_float(data.get("pnl_percent")),
            status=str(data.get("status", "OPEN")).upper(),
            entry_time=_parse_iso_datetime(data.get("entry_time")),
            exit_time=_parse_iso_datetime(data.get("exit_time")),
            trade_reason=str(data.get("trade_reason", "")).strip() or None,
            screenshot_url=str(data.get("screenshot_url", "")).strip() or None,
        )
        db.session.add(journal)
        db.session.commit()
        return jsonify({"ok": True, "note": note.to_dict(viewer_id=user_id), "journal": journal.to_dict()}), 201

    @app.route("/api/notes/journal/<int:journal_id>", methods=["GET"])
    def get_journal(journal_id: int):
        viewer_id = _current_user_id_optional()
        journal = TradingJournal.query.get(journal_id)
        if journal is None:
            return _error("journal 不存在", 404)
        note = Note.query.get(journal.note_id)
        if note is None:
            return _error("关联 note 不存在", 404)
        if not _is_note_visible(note, viewer_id):
            return _error("没有权限访问该 journal", 403)
        return jsonify({"ok": True, "journal": journal.to_dict(), "note": note.to_dict(viewer_id=viewer_id)})

    @app.route("/api/users/<int:user_id>/journals", methods=["GET"])
    def list_user_journals(user_id: int):
        viewer_id = _current_user_id_optional()
        query = TradingJournal.query.filter(TradingJournal.user_id == user_id).join(Note, TradingJournal.note_id == Note.id)
        if viewer_id != user_id:
            query = query.filter(Note.is_public.is_(True))
        journals = query.order_by(TradingJournal.created_at.desc()).all()
        return jsonify({"ok": True, "items": [item.to_dict() for item in journals]})

    @app.route("/api/trade-plans", methods=["POST"])
    @jwt_required()
    def create_trade_plan():
        user_id = int(get_jwt_identity())
        data = _json_body()
        symbol = str(data.get("symbol", "")).strip().upper()
        title = str(data.get("title", "")).strip()
        analysis = str(data.get("analysis", "")).strip()
        if not symbol:
            return _error("symbol 不能为空")
        if not title:
            return _error("title 不能为空")
        if not analysis:
            return _error("analysis 不能为空")

        note_id = _as_int(data.get("note_id"))
        note = None
        if note_id is not None:
            note = Note.query.get(note_id)
            if note is None:
                return _error("note 不存在", 404)
            if note.user_id != user_id:
                return _error("无权使用该 note", 403)
        else:
            note = Note(
                user_id=user_id,
                title=title,
                content=analysis,
                is_public=_as_bool(data.get("is_public"), True),
                is_trading_journal=False,
                note_type="PLAN",
            )
            note.set_tags(_parse_tags(data.get("tags")))
            db.session.add(note)
            db.session.flush()

        entry_zone = _as_float(data.get("entry_zone"))
        stop_loss = _as_float(data.get("stop_loss"))
        take_profit = _as_float(data.get("take_profit"))
        plan = TradePlan(
            user_id=user_id,
            note_id=note.id,
            symbol=symbol,
            title=title,
            analysis=analysis,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=_as_float(data.get("risk_reward_ratio")) or _compute_rr(entry_zone, stop_loss, take_profit),
            confidence_level=_as_int(data.get("confidence_level")),
            status=str(data.get("status", "ACTIVE")).upper(),
            expires_at=_parse_iso_datetime(data.get("expires_at")),
        )
        db.session.add(plan)
        db.session.commit()
        return jsonify({"ok": True, "trade_plan": plan.to_dict(viewer_id=user_id)}), 201

    @app.route("/api/trade-plans", methods=["GET"])
    def list_trade_plans():
        viewer_id = _current_user_id_optional()
        query = TradePlan.query.join(Note, TradePlan.note_id == Note.id)

        mine = _as_bool(request.args.get("mine"), False)
        symbol = str(request.args.get("symbol", "")).strip().upper()
        status = str(request.args.get("status", "")).strip().upper()
        order_by = str(request.args.get("order", "new")).strip().lower()

        if mine:
            if viewer_id is None:
                return _error("需要登录", 401)
            query = query.filter(TradePlan.user_id == viewer_id)
        elif viewer_id is not None:
            query = query.filter(db.or_(Note.is_public.is_(True), TradePlan.user_id == viewer_id))
        else:
            query = query.filter(Note.is_public.is_(True))

        if symbol:
            query = query.filter(TradePlan.symbol == symbol)
        if status:
            query = query.filter(TradePlan.status == status)

        if order_by == "hot":
            query = query.order_by(TradePlan.views.desc(), TradePlan.created_at.desc())
        else:
            query = query.order_by(TradePlan.created_at.desc())

        page = _safe_page(request.args.get("page"))
        page_size = _safe_page_size(request.args.get("page_size"))
        pagination = query.paginate(page=page, per_page=page_size, error_out=False)
        return jsonify(
            {
                "ok": True,
                "items": [item.to_dict(viewer_id=viewer_id) for item in pagination.items],
                "pagination": {"page": page, "page_size": page_size, "total": pagination.total, "pages": pagination.pages},
            }
        )

    @app.route("/api/trade-plans/<int:plan_id>", methods=["GET"])
    def get_trade_plan(plan_id: int):
        viewer_id = _current_user_id_optional()
        plan = TradePlan.query.get(plan_id)
        if plan is None:
            return _error("trade plan 不存在", 404)
        if plan.note is None:
            return _error("关联 note 不存在", 404)
        if not (plan.note.is_public or (viewer_id is not None and plan.user_id == viewer_id)):
            return _error("无权访问该 trade plan", 403)
        if viewer_id is None or viewer_id != plan.user_id:
            plan.views += 1
            db.session.commit()
        return jsonify({"ok": True, "trade_plan": plan.to_dict(viewer_id=viewer_id)})

    @app.route("/api/trade-plans/<int:plan_id>", methods=["PUT"])
    @jwt_required()
    def update_trade_plan(plan_id: int):
        user_id = int(get_jwt_identity())
        plan = TradePlan.query.get(plan_id)
        if plan is None:
            return _error("trade plan 不存在", 404)
        if plan.user_id != user_id:
            return _error("无权修改该 trade plan", 403)
        data = _json_body()

        if "symbol" in data:
            symbol = str(data.get("symbol", "")).strip().upper()
            if not symbol:
                return _error("symbol 不能为空")
            plan.symbol = symbol
        if "title" in data:
            title = str(data.get("title", "")).strip()
            if not title:
                return _error("title 不能为空")
            plan.title = title
            if plan.note:
                plan.note.title = title
        if "analysis" in data:
            analysis = str(data.get("analysis", "")).strip()
            if not analysis:
                return _error("analysis 不能为空")
            plan.analysis = analysis
            if plan.note:
                plan.note.content = analysis

        if "entry_zone" in data:
            plan.entry_zone = _as_float(data.get("entry_zone"))
        if "stop_loss" in data:
            plan.stop_loss = _as_float(data.get("stop_loss"))
        if "take_profit" in data:
            plan.take_profit = _as_float(data.get("take_profit"))
        if "risk_reward_ratio" in data:
            plan.risk_reward_ratio = _as_float(data.get("risk_reward_ratio"))
        elif any(k in data for k in ("entry_zone", "stop_loss", "take_profit")):
            plan.risk_reward_ratio = _compute_rr(plan.entry_zone, plan.stop_loss, plan.take_profit)

        if "confidence_level" in data:
            plan.confidence_level = _as_int(data.get("confidence_level"))
        if "status" in data:
            plan.status = str(data.get("status", "ACTIVE")).upper()
        if "expires_at" in data:
            plan.expires_at = _parse_iso_datetime(data.get("expires_at"))
        if "is_public" in data and plan.note:
            plan.note.is_public = _as_bool(data.get("is_public"), True)

        db.session.commit()
        return jsonify({"ok": True, "trade_plan": plan.to_dict(viewer_id=user_id)})

    @app.route("/api/trade-plans/<int:plan_id>", methods=["DELETE"])
    @jwt_required()
    def delete_trade_plan(plan_id: int):
        user_id = int(get_jwt_identity())
        plan = TradePlan.query.get(plan_id)
        if plan is None:
            return _error("trade plan 不存在", 404)
        if plan.user_id != user_id:
            return _error("无权删除该 trade plan", 403)
        Like.query.filter_by(target_type="TRADE_PLAN", target_id=plan.id).delete()
        note_id = plan.note_id
        db.session.delete(plan)
        # 若 note 仅用于该 plan，顺便删除
        if note_id:
            sibling_count = TradePlan.query.filter_by(note_id=note_id).count()
            if sibling_count <= 1:
                note = Note.query.get(note_id)
                if note and note.user_id == user_id and note.note_type == "PLAN":
                    db.session.delete(note)
        db.session.commit()
        return jsonify({"ok": True})

    def _create_like(user_id: int, target_type: str, target_id: int):
        like = Like.query.filter_by(user_id=user_id, target_type=target_type, target_id=target_id).first()
        if like is None:
            like = Like(user_id=user_id, target_type=target_type, target_id=target_id)
            db.session.add(like)
            db.session.commit()
            return True
        return False

    def _remove_like(user_id: int, target_type: str, target_id: int):
        like = Like.query.filter_by(user_id=user_id, target_type=target_type, target_id=target_id).first()
        if like:
            db.session.delete(like)
            db.session.commit()
            return True
        return False

    def _list_like_users(target_type: str, target_id: int):
        likes = Like.query.filter_by(target_type=target_type, target_id=target_id).order_by(Like.created_at.desc()).all()
        rows = []
        for row in likes:
            user = User.query.get(row.user_id)
            rows.append(
                {
                    "user_id": row.user_id,
                    "username": user.username if user else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
        return rows

    @app.route("/api/notes/<int:note_id>/like", methods=["POST"])
    @jwt_required()
    def like_note(note_id: int):
        user_id = int(get_jwt_identity())
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        if not _is_note_visible(note, user_id):
            return _error("无权点赞该 note", 403)
        created = _create_like(user_id, "NOTE", note_id)
        return jsonify({"ok": True, "liked": True, "created": created, "like_count": note.like_count()})

    @app.route("/api/notes/<int:note_id>/like", methods=["DELETE"])
    @jwt_required()
    def unlike_note(note_id: int):
        user_id = int(get_jwt_identity())
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        removed = _remove_like(user_id, "NOTE", note_id)
        return jsonify({"ok": True, "liked": False, "removed": removed, "like_count": note.like_count()})

    @app.route("/api/notes/<int:note_id>/likes", methods=["GET"])
    def list_note_likes(note_id: int):
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        return jsonify({"ok": True, "items": _list_like_users("NOTE", note_id), "count": note.like_count()})

    @app.route("/api/trade-plans/<int:plan_id>/like", methods=["POST"])
    @jwt_required()
    def like_trade_plan(plan_id: int):
        user_id = int(get_jwt_identity())
        plan = TradePlan.query.get(plan_id)
        if plan is None:
            return _error("trade plan 不存在", 404)
        if plan.note is None or not (plan.note.is_public or plan.user_id == user_id):
            return _error("无权点赞该 trade plan", 403)
        created = _create_like(user_id, "TRADE_PLAN", plan_id)
        return jsonify({"ok": True, "liked": True, "created": created, "like_count": plan.like_count()})

    @app.route("/api/trade-plans/<int:plan_id>/like", methods=["DELETE"])
    @jwt_required()
    def unlike_trade_plan(plan_id: int):
        user_id = int(get_jwt_identity())
        plan = TradePlan.query.get(plan_id)
        if plan is None:
            return _error("trade plan 不存在", 404)
        removed = _remove_like(user_id, "TRADE_PLAN", plan_id)
        return jsonify({"ok": True, "liked": False, "removed": removed, "like_count": plan.like_count()})

    @app.route("/api/trade-plans/<int:plan_id>/likes", methods=["GET"])
    def list_trade_plan_likes(plan_id: int):
        plan = TradePlan.query.get(plan_id)
        if plan is None:
            return _error("trade plan 不存在", 404)
        return jsonify({"ok": True, "items": _list_like_users("TRADE_PLAN", plan_id), "count": plan.like_count()})

    @app.route("/api/notes/<int:note_id>/share", methods=["POST"])
    def share_note_link(note_id: int):
        note = Note.query.get(note_id)
        if note is None:
            return _error("note 不存在", 404)
        link = f"/notes/{note_id}"
        return jsonify({"ok": True, "share_link": link})

    @app.route("/api/chat/boards", methods=["GET"])
    @app.route("/api/boards/list", methods=["GET"])
    def list_boards():
        boards = ChatBoard.query.order_by(ChatBoard.created_at.asc()).all()
        return jsonify({"ok": True, "items": [board.to_dict() for board in boards]})

    @app.route("/api/chat/boards", methods=["POST"])
    @app.route("/api/boards/create", methods=["POST"])
    @jwt_required()
    def create_board():
        user_id = int(get_jwt_identity())
        data = _json_body()
        name = str(data.get("name", "")).strip()
        if not name:
            return _error("name 不能为空")
        board = ChatBoard(
            name=name,
            description=str(data.get("description") or "").strip() or None,
            topic=str(data.get("topic") or "").strip() or None,
            created_by=user_id,
        )
        db.session.add(board)
        db.session.flush()
        db.session.add(ChatMember(board_id=board.id, user_id=user_id, role="ADMIN"))
        db.session.commit()
        return jsonify({"ok": True, "board": board.to_dict()}), 201

    @app.route("/api/chat/boards/<int:board_id>", methods=["GET"])
    @app.route("/api/boards/<int:board_id>", methods=["GET"])
    def get_board(board_id: int):
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        return jsonify({"ok": True, "board": board.to_dict()})

    @app.route("/api/chat/boards/<int:board_id>/join", methods=["POST"])
    @app.route("/api/boards/<int:board_id>/join", methods=["POST"])
    @jwt_required()
    def join_board(board_id: int):
        user_id = int(get_jwt_identity())
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        member = ChatMember.query.filter_by(board_id=board_id, user_id=user_id).first()
        if member is None:
            member = ChatMember(board_id=board_id, user_id=user_id, role="MEMBER")
            db.session.add(member)
            db.session.commit()
            created = True
        else:
            created = False
        return jsonify({"ok": True, "joined": True, "created": created, "member": member.to_dict()})

    @app.route("/api/chat/boards/<int:board_id>/leave", methods=["POST"])
    @app.route("/api/boards/<int:board_id>/leave", methods=["POST"])
    @jwt_required()
    def leave_board(board_id: int):
        user_id = int(get_jwt_identity())
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        member = ChatMember.query.filter_by(board_id=board_id, user_id=user_id).first()
        if member:
            db.session.delete(member)
            db.session.commit()
            removed = True
        else:
            removed = False
        return jsonify({"ok": True, "left": True, "removed": removed})

    @app.route("/api/chat/boards/<int:board_id>/members", methods=["GET"])
    @app.route("/api/boards/<int:board_id>/members", methods=["GET"])
    def board_members(board_id: int):
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        members = ChatMember.query.filter_by(board_id=board_id).order_by(ChatMember.joined_at.asc()).all()
        return jsonify({"ok": True, "items": [item.to_dict() for item in members]})

    @app.route("/api/chat/boards/<int:board_id>/messages", methods=["GET"])
    @app.route("/api/boards/<int:board_id>/messages", methods=["GET"])
    def board_messages(board_id: int):
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        limit = _safe_page_size(request.args.get("limit"), default=50, max_size=200)
        rows = ChatMessage.query.filter_by(board_id=board_id).order_by(ChatMessage.created_at.desc()).limit(limit).all()
        rows.reverse()
        return jsonify({"ok": True, "items": [item.to_dict() for item in rows]})

    @app.route("/api/chat/boards/<int:board_id>/messages", methods=["POST"])
    @app.route("/api/boards/<int:board_id>/messages", methods=["POST"])
    @jwt_required()
    def send_board_message(board_id: int):
        user_id = int(get_jwt_identity())
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        is_member = ChatMember.query.filter_by(board_id=board_id, user_id=user_id).first() is not None
        if not is_member:
            return _error("请先加入该版块", 403)
        data = _json_body()
        content = str(data.get("content", "")).strip()
        if not content:
            return _error("content 不能为空")
        message = ChatMessage(
            board_id=board_id,
            user_id=user_id,
            content=content,
            message_type=str(data.get("message_type", "TEXT")).upper(),
        )
        db.session.add(message)
        db.session.commit()
        return jsonify({"ok": True, "message": message.to_dict()}), 201

    @app.route("/api/chat/boards/<int:board_id>/stats", methods=["GET"])
    def board_stats(board_id: int):
        board = ChatBoard.query.get(board_id)
        if board is None:
            return _error("board 不存在", 404)
        latest_message = (
            ChatMessage.query.filter_by(board_id=board_id).order_by(ChatMessage.created_at.desc()).first()
        )
        return jsonify(
            {
                "ok": True,
                "stats": {
                    "board_id": board_id,
                    "member_count": board.members.count(),
                    "message_count": board.messages.count(),
                    "latest_message_at": latest_message.created_at.isoformat() if latest_message else None,
                },
            }
        )

    @app.cli.command("notes-init")
    def notes_init():
        db.create_all()
        print("Notes tables created.")

    return app


def seed_default_boards(created_by: int | None = None) -> int:
    """幂等插入默认聊天版块。"""
    inserted = 0
    for payload in DEFAULT_CHAT_BOARDS:
        exists = ChatBoard.query.filter_by(name=payload["name"]).first()
        if exists:
            continue
        board = ChatBoard(
            name=payload["name"],
            description=payload.get("description"),
            topic=payload.get("topic"),
            created_by=created_by,
        )
        db.session.add(board)
        db.session.flush()
        if created_by is not None:
            db.session.add(ChatMember(board_id=board.id, user_id=created_by, role="ADMIN"))
        inserted += 1
    db.session.commit()
    return inserted


app = create_app()


if __name__ == "__main__":
    host = os.getenv("NOTES_HOST", "127.0.0.1")
    port = int(os.getenv("NOTES_PORT", "5001"))
    debug = _as_bool(os.getenv("NOTES_DEBUG"), False)
    with app.app_context():
        db.create_all()
    app.run(host=host, port=port, debug=debug)
