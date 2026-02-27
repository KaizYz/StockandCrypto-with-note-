"""
StockAndCrypto Notes - 配置入口
根据环境变量自动选择后端模式:
- USE_SUPABASE=true  → 直接连接 Supabase (Streamlit Cloud)
- USE_SUPABASE=false → 连接本地 Flask API (本地开发)
"""

from __future__ import annotations

import os
from typing import Any

import requests

# 配置 - 支持 Streamlit Cloud secrets.toml 格式
USE_SUPABASE = os.getenv("USE_SUPABASE", "").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not USE_SUPABASE:
    # 尝试从 Streamlit secrets 读取
    try:
        import toml
        secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            USE_SUPABASE = str(secrets.get("USE_SUPABASE", "")).lower() == "true"
            SUPABASE_URL = secrets.get("SUPABASE_URL", SUPABASE_URL)
            SUPABASE_ANON_KEY = secrets.get("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)
    except Exception:
        pass

API_BASE = os.getenv("NOTES_API_URL", "http://127.0.0.1:5001").rstrip("/")

print(f"📦 Notes 模块启动模式: {'Supabase Cloud' if USE_SUPABASE else 'Local API'}")
print(f"   URL: {SUPABASE_URL[:30]}..." if SUPABASE_URL else "   URL: 未设置")


# ========== 统一 API 接口 ==========

def _api(method: str, path: str, token: str | None = None, payload: dict[str, Any] | None = None):
    """统一 API 调用（本地 Flask 模式）"""
    url = f"{API_BASE}{path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.request(method=method, url=url, headers=headers, json=payload, timeout=10)
        data = resp.json()
        return resp.ok, data
    except Exception as e:
        return False, {"error": str(e)}


# ========== Supabase 客户端 ==========

_supabase_client = None
_supabase_auth = None

def get_supabase_client():
    """获取 Supabase 客户端"""
    global _supabase_client, _supabase_auth
    if _supabase_client is None and SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            from supabase import create_client, Client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            # 新版 supabase 需要单独获取 auth
            _supabase_auth = _supabase_client.auth
        except ImportError as e:
            print(f"⚠️ 请安装 supabase: {e}")
            _supabase_client = False
            _supabase_auth = False
    return _supabase_client

def get_supabase_auth():
    """获取 Supabase Auth"""
    global _supabase_auth
    if _supabase_auth is None:
        get_supabase_client()
    return _supabase_auth


# ========== 认证相关 ==========

def sign_up(email: str, password: str, username: str):
    """注册"""
    if USE_SUPABASE:
        auth = get_supabase_auth()
        if not auth:
            return False, {"error": "Supabase 未配置"}
        try:
            # 新版 supabase 注册
            result = auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": {"username": username}}
            })
            return True, result
        except Exception as e:
            return False, {"error": str(e)}
    else:
        return _api("POST", "/api/auth/register", payload={"email": email, "password": password, "username": username})


def sign_in(email: str, password: str):
    """登录"""
    if USE_SUPABASE:
        auth = get_supabase_auth()
        if not auth:
            return False, {"error": "Supabase 未配置"}
        try:
            # 新版 supabase 登录
            result = auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return True, result
        except Exception as e:
            return False, {"error": str(e)}
    else:
        return _api("POST", "/api/auth/login", payload={"email": email, "password": password})


def sign_out():
    """登出"""
    auth = get_supabase_auth()
    if auth:
        try:
            auth.sign_out()
        except Exception:
            pass


def get_current_user():
    """获取当前用户"""
    if USE_SUPABASE:
        auth = get_supabase_auth()
        if auth:
            try:
                session = auth.get_session()
                return session.user if session else None
            except Exception:
                return None
        return None
    else:
        ok, data = _api("GET", "/api/auth/me")
        return data.get("user") if ok else None


def get_current_token():
    """获取当前 token"""
    if USE_SUPABASE:
        auth = get_supabase_auth()
        if auth:
            try:
                session = auth.get_session()
                return session.access_token if session else None
            except Exception:
                return None
        return None
    else:
        return None


# ========== 笔记相关 ==========

def create_note(title: str, content: str, is_public: bool = False, tags: list | None = None):
    """创建笔记"""
    token = get_current_token()
    if not token:
        return False, {"error": "请先登录"}

    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return False, {"error": "Supabase 未配置"}

        user = get_current_user()
        data = {
            "user_id": str(user.id),
            "title": title,
            "content": content,
            "is_public": is_public,
            "tags": tags or [],
            "created_at": "now()"
        }
        try:
            result = client.table("notes").insert(data).execute()
            return True, result.data[0] if result.data else {}
        except Exception as e:
            return False, {"error": str(e)}
    else:
        return _api("POST", "/api/notes", token=token, payload={
            "title": title,
            "content": content,
            "is_public": is_public,
            "tags": tags
        })


def get_notes(mine_only: bool = False):
    """获取笔记列表"""
    token = get_current_token()
    user = get_current_user()

    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return []

        query = client.table("notes").select("*").order("created_at", desc=True)

        if mine_only and user:
            query = query.eq("user_id", str(user.id))
        else:
            query = query.eq("is_public", True)

        try:
            result = query.execute()
            return result.data or []
        except Exception:
            return []
    else:
        params = f"?mine=true" if mine_only else ""
        ok, data = _api("GET", f"/api/notes{params}")
        return data.get("items", []) if ok else []


# ========== 交易计划相关 ==========

def create_trade_plan(symbol: str, title: str, analysis: str, direction: str,
                      entry_price: float, stop_loss: float, take_profit: float,
                      confidence: int = 3, is_public: bool = True):
    """创建交易计划"""
    token = get_current_token()
    if not token:
        return False, {"error": "请先登录"}

    # 计算盈亏比
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr = round(reward / risk, 2) if risk > 0 else 0

    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return False, {"error": "Supabase 未配置"}

        user = get_current_user()
        data = {
            "user_id": str(user.id),
            "symbol": symbol.upper(),
            "title": title,
            "analysis": analysis,
            "direction": direction.upper(),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": rr,
            "confidence_level": confidence,
            "status": "ACTIVE",
            "is_public": is_public,
            "created_at": "now()"
        }
        try:
            result = client.table("trade_plans").insert(data).execute()
            return True, result.data[0] if result.data else {}
        except Exception as e:
            return False, {"error": str(e)}
    else:
        return _api("POST", "/api/trade-plans", token=token, payload={
            "symbol": symbol,
            "title": title,
            "analysis": analysis,
            "direction": direction,
            "entry_zone": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": rr,
            "confidence_level": confidence,
            "is_public": is_public
        })


def get_trade_plans(symbol: str | None = None, mine_only: bool = False):
    """获取交易计划列表"""
    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return []

        query = client.table("trade_plans").select("*").eq("status", "ACTIVE").order("created_at", desc=True)

        if symbol:
            query = query.eq("symbol", symbol.upper())

        try:
            result = query.execute()
            return result.data or []
        except Exception:
            return []
    else:
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if mine_only:
            params.append("mine=true")

        ok, data = _api("GET", f"/api/trade-plans?{'&'.join(params)}" if params else "/api/trade-plans")
        return data.get("items", []) if ok else []


def like_plan(plan_id: int):
    """点赞交易计划"""
    token = get_current_token()
    if not token:
        return False, {"error": "请先登录"}

    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return False, {"error": "Supabase 未配置"}

        user = get_current_user()
        data = {
            "user_id": str(user.id),
            "target_type": "TRADE_PLAN",
            "target_id": str(plan_id),
            "created_at": "now()"
        }
        try:
            client.table("likes").insert(data).execute()
            return True, {}
        except Exception:
            return False, {"error": "点赞失败"}
    else:
        return _api("POST", f"/api/trade-plans/{plan_id}/like", token=token)


# ========== 群聊相关 ==========

def get_chat_boards():
    """获取群聊版块"""
    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return []

        try:
            result = client.table("chat_boards").select("*").execute()
            return result.data or []
        except Exception:
            return []
    else:
        ok, data = _api("GET", "/api/chat/boards")
        return data.get("items", []) if ok else []


def get_chat_messages(board_id: int):
    """获取聊天消息"""
    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return []

        try:
            result = client.table("chat_messages").select("*, users(username, avatar_url)") \
                .eq("board_id", str(board_id)).order("created_at").execute()
            return result.data or []
        except Exception:
            return []
    else:
        ok, data = _api("GET", f"/api/chat/boards/{board_id}/messages")
        return data.get("items", []) if ok else []


def send_chat_message(board_id: int, content: str):
    """发送消息"""
    token = get_current_token()
    if not token:
        return False, {"error": "请先登录"}

    if USE_SUPABASE:
        client = get_supabase_client()
        if not client:
            return False, {"error": "Supabase 未配置"}

        user = get_current_user()
        data = {
            "board_id": str(board_id),
            "user_id": str(user.id),
            "content": content,
            "created_at": "now()"
        }
        try:
            client.table("chat_messages").insert(data).execute()
            return True, {}
        except Exception as e:
            return False, {"error": str(e)}
    else:
        return _api("POST", f"/api/chat/boards/{board_id}/messages", token=token, payload={"content": content})
