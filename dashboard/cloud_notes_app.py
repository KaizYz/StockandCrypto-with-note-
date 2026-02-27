"""
StockAndCrypto Notes - Supabase 版本 (可直接部署到 Streamlit Cloud)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import streamlit as st
import requests
from supabase import create_client, Client

# Supabase 配置（从环境变量读取）
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("请配置 Supabase 环境变量: SUPABASE_URL 和 SUPABASE_ANON_KEY")
    st.stop()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"Supabase 连接失败: {e}")
    st.stop()


def _api(method: str, path: str, token: str | None = None, payload: dict[str, Any] | None = None):
    """调用 Supabase REST API"""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{SUPABASE_URL}/rest/v1/{path}"
    params = {}

    if method == "GET":
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    elif method == "POST":
        headers["Prefer"] = "return=minimal"
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
    elif method == "PUT":
        headers["Prefer"] = "return=minimal"
        resp = requests.put(url, headers=headers, json=payload, timeout=10)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, params=params, timeout=10)
    else:
        return False, {"error": "不支持的方法"}

    if resp.ok:
        return True, resp.json() if resp.text else {}
    return False, {"error": resp.text[:200]}


def sign_up(email: str, password: str, username: str):
    """注册"""
    try:
        auth = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"username": username}}
        })
        return True, auth
    except Exception as e:
        return False, {"error": str(e)}


def sign_in(email: str, password: str):
    """登录"""
    try:
        auth = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return True, auth
    except Exception as e:
        return False, {"error": str(e)}


def sign_out():
    """登出"""
    supabase.auth.sign_out()


def get_user():
    """获取当前用户"""
    try:
        session = supabase.auth.get_session()
        if session:
            return session.user
    except Exception:
        pass
    return None


def create_note(title: str, content: str, is_public: bool = False, tags: list | None = None):
    """创建笔记"""
    user = get_user()
    if not user:
        return False, {"error": "请先登录"}

    data = {
        "user_id": user.id,
        "title": title,
        "content": content,
        "is_public": is_public,
        "tags": tags or [],
        "created_at": datetime.now().isoformat()
    }

    result = supabase.table("notes").insert(data).execute()
    return True, result.data


def get_notes(mine_only: bool = False):
    """获取笔记列表"""
    user = get_user()
    query = supabase.table("notes").select("*").order("created_at", desc=True)

    if mine_only and user:
        query = query.eq("user_id", user.id)
    else:
        query = query.eq("is_public", True)

    result = query.execute()
    return result.data if result.data else []


def create_trade_plan(symbol: str, title: str, analysis: str, direction: str,
                      entry_price: float, stop_loss: float, take_profit: float,
                      confidence: int = 3):
    """创建交易计划"""
    user = get_user()
    if not user:
        return False, {"error": "请先登录"}

    # 计算盈亏比
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr = round(reward / risk, 2) if risk > 0 else 0

    data = {
        "user_id": user.id,
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
        "created_at": datetime.now().isoformat()
    }

    result = supabase.table("trade_plans").insert(data).execute()
    return True, result.data


def get_trade_plans(symbol: str | None = None):
    """获取交易计划列表"""
    query = supabase.table("trade_plans").select("*").eq("status", "ACTIVE").order("created_at", desc=True)

    if symbol:
        query = query.eq("symbol", symbol.upper())

    result = query.execute()
    return result.data if result.data else []


def like_plan(plan_id: str):
    """点赞交易计划"""
    user = get_user()
    if not user:
        return False, {"error": "请先登录"}

    data = {
        "user_id": user.id,
        "target_type": "TRADE_PLAN",
        "target_id": plan_id,
        "created_at": datetime.now().isoformat()
    }

    result = supabase.table("likes").insert(data).execute()
    return True, result.data


def get_boards():
    """获取群聊版块列表"""
    result = supabase.table("chat_boards").select("*").execute()
    return result.data if result.data else []


def get_messages(board_id: str):
    """获取版块消息"""
    result = supabase.table("chat_messages").select("*, users(username, avatar_url)") \
        .eq("board_id", board_id).order("created_at").execute()
    return result.data if result.data else []


def send_message(board_id: str, content: str):
    """发送消息"""
    user = get_user()
    if not user:
        return False, {"error": "请先登录"}

    data = {
        "board_id": board_id,
        "user_id": user.id,
        "content": content,
        "created_at": datetime.now().isoformat()
    }

    result = supabase.table("chat_messages").insert(data).execute()
    return True, result.data


# ========== UI ==========

st.set_page_config(page_title="StockAndCrypto Notes", page_icon="📈", layout="wide")

if "user" not in st.session_state:
    st.session_state["user"] = None
if "token" not in st.session_state:
    st.session_state["token"] = None

user = get_user()
if user:
    st.session_state["user"] = user

# 侧边栏导航
with st.sidebar:
    st.title("📈 StockAndCrypto")
    st.markdown("---")

    page = st.radio("导航", ["笔记", "交易计划", "群聊", "统计"])

    st.markdown("---")
    if st.session_state["user"]:
        st.write(f"👤 {user.email}")
        if st.button("退出登录"):
            sign_out()
            st.rerun()
    else:
        st.write("未登录")

# 主页面
if page == "笔记":
    st.title("📝 笔记")

    with st.expander("新建笔记", expanded=False):
        with st.form("new_note"):
            title = st.text_input("标题")
            content = st.text_area("内容", height=100)
            is_public = st.checkbox("公开分享")
            tags = st.text_input("标签 (逗号分隔)").split(",")
            tags = [t.strip() for t in tags if t.strip()]

            if st.form_submit_button("保存"):
                ok, result = create_note(title, content, is_public, tags)
                if ok:
                    st.success("笔记创建成功!")
                    st.rerun()
                else:
                    st.error(result.get("error", "创建失败"))

    st.markdown("### 我的笔记")
    notes = get_notes(mine_only=True)
    for note in notes:
        with st.expander(f"{'🔓 ' if note['is_public'] else '🔒 '} {note['title']}"):
            st.write(note["content"])
            st.caption(f"标签: {', '.join(note.get('tags', []))} | {note['created_at'][:10]}")

    st.markdown("### 公开笔记")
    public_notes = get_notes(mine_only=False)
    for note in public_notes:
        if note["user_id"] != st.session_state.get("user", {}).get("id"):
            with st.expander(f"👤 {note['title']}"):
                st.write(note["content"])

elif page == "交易计划":
    st.title("📊 交易计划")

    with st.expander("发布交易计划", expanded=False):
        with st.form("new_plan"):
            symbol = st.text_input("标的 (如 BTCUSD, AAPL)", placeholder="BTCUSD")
            title = st.text_input("计划标题")
            direction = st.selectbox("方向", ["LONG", "SHORT"])
            col1, col2, col3 = st.columns(3)
            entry_price = col1.number_input("入场价", min_value=0.0, format="%.2f")
            stop_loss = col2.number_input("止损", min_value=0.0, format="%.2f")
            take_profit = col3.number_input("止盈", min_value=0.0, format="%.2f")
            analysis = st.text_area("分析理由", height=100)
            confidence = st.slider("置信度", 1, 5, 3)

            if st.form_submit_button("发布"):
                ok, result = create_trade_plan(symbol, title, analysis, direction, entry_price, stop_loss, take_profit, confidence)
                if ok:
                    st.success("计划发布成功!")
                    st.rerun()
                else:
                    st.error(result.get("error", "发布失败"))

    st.markdown("### 热门计划")
    symbol_filter = st.text_input("筛选标的").upper()
    plans = get_trade_plans(symbol_filter if symbol_filter else None)

    for plan in plans:
        with st.container():
            st.markdown(f"""
            **{plan['symbol']}** | {plan['direction']} | 💰 {plan['entry_price']:.2f} → 🎯 {plan['take_profit']:.2f}
            - 止损: {plan['stop_loss']:.2f} | 盈亏比: {plan['risk_reward_ratio']} | ⭐ {plan['confidence_level']}/5
            """)
            st.caption(f"分析: {plan['analysis'][:200]}...")
            col1, col2 = st.columns([1, 8])
            if col1.button("👍 点赞", key=f"like_{plan['id']}"):
                like_plan(plan['id'])
                st.rerun()
            st.divider()

elif page == "群聊":
    st.title("💬 交易社区")

    boards = get_boards()
    board_names = [b["name"] for b in boards] if boards else ["BTC讨论区", "股票交流区", "外汇策略区"]

    selected_board = st.selectbox("选择版块", board_names)
    board_id = boards[board_names.index(selected_board)]["id"] if boards else None

    if board_id:
        messages = get_messages(board_id)
        for msg in messages:
            user_name = msg.get("users", {}).get("username", "匿名") if isinstance(msg.get("users"), dict) else "匿名"
            st.write(f"**{user_name}**: {msg['content']}")
            st.caption(msg['created_at'][:19])

        with st.form("send_msg"):
            content = st.text_input("发送消息")
            if st.form_submit_button("发送") and content:
                send_message(board_id, content)
                st.rerun()

elif page == "统计":
    st.title("📈 交易统计")

    plans = get_trade_plans()
    if plans:
        longs = [p for p in plans if p["direction"] == "LONG"]
        shorts = [p for p in plans if p["direction"] == "SHORT"]

        col1, col2, col3 = st.columns(3)
        col1.metric("多头计划", len(longs))
        col2.metric("空头计划", len(shorts))
        col3.metric("总计划", len(plans))

        st.bar_chart({"多头": len(longs), "空头": len(shorts)})
    else:
        st.info("暂无交易计划数据")

st.markdown("---")
st.caption("Powered by Streamlit + Supabase")
