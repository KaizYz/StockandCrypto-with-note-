"""
StockAndCrypto Notes - 统一版 UI
支持两种模式:
- 本地开发: NOTES_API_URL=http://127.0.0.1:5001
- 云端部署: USE_SUPABASE=true + SUPABASE_URL + SUPABASE_ANON_KEY
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

# 导入统一配置
from notes_config import (
    sign_up, sign_in, sign_out, get_current_user, get_current_token,
    create_note, get_notes,
    create_trade_plan, get_trade_plans, like_plan,
    get_chat_boards, get_chat_messages, send_chat_message,
    USE_SUPABASE
)

st.set_page_config(page_title="StockAndCrypto Notes", page_icon="📈", layout="wide")

if "user" not in st.session_state:
    st.session_state["user"] = None
if "token" not in st.session_state:
    st.session_state["token"] = None

# 初始化用户
current_user = get_current_user()
if current_user:
    st.session_state["user"] = current_user

# ========== UI 组件 ==========

def login_panel():
    """登录/注册面板"""
    st.subheader("🔐 登录 / 注册")
    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        username = st.text_input("用户名 / 邮箱", key="login_user")
        password = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录", key="btn_login"):
            # 尝试用户名或邮箱登录
            if "@" in username:
                ok, data = sign_in(username, password)
            else:
                # 本地模式用 username，Supabase 用 email
                if USE_SUPABASE:
                    ok, data = sign_in(username, password)  # Supabase 用邮箱
                else:
                    ok, data = sign_in(username, password)

            if ok:
                st.session_state["user"] = get_current_user()
                st.session_state["token"] = get_current_token()
                st.success("登录成功!")
                st.rerun()
            else:
                st.error(data.get("error", "登录失败"))

    with register_tab:
        username = st.text_input("用户名", key="reg_user")
        email = st.text_input("邮箱", key="reg_email")
        password = st.text_input("密码", type="password", key="reg_pass")
        if st.button("注册", key="btn_reg"):
            ok, data = sign_up(email, password, username)
            if ok:
                st.success("注册成功! 请登录")
            else:
                st.error(data.get("error", "注册失败"))


def notes_page():
    """笔记页面"""
    st.title("📝 笔记")

    # 创建笔记
    with st.expander("新建笔记", expanded=False):
        with st.form("new_note"):
            title = st.text_input("标题")
            content = st.text_area("内容", height=120)
            col1, col2 = st.columns(2)
            is_public = col1.checkbox("公开分享", value=False)
            tags_raw = col2.text_input("标签 (逗号分隔)")
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

            if st.form_submit_button("保存"):
                ok, result = create_note(title, content, is_public, tags)
                if ok:
                    st.success("笔记创建成功!")
                    st.rerun()
                else:
                    st.error(result.get("error", "创建失败"))

    # 我的笔记
    st.markdown("### 我的笔记")
    notes = get_notes(mine_only=True)
    if not notes:
        st.info("还没有笔记")

    for note in notes:
        visible = "🔓" if note.get("is_public") else "🔒"
        with st.expander(f"{visible} {note.get('title', '无标题')}"):
            st.write(note.get("content", ""))
            tags = note.get("tags", [])
            if tags:
                st.caption(f"标签: {', '.join(tags) if isinstance(tags, list) else tags}")
            st.caption(f"更新时间: {note.get('updated_at', note.get('created_at', ''))[:10]}")

    # 公开笔记
    st.markdown("### 社区笔记")
    public_notes = get_notes(mine_only=False)
    user_id = str(st.session_state.get("user", {}).get("id", "")) if USE_SUPABASE else st.session_state.get("user", {}).get("id")

    for note in notes:
        note_user_id = str(note.get("user_id"))
        if note_user_id != str(user_id):
            with st.expander(f"👤 {note.get('title', '无标题')}"):
                st.write(note.get("content", ""))


def trade_plans_page():
    """交易计划页面"""
    st.title("📊 交易计划")

    # 发布计划
    with st.expander("发布交易计划", expanded=False):
        with st.form("new_plan"):
            col1, col2 = st.columns(2)
            symbol = col1.text_input("标的 (BTCUSD, AAPL)", placeholder="BTCUSD").upper()
            direction = col2.selectbox("方向", ["LONG", "SHORT"])

            col_a, col_b, col_c = st.columns(3)
            entry_price = col_a.number_input("入场价", min_value=0.0, format="%.2f")
            stop_loss = col_b.number_input("止损", min_value=0.0, format="%.2f")
            take_profit = col_c.number_input("止盈", min_value=0.0, format="%.2f")

            title = st.text_input("计划标题")
            analysis = st.text_area("分析理由", height=80)
            confidence = st.slider("置信度", 1, 5, 3)

            if st.form_submit_button("发布"):
                if not symbol or not title:
                    st.error("请填写标的和标题")
                else:
                    ok, result = create_trade_plan(
                        symbol, title, analysis, direction,
                        entry_price, stop_loss, take_profit, confidence
                    )
                    if ok:
                        st.success("计划发布成功!")
                        st.rerun()
                    else:
                        st.error(result.get("error", "发布失败"))

    # 筛选
    filter_symbol = st.text_input("筛选标的").upper()

    # 计划列表
    st.markdown("### 活跃计划")
    plans = get_trade_plans(symbol=filter_symbol if filter_symbol else None)

    if not plans:
        st.info("暂无交易计划")

    for plan in plans:
        symbol = plan.get("symbol", "")
        direction = plan.get("direction", "LONG")
        direction_emoji = "🟢" if direction == "LONG" else "🔴"

        st.markdown(f"""
        **{direction_emoji} {symbol}** | {plan.get('title', '')}
        - 入场: {plan.get('entry_price', 0):.2f} | 止损: {plan.get('stop_loss', 0):.2f} | 止盈: {plan.get('take_profit', 0):.2f}
        - 盈亏比: **{plan.get('risk_reward_ratio', 0)}** | 置信度: {"⭐" * plan.get('confidence_level', 0)}
        """)
        st.caption(f"分析: {plan.get('analysis', '')[:150]}...")

        col_like, _ = st.columns([1, 6])
        if col_like.button("👍 点赞", key=f"like_{plan.get('id')}"):
            like_plan(plan.get('id'))
            st.rerun()

        st.divider()


def chat_page():
    """群聊页面"""
    st.title("💬 交易社区")

    boards = get_chat_boards()
    board_names = [b["name"] for b in boards] if boards else ["BTC讨论区", "股票交流区", "外汇策略区"]

    if not boards:
        # 使用默认名称
        board_names = ["BTC讨论区", "股票交流区", "外汇策略区"]
        boards = [{"id": i+1, "name": name} for i, name in enumerate(board_names)]

    selected_idx = st.selectbox("选择版块", range(len(board_names)), format_func=lambda x: board_names[x])
    board = boards[selected_idx] if boards else {"id": selected_idx+1, "name": board_names[selected_idx]}
    board_id = board.get("id", selected_idx + 1)

    # 消息列表
    messages = get_chat_messages(board_id)
    for msg in messages:
        user_name = "匿名"
        if isinstance(msg.get("users"), dict):
            user_name = msg.get("users", {}).get("username", "匿名")
        elif USE_SUPABASE and isinstance(msg.get("profiles"), dict):
            user_name = msg.get("profiles", {}).get("username", "匿名")

        st.write(f"**{user_name}**: {msg.get('content', '')}")
        time_str = msg.get('created_at', '')
        if isinstance(time_str, str) and len(time_str) > 19:
            time_str = time_str[:19]
        st.caption(time_str)

    # 发送消息
    with st.form("send_msg"):
        content = st.text_input("消息内容", placeholder="说点什么...")
        if st.form_submit_button("发送"):
            if content:
                ok, _ = send_chat_message(board_id, content)
                if ok:
                    st.rerun()
                else:
                    st.error("发送失败，请先登录")


def stats_page():
    """统计页面"""
    st.title("📈 交易统计")

    plans = get_trade_plans()
    if plans:
        longs = [p for p in plans if p.get("direction") == "LONG"]
        shorts = [p for p in plans if p.get("direction") == "SHORT"]

        col1, col2, col3 = st.columns(3)
        col1.metric("多头计划", len(longs))
        col2.metric("空头计划", len(shorts))
        col3.metric("总计划", len(plans))

        st.bar_chart({"多头": len(longs), "空头": len(shorts)})

        # 按标的统计
        symbols = {}
        for p in plans:
            sym = p.get("symbol", "其他")
            symbols[sym] = symbols.get(sym, 0) + 1

        if symbols:
            st.subheader("标的分布")
            st.bar_chart(symbols)
    else:
        st.info("暂无交易计划数据")


# ========== 主应用 ==========

with st.sidebar:
    st.title("📈 StockAndCrypto")
    st.markdown("---")

    mode = "☁️ 云端 (Supabase)" if USE_SUPABASE else "🏠 本地 API"
    st.caption(f"运行模式: {mode}")

    page = st.radio("导航", ["笔记", "交易计划", "群聊", "统计"])

    st.markdown("---")

    # 用户信息
    if st.session_state["user"]:
        user = st.session_state["user"]
        if USE_SUPABASE:
            user_name = user.email if hasattr(user, 'email') else str(user).split('@')[0] if '@' in str(user) else "用户"
        else:
            user_name = user.get("username", "用户")
        st.write(f"👤 **{user_name}**")
        if st.button("退出登录"):
            sign_out()
            st.session_state["user"] = None
            st.session_state["token"] = None
            st.rerun()
    else:
        st.write("👤 **未登录**")

    st.markdown("---")
    st.caption("支持: 笔记 | 交易日记 | 交易计划 | 社区分享 | 群聊")

# 主页面
if not st.session_state["user"] and page != "统计":
    login_panel()
elif page == "笔记":
    notes_page()
elif page == "交易计划":
    trade_plans_page()
elif page == "群聊":
    chat_page()
elif page == "统计":
    stats_page()

st.markdown("---")
st.caption(f"StockAndCrypto Notes | 模式: {'Supabase Cloud' if USE_SUPABASE else 'Local API'}")
