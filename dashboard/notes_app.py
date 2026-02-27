"""Notes 模块的最小可用 Streamlit 页面。"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("NOTES_API_URL", "http://127.0.0.1:5001").rstrip("/")


def _api(method: str, path: str, token: str | None = None, payload: dict[str, Any] | None = None):
    url = f"{API_BASE}{path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.request(method=method, url=url, headers=headers, json=payload, timeout=10)
    except requests.RequestException as exc:
        return False, {"error": f"请求失败: {exc}"}
    try:
        data = resp.json()
    except ValueError:
        data = {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    if resp.ok:
        return True, data
    return False, data


def _login_panel():
    st.subheader("登录 / 注册")
    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        username = st.text_input("用户名", key="login_username")
        password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录", key="btn_login"):
            ok, data = _api("POST", "/api/auth/login", payload={"username": username, "password": password})
            if ok:
                st.session_state["token"] = data.get("token")
                st.session_state["user"] = data.get("user")
                st.success("登录成功")
                st.rerun()
            else:
                st.error(data.get("error", "登录失败"))

    with register_tab:
        username = st.text_input("新用户名", key="reg_username")
        email = st.text_input("邮箱", key="reg_email")
        password = st.text_input("新密码", type="password", key="reg_password")
        if st.button("注册", key="btn_register"):
            ok, data = _api(
                "POST",
                "/api/auth/register",
                payload={"username": username, "email": email, "password": password},
            )
            if ok:
                st.success("注册成功，请回到登录页登录")
            else:
                st.error(data.get("error", "注册失败"))


def _notes_page(token: str):
    st.subheader("我的笔记")
    with st.form("create_note"):
        title = st.text_input("标题")
        content = st.text_area("内容", height=140)
        tags = st.text_input("标签（逗号分隔）")
        is_public = st.checkbox("公开到社区", value=False)
        submitted = st.form_submit_button("创建笔记")
    if submitted:
        ok, data = _api(
            "POST",
            "/api/notes",
            token=token,
            payload={
                "title": title,
                "content": content,
                "tags": tags,
                "is_public": is_public,
                "note_type": "NOTE",
            },
        )
        if ok:
            st.success("创建成功")
        else:
            st.error(data.get("error", "创建失败"))

    ok, data = _api("GET", "/api/notes?mine=true&page_size=50", token=token)
    if not ok:
        st.error(data.get("error", "读取笔记失败"))
        return
    items = data.get("items", [])
    if not items:
        st.info("还没有笔记")
    for item in items:
        with st.expander(f"{item.get('title')}  · {'公开' if item.get('is_public') else '私密'}"):
            st.write(item.get("content", ""))
            tags = item.get("tags") or []
            if tags:
                st.caption("标签: " + ", ".join(tags))
            col1, col2 = st.columns(2)
            if col1.button("公开", key=f"pub_{item['id']}"):
                _api("POST", f"/api/notes/{item['id']}/publish", token=token)
                st.rerun()
            if col2.button("取消公开", key=f"unpub_{item['id']}"):
                _api("POST", f"/api/notes/{item['id']}/unpublish", token=token)
                st.rerun()


def _community_page(token: str):
    st.subheader("社区公开内容")
    ok, notes_data = _api("GET", "/api/community/notes?page_size=20", token=token)
    if not ok:
        st.error(notes_data.get("error", "读取社区笔记失败"))
    else:
        for row in notes_data.get("items", []):
            st.markdown(f"**{row.get('title', '')}** · @{row.get('username', '-')}")
            st.write(row.get("content", ""))
            st.caption(f"点赞: {row.get('like_count', 0)}")
            st.divider()

    st.subheader("交易计划")
    ok, plans_data = _api("GET", "/api/trade-plans?page_size=20", token=token)
    if not ok:
        st.error(plans_data.get("error", "读取交易计划失败"))
        return
    for plan in plans_data.get("items", []):
        st.markdown(f"**{plan.get('title', '')}** ({plan.get('symbol', '-')})")
        st.write(plan.get("analysis", ""))
        st.caption(
            f"置信度: {plan.get('confidence_level')} | 点赞: {plan.get('like_count', 0)} | 浏览: {plan.get('views', 0)}"
        )
        if st.button("点赞计划", key=f"plan_like_{plan['id']}"):
            ok_like, like_data = _api("POST", f"/api/trade-plans/{plan['id']}/like", token=token)
            if ok_like:
                st.success("已点赞")
            else:
                st.error(like_data.get("error", "点赞失败"))
        st.divider()


def _chat_page(token: str):
    st.subheader("版块聊天")
    ok, data = _api("GET", "/api/chat/boards", token=token)
    if not ok:
        st.error(data.get("error", "读取版块失败"))
        return
    boards = data.get("items", [])
    if not boards:
        st.info("暂无版块")
        return
    board_map = {f"{b['name']} (#{b['id']})": b for b in boards}
    selected_label = st.selectbox("选择版块", options=list(board_map.keys()))
    board = board_map[selected_label]

    col1, col2 = st.columns(2)
    if col1.button("加入版块"):
        _api("POST", f"/api/chat/boards/{board['id']}/join", token=token)
    if col2.button("离开版块"):
        _api("POST", f"/api/chat/boards/{board['id']}/leave", token=token)

    ok_msg, msg_data = _api("GET", f"/api/chat/boards/{board['id']}/messages?limit=50", token=token)
    if not ok_msg:
        st.error(msg_data.get("error", "读取消息失败"))
    else:
        for msg in msg_data.get("items", []):
            st.markdown(f"**@{msg.get('username', '-')}:** {msg.get('content', '')}")

    content = st.text_input("发送消息", key=f"board_input_{board['id']}")
    if st.button("发送", key=f"send_btn_{board['id']}"):
        ok_send, send_data = _api(
            "POST",
            f"/api/chat/boards/{board['id']}/messages",
            token=token,
            payload={"content": content},
        )
        if ok_send:
            st.rerun()
        else:
            st.error(send_data.get("error", "发送失败"))


def main():
    st.set_page_config(page_title="StockandCrypto Notes", page_icon="📝", layout="wide")
    st.title("📝 StockandCrypto Notes")
    st.caption(f"API: {API_BASE}")

    token = st.session_state.get("token")
    user = st.session_state.get("user")

    if not token:
        _login_panel()
        return

    st.sidebar.success(f"当前用户: {user.get('username') if isinstance(user, dict) else '-'}")
    if st.sidebar.button("退出登录"):
        st.session_state.pop("token", None)
        st.session_state.pop("user", None)
        st.rerun()

    menu = st.sidebar.radio("导航", options=["我的笔记", "社区", "版块聊天"])
    if menu == "我的笔记":
        _notes_page(token)
    elif menu == "社区":
        _community_page(token)
    else:
        _chat_page(token)


if __name__ == "__main__":
    main()
