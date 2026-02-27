# StockandCrypto Notes 模块

## 快速开始

1. 安装依赖
```bash
pip install -r requirements-notes.txt
```

2. 初始化数据库（含 seed）
```bash
python scripts/init_notes_db.py --reset
```

3. 启动后端服务
```bash
python -m src.notes.app
```
默认地址：`http://127.0.0.1:5001`

4. 启动 Streamlit 页面
```bash
streamlit run dashboard/notes_app.py
```
默认地址：`http://127.0.0.1:8501`

## 默认测试账号

- `demo / demo123456`
- `guest / guest123456`

## 主要 API

### 认证
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `PUT /api/auth/profile`

### 笔记与交易日记
- `POST /api/notes`
- `GET /api/notes`
- `GET /api/notes/:id`
- `PUT /api/notes/:id`
- `DELETE /api/notes/:id`
- `POST /api/notes/:id/publish`
- `POST /api/notes/:id/unpublish`
- `GET /api/community/notes`
- `POST /api/notes/journal`
- `GET /api/notes/journal/:id`
- `GET /api/users/:id/journals`

### 交易计划与点赞
- `POST /api/trade-plans`
- `GET /api/trade-plans`
- `GET /api/trade-plans/:id`
- `PUT /api/trade-plans/:id`
- `DELETE /api/trade-plans/:id`
- `POST /api/trade-plans/:id/like`
- `DELETE /api/trade-plans/:id/like`
- `GET /api/trade-plans/:id/likes`
- `POST /api/notes/:id/like`
- `DELETE /api/notes/:id/like`
- `GET /api/notes/:id/likes`

### 群聊版块
- `GET /api/chat/boards`
- `POST /api/chat/boards`
- `GET /api/chat/boards/:id`
- `POST /api/chat/boards/:id/join`
- `POST /api/chat/boards/:id/leave`
- `GET /api/chat/boards/:id/members`
- `GET /api/chat/boards/:id/messages`
- `POST /api/chat/boards/:id/messages`
- `GET /api/chat/boards/:id/stats`
