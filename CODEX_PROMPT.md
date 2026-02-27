# Codex Prompt - StockandCrypto Notes Feature

请实现 StockandCrypto 项目的 Notes 功能模块。

## 项目位置
`/home/dream/文档/StockandCrypto`

## 功能需求
1. **笔记功能**
   - 普通笔记（Markdown 编辑）
   - 交易日记（带模板：标的、方向、入场、止损、止盈、仓位等）
   - 交易计划（可分享给他人）
   - CRUD 操作
   - 公开/私密设置

2. **点赞系统**
   - 点赞/取消点赞
   - 显示点赞数

3. **板块群聊**
   - 预设板块：加密货币、A股、美股、综合交易等
   - 实时消息（WebSocket）
   - 表情 reaction（👍❤️😂😮😢）
   - 分享交易计划到群聊

## 技术栈
- **后端**: Flask + Flask-SocketIO + Flask-SQLAlchemy + JWT
- **数据库**: SQLite（可扩展为 PostgreSQL）
- **前端**: Streamlit（与现有 Dashboard 保持一致）
- **文件位置**: `/home/dream/文档/StockandCrypto/src/notes/`

## 参考文档
- 功能计划：`/home/dream/文档/StockandCrypto/notes_feature_plan.md`
- 数据库模型：`/home/dream/文档/StockandCrypto/src/notes/models.py`（已创建基础结构）

## 需要创建的文件

```
src/notes/
├── __init__.py
├── app.py              # Flask 主应用 + WebSocket
├── auth.py             # JWT 认证
├── notes_api.py        # 笔记 API
├── boards_api.py       # 板块 API
├── messages_api.py     # 消息 API
└── utils.py            # 工具函数

dashboard/notes/
├── notes_app.py        # Streamlit 主入口
├── pages/
│   ├── 1_我的笔记.py
│   ├── 2_发现.py
│   └── 3_群聊.py
├── components/
│   ├── note_editor.py
│   ├── journal_template.py
│   ├── plan_template.py
│   ├── like_button.py
│   └── chat_component.py
└── auth.py

scripts/
└── init_notes_db.py
```

## 开始实现

请依次创建以下文件，并确保：
1. 数据库初始化脚本能正常工作
2. API 能通过 Postman/curl 测试
3. Streamlit 页面能正常显示和交互
