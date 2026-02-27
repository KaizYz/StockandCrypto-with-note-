# StockandCrypto Notes 功能实现计划

## 1. 功能概述

为 StockandCrypto 项目添加 Notes 模块，包含：
- 📝 普通笔记功能
- 📔 交易日记
- 🔗 分享交易计划
- ❤️ 点赞系统
- 💬 版块群聊

## 2. 技术选型

### 2.1 后端
- **框架**: Python Flask / FastAPI
- **数据库**: SQLite (MVP) / PostgreSQL (生产)
- **实时通信**: WebSocket (群聊)
- **认证**: JWT

### 2.2 前端
- **框架**: Streamlit (与现有 Dashboard 保持一致)
- **状态管理**: Streamlit Session State

### 2.3 数据存储结构

```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    avatar_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 笔记表
CREATE TABLE notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title VARCHAR(200),
    content TEXT,
    note_type VARCHAR(20) DEFAULT 'normal', -- 'normal', 'journal', 'plan'
    is_public BOOLEAN DEFAULT FALSE,
    market_type VARCHAR(20), -- 'crypto', 'stock', 'all'
    symbols TEXT, -- JSON array of symbols
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 点赞表
CREATE TABLE likes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(note_id, user_id),
    FOREIGN KEY (note_id) REFERENCES notes(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 板块表
CREATE TABLE boards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    market_type VARCHAR(20), -- 'crypto', 'stock', 'all'
    created_by INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- 群聊消息表
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'text', -- 'text', 'plan_share'
    related_note_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (board_id) REFERENCES boards(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (related_note_id) REFERENCES notes(id)
);

-- 消息表情 reactions
CREATE TABLE reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    emoji VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(message_id, user_id, emoji),
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

## 3. API 设计

### 3.1 笔记相关
```
POST   /api/notes/create          # 创建笔记
GET    /api/notes/list            # 列出笔记
GET    /api/notes/<id>            # 获取笔记详情
PUT    /api/notes/<id>            # 更新笔记
DELETE /api/notes/<id>            # 删除笔记
POST   /api/notes/<id>/like       # 点赞/取消点赞
GET    /api/notes/<id>/likes      # 获取点赞列表
POST   /api/notes/<id>/share      # 分享笔记
GET    /api/notes/public          # 获取公开笔记
GET    /api/notes/search          # 搜索笔记
```

### 3.2 板块群聊相关
```
POST   /api/boards/create         # 创建板块
GET    /api/boards/list           # 列出板块
GET    /api/boards/<id>           # 获取板块详情
POST   /api/boards/<id>/join      # 加入板块
POST   /api/boards/<id>/leave     # 离开板块

GET    /api/boards/<id>/messages  # 获取消息历史
POST   /api/boards/<id>/messages  # 发送消息
DELETE /api/boards/messages/<id>  # 删除消息

POST   /api/messages/<id>/react   # 添加表情反应
DELETE /api/messages/<id>/react   # 移除表情反应
```

### 3.3 用户相关
```
POST   /api/auth/register         # 注册
POST   /api/auth/login            # 登录
GET    /api/auth/me               # 获取当前用户
PUT    /api/auth/profile          # 更新个人资料
```

## 4. Streamlit 页面结构

### 4.1 页面导航
```
📝 Notes
├── 📓 我的笔记 (My Notes)
│   ├── 全部
│   ├── 交易日记 (Trading Journal)
│   └── 交易计划 (Trading Plans)
│
├── 🌍 发现 (Discover)
│   ├── 公开笔记
│   ├── 热门计划
│   └── 用户列表
│
├── 💬 板块群聊 (Boards)
│   ├── 加密货币讨论
│   ├── A股交流
│   └── 美股专区
│
└── 👤 个人中心 (Profile)
    ├── 我的资料
    └── 设置
```

### 4.2 主要组件

#### 笔记编辑器组件
- Markdown 编辑器
- 标签添加 (支持市场、符号)
- 交易计划模板
- 保存/发布按钮

#### 交易日记模板
```markdown
# 交易日记 - [日期]

## 交易标的
- 品种: BTC/USDT
- 方向: Long/Short
- 入场价: [价格]
- 止损: [价格]
- 止盈: [价格]

## 交易理由
- 技术面: ...
- 基本面: ...
- 风险管理: ...

## 执行情况
- 执行状态: 已执行/未执行
- 复盘: ...

## 仓位信息
- 仓位: [百分比]
- 杠杆: [倍数]
- 风险预算: [金额]
```

#### 交易计划分享模板
```markdown
# 交易计划分享

## 计划概述
- 标题: [标题]
- 标的: [BTC/ETH/股票代码]
- 类型: 现货/合约
- 预计时长: [短/中/长线]

## 分析
- 技术面分析
- 基本面分析
- 风险评估

## 交易规则
- 入场条件
- 止损规则
- 止盈规则
- 仓位管理

## 预期收益
- 预期收益: [百分比]
- 最大风险: [百分比]
```

#### 点赞卡片组件
- 显示点赞数
- 点赞按钮
- 点赞用户列表

#### 群聊组件
- 消息列表
- 输入框
- 表情 reaction
- 分享交易计划按钮

## 5. 实现步骤

### 步骤 1: 数据库和后端基础 (Day 1)
- [ ] 创建数据库表结构
- [ ] 实现用户认证 (JWT)
- [ ] 实现笔记 CRUD API
- [ ] 实现点赞功能 API

### 步骤 2: 群聊功能 (Day 2)
- [ ] 实现 WebSocket 实时通信
- [ ] 实现板块管理 API
- [ ] 实现消息发送/获取 API
- [ ] 实现表情 reaction 功能

### 步骤 3: Streamlit 前端 - 笔记 (Day 3-4)
- [ ] 笔记列表页面
- [ ] 笔记编辑器
- [ ] 交易日记模板
- [ ] 交易计划模板
- [ ] 公开笔记页面
- [ ] 点赞交互

### 步骤 4: Streamlit 前端 - 群聊 (Day 5)
- [ ] 板块列表页面
- [ ] 群聊页面
- [ ] 消息实时更新
- [ ] 表情 reaction UI
- [ ] 分享交易计划到群聊

### 步骤 5: 用户界面优化 (Day 6)
- [ ] 个人中心页面
- [ ] 设置页面
- [ ] 响应式布局
- [ ] 深色/浅色主题

### 步骤 6: 测试和部署 (Day 7)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 文档编写
- [ ] 部署配置

## 6. 文件结构

```
StockandCrypto/
├── src/
│   └── notes/
│       ├── __init__.py
│       ├── app.py                  # Flask/FastAPI 主应用
│       ├── models.py               # 数据库模型
│       ├── auth.py                 # 认证相关
│       ├── notes_api.py            # 笔记 API
│       ├── boards_api.py           # 板块 API
│       ├── messages_api.py         # 消息 API
│       ├── websocket.py            # WebSocket 处理
│       └── utils.py                # 工具函数
│
├── dashboard/
│   ├── notes_pages.py              # 笔记相关页面
│   ├── journal_page.py             # 交易日记页面
│   ├── plans_page.py               # 交易计划页面
│   ├── boards_page.py              # 板块群聊页面
│   ├── profile_page.py             # 个人中心页面
│   ├── components/
│   │   ├── note_editor.py          # 笔记编辑器
│   │   ├── like_button.py          # 点赞按钮
│   │   ├── chat_component.py       # 群聊组件
│   │   └── templates/
│   │       ├── journal_template.py # 日记模板
│   │       └── plan_template.py    # 计划模板
│   └── notes_auth.py               # 页面认证
│
├── data/
│   └── notes.db                    # SQLite 数据库
│
├── requirements-notes.txt          # 新增依赖
│
└── scripts/
    └── init_notes_db.py            # 数据库初始化脚本
```

## 7. 依赖包

```
# 后端
Flask==3.0.0
Flask-SocketIO==5.3.6
Flask-SQLAlchemy==3.1.1
Flask-JWT-Extended==4.6.0
Werkzeug==3.0.1
SQLAlchemy==2.0.23

# 前端 (Streamlit 已有)
streamlit==1.31.0
```

## 8. 安全考虑

- JWT Token 过期时间: 24 小时
- 密码哈希: bcrypt
- API 速率限制: 100 请求/分钟
- 输入验证和消毒
- CORS 配置

## 9. 后续扩展

- [ ] WebSocket 重连机制
- [ ] 消息加密
- [ ] 群聊房间管理员
- [ ] 笔记版本历史
- [ ] 数据导出功能
- [ ] 推送通知

## 10. 成功指标

- ✅ 用户可以创建和编辑笔记
- ✅ 用户可以写交易日记
- ✅ 用户可以分享交易计划
- ✅ 用户可以点赞/取消点赞
- ✅ 用户可以加入板块群聊
- ✅ 群聊支持实时消息
- ✅ 消息支持表情 reaction
- ✅ 可以分享笔记到群聊

---

*创建日期: 2026-02-25*
