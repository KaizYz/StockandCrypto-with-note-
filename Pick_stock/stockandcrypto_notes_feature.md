# StockAndCrypto - Notes 功能详细规划

## 一、需求文档

### 1.1 用户故事

| 编号 | 角色 | 场景 | 价值 |
|------|------|------|------|
| US-01 | 普通用户 | 作为普通用户，我想创建私人笔记 | 记录交易心得 |
| US-02 | 交易者 | 作为交易者，我想写交易日记（包含入场/出场点、止损、盈亏比） | 复盘交易表现 |
| US-03 | 分享者 | 作为交易者，我想分享交易计划给其他人 | 获取反馈和认可 |
| US-04 | 浏览者 | 作为浏览者，我想看别人的交易计划并点赞 | 发现优秀交易思路 |
| US-05 | 群聊参与者 | 作为交易者，我想加入某个版块的群聊讨论 | 与同好交流 |

### 1.2 功能模块

#### 1.2.1 笔记模块 (Notes)
- 创建笔记（标题、内容、标签）
- 编辑笔记
- 删除笔记
- 笔记列表（按时间/标签筛选）
- 搜索笔记
- 设为私密/公开
- Markdown 支持

#### 1.2.2 交易日记模块 (Trading Journal)
- 创建交易记录：
  - 交易标的 (symbol)
  - 方向 (做多/做空)
  - 入场价格 (entry_price)
  - 出场价格 (exit_price)
  - 仓位大小 (position_size)
  - 止损 (stop_loss)
  - 止盈 (take_profit)
  - 盈亏比 (risk_reward_ratio)
  - 交易日期/时间
  - 交易理由/分析
  - 截图附件
- 交易统计：
  - 胜率
  - 总盈亏
  - 最大回撤
  - 平均盈亏比
  - 月度/周度统计
- 交易曲线图表

#### 1.2.3 分享模块 (Share & Like)
- 分享交易计划到社区
- 公开/私密设置
- 点赞/取消点赞
- 点赞数统计
- 点赞用户列表
- 分享链接复制

#### 1.2.4 群聊模块 (Group Chat)
- 创建群聊版块（按品种：Crypto, Stock, Forex...）
- 加入/退出版块
- 实时发送消息
- 消息历史
- @提及用户
- 群成员列表

---

## 二、数据库设计 (Schema)

### 2.1 数据库选择
- **主数据库**: PostgreSQL (关系型，适合结构化数据)
- **缓存**: Redis (会话、实时消息)
- **文件存储**: 本地存储或 S3 (截图、附件)

### 2.2 表结构

#### users 用户表
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    avatar_url TEXT,
    bio TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### notes 笔记表
```sql
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    is_public BOOLEAN DEFAULT FALSE,
    is_trading_journal BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### trading_journals 交易日记表
```sql
CREATE TABLE trading_journals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    note_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,           -- e.g., BTCUSD, AAPL
    direction VARCHAR(10) CHECK (direction IN ('LONG', 'SHORT')),
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    position_size DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    risk_reward_ratio DECIMAL(5, 2),
    pnl DECIMAL(20, 8),                    -- 盈亏金额
    pnl_percent DECIMAL(10, 4),            -- 盈亏百分比
    status VARCHAR(20) CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    trade_reason TEXT,
    screenshot_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### trade_plans 交易计划表
```sql
CREATE TABLE trade_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    note_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    analysis TEXT NOT NULL,
    entry_zone DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    risk_reward_ratio DECIMAL(5, 2),
    confidence_level INTEGER CHECK (confidence_level BETWEEN 1 AND 5),
    status VARCHAR(20) DEFAULT 'ACTIVE',
    views INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);
```

#### likes 点赞表
```sql
CREATE TABLE likes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    target_type VARCHAR(20) CHECK (target_type IN ('TRADE_PLAN', 'NOTE')),
    target_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, target_type, target_id)
);
```

#### chat_版块s 群聊版块表
```sql
CREATE TABLE chat_版块s (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,            -- e.g., "BTC交易讨论区"
    description TEXT,
    topic VARCHAR(50),                     -- e.g., "crypto", "stock", "forex"
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### chat_messages 聊天消息表
```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    版块_id UUID REFERENCES chat_版块s(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'TEXT', -- TEXT, IMAGE, FILE
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### chat_members 群聊成员表
```sql
CREATE TABLE chat_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    版块_id UUID REFERENCES chat_版块s(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'MEMBER',      -- ADMIN, MEMBER
    joined_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(版块_id, user_id)
);
```

---

## 三、API 接口设计

### 3.1 认证相关
```
POST   /api/auth/register        # 注册
POST   /api/auth/login           # 登录
POST   /api/auth/logout          # 登出
GET    /api/auth/me              # 获取当前用户
```

### 3.2 笔记 API
```
# 笔记 CRUD
GET    /api/notes                # 获取笔记列表
POST   /api/notes                # 创建笔记
GET    /api/notes/:id            # 获取笔记详情
PUT    /api/notes/:id            # 更新笔记
DELETE /api/notes/:id            # 删除笔记

# 交易日记
POST   /api/notes/journal        # 创建交易日记
GET    /api/notes/journal/:id    # 获取交易日记详情
GET    /api/users/:id/journals   # 获取用户的交易记录

# 分享
POST   /api/notes/:id/publish    # 公开分享笔记
POST   /api/notes/:id/unpublish  # 取消分享
GET    /api/community/notes      # 获取社区公开笔记
```

### 3.3 交易计划 API
```
POST   /api/trade-plans          # 创建交易计划
GET    /api/trade-plans          # 获取交易计划列表
GET    /api/trade-plans/:id      # 获取交易计划详情
PUT    /api/trade-plans/:id      # 更新交易计划
DELETE /api/trade-plans/:id      # 删除交易计划

# 点赞
POST   /api/trade-plans/:id/like # 点赞
DELETE /api/trade-plans/:id/like # 取消点赞
GET    /api/trade-plans/:id/likes # 获取点赞列表
```

### 3.4 群聊 API
```
# 版块管理
GET    /api/chat/版块s           # 获取版块列表
POST   /api/chat/版块s           # 创建版块
GET    /api/chat/版块s/:id       # 获取版块详情

# 成员管理
POST   /api/chat/版块s/:id/join  # 加入版块
POST   /api/chat/版块s/:id/leave # 离开版块
GET    /api/chat/版块s/:id/members # 获取成员列表

# 消息
GET    /api/chat/版块s/:id/messages     # 获取消息历史
POST   /api/chat/版块s/:id/messages     # 发送消息
WS     /api/chat/版块s/:id/ws           # WebSocket 连接

# 统计
GET    /api/chat/版块s/:id/stats        # 版块统计
```

---

## 四、前端页面结构

```
src/
├── pages/
│   ├── Notes/
│   │   ├── NotesList.tsx          # 笔记列表
│   │   ├── NotesDetail.tsx        # 笔记详情
│   │   ├── NotesEditor.tsx        # 笔记编辑器
│   │   ├── JournalList.tsx        # 交易日记列表
│   │   ├── JournalEditor.tsx      # 交易日记编辑器
│   │   └── JournalStats.tsx       # 交易统计
│   ├── Community/
│   │   ├── TradePlans.tsx         # 交易计划社区
│   │   ├── TradePlanDetail.tsx    # 交易计划详情
│   │   └── LikesList.tsx          # 点赞列表
│   └── Chat/
│       ├── ChatList.tsx           # 版块列表
│       ├── ChatRoom.tsx           # 聊天室
│       └── ChatSidebar.tsx        # 侧边栏
├── components/
│   ├── Notes/
│   │   ├── NoteCard.tsx
│   │   ├── JournalCard.tsx
│   │   ├── TagInput.tsx
│   │   └── MarkdownEditor.tsx
│   ├── Trading/
│   │   ├── TradeForm.tsx
│   │   ├── TradeStats.tsx
│   │   └── PnLChart.tsx
│   ├── Chat/
│   │   ├── MessageList.tsx
│   │   ├── MessageInput.tsx
│   │   └── MemberList.tsx
│   └── Shared/
│       ├── LikeButton.tsx
│       ├── ShareButton.tsx
│       └── Modal.tsx
└── hooks/
    ├── useNotes.ts
    ├── useTradePlan.ts
    ├── useChat.ts
    └── useAuth.ts
```

---

## 五、用户界面原型描述

### 5.1 笔记列表页面
- 顶部：搜索栏 + 新建笔记按钮
- 左侧：标签筛选、笔记本分类
- 右侧：笔记卡片网格
  - 卡片显示：标题、标签、最后更新时间
  - 交易日记卡片额外显示：胜率、总盈亏

### 5.2 交易日记编辑器
- 标的输入（下拉选择或搜索）
- 方向选择（多/空）
- 价格输入（入场、出场、止损、止盈）
- 自动计算盈亏比
- 备注区域（Markdown）
- 上传截图

### 5.3 交易计划社区
- 卡片流布局
- 每个卡片显示：
  - 用户头像 + 名称
  - 标的、方向
  - 点赞数
  - 置信度星级
- 筛选：按标的、按时间、按热度

### 5.4 版块聊天室
- 左侧：版块列表
- 中间：消息流
- 右侧：在线成员列表
- 底部：输入框 + 发送按钮

---

## 六、开发优先级 (MVP)

### 第一阶段 (Must Have)
1. 笔记 CRUD
2. 交易日记记录
3. 交易统计
4. 交易计划分享
5. 点赞功能

### 第二阶段 (Should Have)
1. 群聊版块
2. 实时消息 (WebSocket)
3. 搜索功能
4. 消息通知

### 第三阶段 (Nice to Have)
1. 交易曲线图表
2. 社区评论
3. 私信功能
4. 数据导出

---

## 七、输出要求

请生成：
1. ✅ 完整的数据库 Schema (DDL)
2. ✅ API 接口文档 (OpenAPI/Swagger 格式或详细说明)
3. ✅ 后端核心代码 (路由、控制器、模型)
4. ✅ 前端页面代码 (React 组件)
5. ✅ 数据库初始化脚本 (seed data)

技术栈：前端使用 React + TypeScript，后端使用 Node.js/Express + PostgreSQL。
