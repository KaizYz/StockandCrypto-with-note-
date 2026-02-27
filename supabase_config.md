# Supabase 配置指南

## 1. 创建 Supabase 项目

1. 访问 https://supabase.com 注册登录
2. 点击 "New Project"
3. 填写项目名称（如 stockandcrypto）
4. 生成并保存数据库密码
5. 等待项目创建完成（约1分钟）

## 2. 获取连接信息

在项目设置中获取：
- **Project URL**: https://xxx.supabase.co
- **anon public key**: xxx
- **service_role secret**: xxx（仅后端使用）

## 3. 创建数据库表

在 Supabase SQL Editor 中执行：

```sql
-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT,
    avatar_url TEXT,
    bio TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 笔记表
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    is_public BOOLEAN DEFAULT FALSE,
    is_trading_journal BOOLEAN DEFAULT FALSE,
    note_type VARCHAR(20) DEFAULT 'NOTE',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 交易日记表
CREATE TABLE trading_journals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    note_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10),
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    position_size DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    risk_reward_ratio DECIMAL(5, 2),
    pnl DECIMAL(20, 8),
    pnl_percent DECIMAL(10, 4),
    status VARCHAR(20) DEFAULT 'OPEN',
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    trade_reason TEXT,
    screenshot_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 交易计划表
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

-- 点赞表
CREATE TABLE likes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    target_type VARCHAR(20) CHECK (target_type IN ('TRADE_PLAN', 'NOTE')),
    target_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, target_type, target_id)
);

-- 群聊版块表
CREATE TABLE chat_boards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    topic VARCHAR(50),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 群聊成员表
CREATE TABLE chat_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id UUID REFERENCES chat_boards(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'MEMBER',
    joined_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(board_id, user_id)
);

-- 聊天消息表
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id UUID REFERENCES chat_boards(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'TEXT',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_notes_user_id ON notes(user_id);
CREATE INDEX idx_notes_is_public ON notes(is_public);
CREATE INDEX idx_trading_journals_user_id ON trading_journals(user_id);
CREATE INDEX idx_trade_plans_user_id ON trade_plans(user_id);
CREATE INDEX idx_likes_target ON likes(target_type, target_id);
CREATE INDEX idx_chat_messages_board_id ON chat_messages(board_id);
```

## 4. 启用 API

在 Supabase 中：
1. API → Table & View → 勾选所有表
2. 启用 Row Level Security (RLS)
3. 配置允许匿名访问的策略

## 5. 环境变量配置

创建 `.env` 文件：

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# JWT
JWT_SECRET_KEY=your-jwt-secret-key-min-32-chars

# API (Streamlit Cloud 会自动设置)
NOTES_API_URL=https://your-backend.herokuapp.com
```
