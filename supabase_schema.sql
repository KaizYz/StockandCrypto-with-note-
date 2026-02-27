-- StockAndCrypto Supabase 数据库初始化脚本
-- 在 Supabase SQL Editor 中执行此脚本

-- 1. 用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT,
    avatar_url TEXT,
    bio TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. 笔记表
CREATE TABLE IF NOT EXISTS notes (
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

-- 3. 交易日记表
CREATE TABLE IF NOT EXISTS trading_journals (
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

-- 4. 交易计划表
CREATE TABLE IF NOT EXISTS trade_plans (
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
    is_public BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- 5. 点赞表
CREATE TABLE IF NOT EXISTS likes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    target_type VARCHAR(20) CHECK (target_type IN ('TRADE_PLAN', 'NOTE')),
    target_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, target_type, target_id)
);

-- 6. 群聊版块表
CREATE TABLE IF NOT EXISTS chat_boards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    topic VARCHAR(50),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 7. 群聊成员表
CREATE TABLE IF NOT EXISTS chat_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id UUID REFERENCES chat_boards(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'MEMBER',
    joined_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(board_id, user_id)
);

-- 8. 聊天消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id UUID REFERENCES chat_boards(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'TEXT',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_is_public ON notes(is_public);
CREATE INDEX IF NOT EXISTS idx_trading_journals_user_id ON trading_journals(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_plans_user_id ON trade_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_plans_symbol ON trade_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_likes_target ON likes(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_board_id ON chat_messages(board_id);

-- 插入默认群聊版块
INSERT INTO chat_boards (name, description, topic) VALUES
    ('Crypto Trading', 'BTC/ETH/SOL 等加密资产讨论', 'crypto'),
    ('Stock Trading', 'A股、美股与指数交易讨论', 'stock'),
    ('Forex Trading', '外汇行qing与交易策略', 'forex')
ON CONFLICT DO NOTHING;

SELECT '数据库初始化完成!';
