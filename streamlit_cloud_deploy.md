# Streamlit Cloud 免费部署指南

## 方案架构

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Cloud                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │              StockAndCrypto App                    │  │
│  │  ┌─────────────┐  ┌─────────────────────────────┐ │  │
│  │  │  Notes UI   │  │  Supabase (前端认证+数据)    │ │  │
│  │  └─────────────┘  └─────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
        │
        │ HTTPS
        ▼
   Supabase (PostgreSQL + Auth)
```

## 免费额度

| 服务 | 免费额度 | 适合场景 |
|------|---------|---------|
| Streamlit Cloud | 1个私有app / 3个公开app | 前端托管 |
| Supabase | 500MB 数据库 + 50MB 文件 | 数据库 + 认证 |

## 部署步骤

### 1. 准备 Git 仓库

确保代码在 GitHub 上：

```bash
cd "/home/dream/文档/StockandCrypto"
git add .
git commit -m "Add notes feature for cloud deployment"
git push origin main
```

### 2. 登录 Streamlit Cloud

1. 访问 https://share.streamlit.io
2. 用 GitHub 账号登录
3. 点击 "New app"

### 3. 配置部署

| 配置项 | 值 |
|--------|-----|
| Repository | 你的 GitHub 用户名/StockandCrypto |
| Branch | main |
| Main file path | dashboard/notes_app.py |
| Python version | 3.11 |

### 4. 设置环境变量

在 Streamlit Cloud 设置页面添加：

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
JWT_SECRET_KEY=your-jwt-secret-key
```

### 5. 部署成功

访问 https://share.streamlit.io/你的用户名/stockandCrypto
