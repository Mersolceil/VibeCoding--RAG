# BUPT-RAG: 多格式文档 AI 智能助手

这是一个基于 DeepSeek 大模型和本地 Embedding 技术的 RAG（检索增强生成）系统。

### 🌟 功能亮点
- **全格式支持**: 支持 PDF, DOCX, C, PY, TXT 等多种格式。
- **本地化加速**: 使用 `sentence-transformers` 在本地进行文本向量化，保护隐私。
- **交互式界面**: 基于 Streamlit 构建，支持对话历史记忆。
- **适配 Windows**: 针对 C 盘空间不足和代理网络环境进行了深度优化。

### 🛠️ 快速开始
1. 克隆项目：`git clone [你的仓库链接]`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行：`streamlit run web_app.py`