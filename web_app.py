import streamlit as st
import os
# 导入核心组件
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 自动化环境配置 (北邮 CS 专属优化版) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# 强制模型和缓存下载到 D 盘当前目录，保护 C 盘
os.environ["HF_HOME"] = os.path.join(current_dir, "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="BUPT 智能文档助手", layout="wide", page_icon="🤖")

# 初始化 Session State，确保刷新网页不丢数据
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# --- 2. 侧边栏界面 ---
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("输入 DeepSeek API Key", type="password", help="从 deepseek.com 获取")
    uploaded_files = st.file_uploader(
        "上传文档 (支持 PDF, DOCX, C, PY, TXT 等)",
        accept_multiple_files=True
    )
    process_button = st.button("🚀 开始分析文档", use_container_width=True)

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# --- 3. 文档处理逻辑 ---
if process_button:
    if not api_key:
        st.error("❌ 请先输入 API Key！")
    elif not uploaded_files:
        st.error("❌ 请先上传至少一个文档！")
    else:
        with st.spinner("🧠 正在读取并建立索引，首次运行可能需要下载模型..."):
            try:
                # 初始化 LLM
                st.session_state.llm = ChatOpenAI(
                    model='deepseek-chat',
                    openai_api_key=api_key,
                    openai_api_base="https://api.deepseek.com/v1"
                )

                # 初始化 Embedding (本地 D 盘加载)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                all_docs = []
                for uploaded_file in uploaded_files:
                    # 写入临时文件供 Loader 读取
                    temp_path = os.path.join(current_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        # 分流处理：PDF 用 PyPDF，其余用 Unstructured
                        if uploaded_file.name.lower().endswith('.pdf'):
                            loader = PyPDFLoader(temp_path)
                        else:
                            loader = UnstructuredFileLoader(temp_path)
                        all_docs.extend(loader.load())
                    except Exception as e:
                        st.warning(f"⚠️ 文件 {uploaded_file.name} 解析失败: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                # 切分文档
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(all_docs)

                # 构建向量库并存入 Session
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings
                )
                st.success(f"✅ 成功索引 {len(uploaded_files)} 个文档！现在可以开始对话了。")

            except Exception as e:
                st.error(f"🔴 运行出错: {str(e)}")

# --- 4. 聊天交互界面 ---
st.divider()

if st.session_state.vectorstore:
    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理新用户输入
    if prompt := st.chat_input("向 AI 提问关于文档的内容..."):
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 检索并生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 检索最相关的 4 个片段
                docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in docs])

                # 构建最终 Prompt
                system_prompt = f"你是一个文档助手。请根据以下已知信息回答问题。如果信息中没提到，请直说。 \n\n【已知资料】：\n{context}"

                response = st.session_state.llm.invoke([
                    ("system", system_prompt),
                    ("user", prompt)
                ]).content

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 请在左侧上传文档并点击『开始分析』来唤醒 AI。")