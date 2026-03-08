import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 强制改变缓存目录到当前项目的特定文件夹
# 这样模型就会下载到 D:\Projects\BUPT-RAG\models 文件夹里
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(current_dir, "models")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(current_dir, "models")

# 2. 顺便关掉那个讨厌的软链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 然后再写其他的 import
from langchain_huggingface import HuggingFaceEmbeddings
# ... 其余代码保持不变 ...


# 1. 配置你的 DeepSeek (Windows 下环境变量建议直接写这里调试)
DEEPSEEK_API_KEY = "输入你自己的api"

llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1"
)


def run_rag():
    # 2. 自动创建 data 目录
    if not os.path.exists("data"):
        os.makedirs("data")
        print("📁 已为你创建 data 文件夹，请把 PDF 丢进去再运行！")
        return

    pdf_path = "data/test.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ 找不到文件: {pdf_path}")
        return

    # 3. 本地嵌入（Windows 会自动下载模型到 C:\Users\你的名字\.cache）
    print("🧠 正在初始化本地 AI 大脑...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 加载与切分
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(pages)

    # 建立索引
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 检索
    question = "请总结这份文档。"
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    print("🤖 DeepSeek 正在思考...")
    response = llm.invoke(f"已知信息：{context}\n问题：{question}")
    print(f"\n✨ AI 回答：\n{response.content}")


if __name__ == "__main__":
    run_rag()