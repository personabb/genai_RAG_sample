# gcloud auth application-default login
from typing import Any, Dict, List


# --- langchain / external libraries ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain.document_loaders import TextLoader
from langchain_core.documents import Document


def main():
    """
    メイン処理を行う関数。
    Matching Engine と VertexAIEmbeddings を使い、
    ドキュメントを読み込んでチャンク分割し、BM25 と dense embedding の両方を作成し、
    最終的に Vector Search (Matching Engine) に登録する。
    """

    #EM_MODEL_NAME = "text-multilingual-embedding-002"
    EM_MODEL_NAME = "models/text-embedding-004"
    RAG_FILE = "./inputs/sample.txt"
    CHROMA_DB = "./chroma/chroma_langchain_db"
    CHROMA_NAME = "example_collection"

    
    # テキスト エンベディング モデルを定義する (dense embedding)
    embedding_model = GoogleGenerativeAIEmbeddings(model=EM_MODEL_NAME)

    vector_store = Chroma(
        collection_name=CHROMA_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB,  # Where to save data locally, remove if not necessary
    )


    # テキストファイルを読み込む
    loader = TextLoader(RAG_FILE)
    document = loader.load()

    # チャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(document)

    # チャンクのテキスト部分を抽出
    texts = [doc.page_content for doc in doc_splits]
    print("texts:", texts)

    # optional IDs とメタデータ
    ids = ["i_" + str(i + 1) for i in range(len(texts))]
    metadatas = [{"my_metadata": i} for i in range(len(texts))]

        # ---- dense embedding (Vertex AI で生成) ----
    dense_embeddings = embedding_model.embed_documents(texts)
    
    print("dense embeddings:", dense_embeddings[0])  # 最初の埋め込みを確認
    print("dense embeddings length:", len(dense_embeddings))

    #https://github.com/langchain-ai/langchain/blob/5d581ba22c68ab46818197da907278c1c45aad41/libs/partners/chroma/langchain_chroma/vectorstores.py#L502
    result = vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    print("データの登録が完了しました。")

    # 全データの取得 (ドキュメントとメタデータだけ取得する例)
    data = vector_store._collection.get(include=["documents", "metadatas"])


    print("Documents:", data["documents"])
    print("Metadatas:", data["metadatas"])

if __name__ == "__main__":
    main()
