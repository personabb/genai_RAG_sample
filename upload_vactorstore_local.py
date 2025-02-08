# gcloud auth application-default login
from typing import Any, Dict, List


# --- langchain / external libraries ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

import os
import glob

os.makedirs("chroma", exist_ok=True)

def load_text_files_from_folder(folder_path):
    """
    指定したフォルダ内のすべてのテキストファイル(.txt)を読み込む関数
    
    :param folder_path: 読み込むフォルダのパス
    :return: 読み込んだドキュメントのリスト
    """
    # フォルダ内のすべての .txt ファイルを取得
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))

    # すべてのテキストファイルを読み込む
    documents = []
    for file in text_files:
        loader = TextLoader(file)
        documents.extend(loader.load())  # 各ファイルの内容をリストに追加

    print(f"Loaded {len(documents)} documents from {folder_path}")
    return documents  # 読み込んだドキュメントのリストを返す


def main():
    # --- 定数定義 ---
    EM_MODEL_NAME = "models/text-embedding-004"
    RAG_FOLDER_PATH = "./inputs"
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
    documents = load_text_files_from_folder(RAG_FOLDER_PATH)

    # チャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(documents)

    # チャンクのテキスト部分を抽出
    texts = [doc.page_content for doc in doc_splits]

    # optional IDs とメタデータ
    ids = ["i_" + str(i + 1) for i in range(len(texts))]
    metadatas = [{"my_metadata": i} for i in range(len(texts))]

        # ---- dense embedding (Vertex AI で生成) ----
    dense_embeddings = embedding_model.embed_documents(texts)
    
    # embeddingsの中身を確認
    print("dense embeddings（一部）:", dense_embeddings[0][:5])  # 最初の埋め込みを確認
    print("dense embeddings length:", len(dense_embeddings))

    #https://github.com/langchain-ai/langchain/blob/5d581ba22c68ab46818197da907278c1c45aad41/libs/partners/chroma/langchain_chroma/vectorstores.py#L502
    result = vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    print("\nデータの登録が完了しました。\n")

    #　以下は、chroma DBに保存されたデータの中身を確認するためのコード
    # 全データの取得 (ドキュメントとメタデータだけ取得する例)
    data = vector_store._collection.get(include=["documents", "metadatas"])

    print("Documents（3件）:", data["documents"][:3])
    print("Metadatas（全件）:", data["metadatas"])

if __name__ == "__main__":
    main()
