from typing import Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_core.retrievers import BaseRetriever
from pydantic import SkipValidation

class VectorSearchRetriever(BaseRetriever):
    """
    ベクトル検索を行うためのRetrieverクラス。
    """
    vector_store: SkipValidation[Any]
    embedding_model: SkipValidation[Any]
    k: int = 5 # 返すDocumentのチャンク数

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        #Dense embedding
        embedding = self.embedding_model.embed_query(query)
        search_results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding,
            k=self.k,
        )

        #Document のリストだけ取り出す
        return [doc for doc, _ in search_results]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def main():
    # --- 定数定義 ---
    LLM_MODEL_NAME = "gemini-2.0-flash-001" 
    EM_MODEL_NAME = "models/text-embedding-004"
    CHROMA_DB = "./chroma/chroma_langchain_db"
    CHROMA_NAME = "example_collection"

    # 質問文
    query = "16歳未満のユーザーが海外から当社サービスを利用した場合、親権者が同意していないときはどう扱われますか？ そのときデータは国外にも保存される可能性がありますか？"
    #query = "おはよう"
    #query = "今日は12歳の誕生日なんだ。これから初めて海外に行くんだよね"


   
    # テキスト エンベディング モデルを定義する (dense embedding)
    embedding_model = GoogleGenerativeAIEmbeddings(model=EM_MODEL_NAME)

    # Chroma
    vector_store = Chroma(
        collection_name=CHROMA_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB,  # Where to save data locally, remove if not necessary
    )

    # Chatモデル (LLM)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=0.2,
        max_tokens=512,
    )

    
    # DenceRetrieverを用意
    dence_retriever = VectorSearchRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        k=5,
    )

    # dence_retriever = vector_store.as_retriever() # search_kwargs={'k': 4}

    
    # Prompt 定義
    prompt_template = """
あなたは、株式会社asapに所属するAIアシスタントです。
ユーザからサービスに関連する質問や雑談を振られた場合に、適切な情報を提供することもが求められています。
ユーザからサービスに関する情報を質問された場合は、下記の情報源から情報を取得して回答してください。
下記の情報源から情報を取得して回答する場合は、ユーザが一時情報を確認できるように、取得した情報の文章も追加で出力してください。

情報源
{context}
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{query}"),
        ]
    )


    # チェーンを定義（retriever で文脈を取り、Prompt に当てはめて、LLM へ）
    chain = (
        {"context": dence_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 実行
    print("===== DenseRetriever の実行結果 =====")
    dense_docs = dence_retriever.invoke(query)
    print("\nDenseRetrieved Documents:", dense_docs)

    print("\n================= LLMの実行結果 =================")
    result = chain.invoke(query)
    print(result)

  
    

if __name__ == "__main__":
    main()
