from typing import Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.retrievers import BaseRetriever
from pydantic import SkipValidation, BaseModel
from langgraph.prebuilt import create_react_agent

class VectorSearchRetriever(BaseRetriever, BaseModel):
    """
    ベクトル検索を行うためのRetrieverクラス。
    """
    vector_store: SkipValidation[Any]
    embedding_model: SkipValidation[Any]
    k: int = 5 # 返すDocumentのチャンク数

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Dense embedding
        embedding = self.embedding_model.embed_query(query)
        search_results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding,
            k=self.k,
        )

        # Document のリストだけ取り出す
        return [doc for doc, _ in search_results]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def print_agent_result_details(result):
    messages = result["messages"]
    for idx, msg in enumerate(messages):
        # メッセージの型名 (HumanMessage, AIMessage, ToolMessageなど)
        msg_type = type(msg).__name__
        print(f"\n=== Message {idx} ===")
        print(f"Type: {msg_type}")

        # content (人間の発話・ツールの返答・AIの返答など) を表示
        if hasattr(msg, "content"):
            print("Content:")
            print(msg.content)

        # AIMessage や ToolMessage だけにあるパラメータを表示したい場合はチェック
        if msg_type == "AIMessage":
            # tool_callsがあれば表示
            if hasattr(msg, "tool_calls"):
                print("Tool calls:")
                print(msg.tool_calls)
            # usage_metadataなど
            if hasattr(msg, "response_metadata"):
                print("Response Metadata:")
                print(msg.response_metadata)

        if msg_type == "ToolMessage":
            # ツールの name など
            if hasattr(msg, "name"):
                print(f"Tool name: {msg.name}")




def main():
    # --- 定数定義 ---
    LLM_MODEL_NAME = "gemini-2.0-flash-001" 
    EM_MODEL_NAME = "models/text-embedding-004"
    CHROMA_DB = "./chroma/chroma_langchain_db"
    CHROMA_NAME = "example_collection"

    #質問文の例
    query = "16歳未満のユーザーが海外から当社サービスを利用した場合、親権者が同意していないときはどう扱われますか？ そのときデータは国外にも保存される可能性がありますか？"
    query = "おはよう"
    query = "今日は12歳の誕生日なんだ。これから初めて海外に行くんだよね"



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

    tool = dence_retriever.as_tool(
    name="Document_Search_Tool",
    description="サービスの規約に関する情報を取得するためのtoolです。"
    )

    
    # Prompt 定義
    prompt_template = """
あなたは、株式会社asapに所属するAIアシスタントです。
ユーザからサービスに関連する質問や雑談を振られた場合に、適切な情報を提供することもが求められています。
ユーザからサービスに関する情報を質問された場合は、toolから情報を取得して回答してください。
toolから情報を取得して回答する場合は、ユーザが一時情報を確認できるように、取得した情報の文章も追加で出力してください。
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{messages}"),
        ]
    )

    # エージェントの作成
    agent = create_react_agent(
        model=llm,
        tools=[tool],
        state_modifier=prompt
    )

    print("\n================= エージェントの実行結果 =================")
    result = agent.invoke({'messages':query})
    print_agent_result_details(result)

    print("\n================= 最終的な出力結果 =================")
    print(result['messages'][-1].content)
   
    

if __name__ == "__main__":
    main()
