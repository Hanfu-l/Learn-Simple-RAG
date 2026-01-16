
import requests
from typing import List
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI

class GiteeAIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "Qwen3-Embedding-8B"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://ai.gitee.com/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入"""
        embeddings = []
        for text in texts:
            content = text.page_content if hasattr(text, "page_content") else str(text)
            if not content.strip():
                continue  # 跳过空文本
            response = self._call_api(content)
            embeddings.append(response["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """生成单个查询嵌入"""
        return self.embed_documents([text])[0]

    def _call_api(self, text: str) -> dict:
        """调用 Gitee AI Embedding API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
            "dimensions": 4096
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()  # 检查请求是否成功
        return response.json()