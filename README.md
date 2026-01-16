# 使用步骤

### 1.安装依赖

```
pip install -r requirements.txt
```

### 2.配置API

```
deepseek_llm=ChatOpenAI(

    base_url="https://api.siliconflow.cn/v1",

    api_key="*",#替换硅基流动API

    model="deepseek-ai/DeepSeek-V3.2"

)
```


```
api_key = "*"  # 替换模力方舟 API Key

embeddings = GiteeAIEmbeddings(api_key)
```

### 3.其他

	在实际使用时，嵌入模型和LLM模型可以自行调整
