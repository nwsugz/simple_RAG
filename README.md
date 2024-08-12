# simple_RAG

### 라이브러리 설치
```
pip install transformers faiss-cpu
```

### 데이터 준비
```
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "The Louvre Museum is the world's largest art museum.",
    "Paris is known for its cafe culture and landmarks."
]
```

### 검색모델 구현
```
import faiss
from transformers import BertTokenizer, BertModel
import numpy as np

# 문서 임베딩 생성 함수
def get_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 문서 임베딩 생성
doc_embeddings = get_embeddings(documents, model, tokenizer)

# FAISS 인덱스 생성
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# 검색 함수
def search(query, top_k=1):
    query_embedding = get_embeddings([query], model, tokenizer)
    D, I = index.search(query_embedding, top_k)
    return [documents[i] for i in I[0]]
```

### 생성모델 구현
```
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 모델과 토크나이저 로드
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# 답변 생성 함수
def generate_answer(query, context):
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### RAG모델 실행
```
# 사용자 질문
query = "What is the capital of France?"

# 관련 문서 검색
retrieved_docs = search(query, top_k=1)
context = retrieved_docs[0]

# 답변 생성
answer = generate_answer(query, context)
print(f"Question: {query}")
print(f"Answer: {answer}")
```
![](https://velog.velcdn.com/images/nwsugz/post/de6d25c4-431a-45fa-aa4d-7299ba22f20e/image.png)



### 번외
한국어로 데이터를 만들고 한국어로 질문했더니 결과가 좋지 못했다:(
이러한 이유는 코드에서 사용된 BERT 모델('bert-base-uncased')과 GPT-2 모델('gpt2')이 영어를 기반으로 훈련된 모델이기 때문에 한국어 텍스트를 입력으로 받으면 적절한 임베딩을 생성하지 못할 수 있기 때문이다.
![](https://velog.velcdn.com/images/nwsugz/post/3491d471-b261-451e-9cd5-bed05540c540/image.png)

![](https://velog.velcdn.com/images/nwsugz/post/850cb967-caf8-4b8d-aa65-8cbed492b94a/image.png)


기존 코드에서 한국어 텍스트에 대해 좀 더 나은 결과를 보기 위해 모델을 바꿔 테스트해봤다.

BERT 모델('klue/bert-base') 사용
```
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = BertModel.from_pretrained('klue/bert-base')

```
GPT-2 모델('skt/kogpt2-base-v2') 사용
```
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('skt/kogpt2-base-v2')
gpt2_model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

```
![](https://velog.velcdn.com/images/nwsugz/post/71ff3d08-c884-4c00-b758-1299031c120b/image.png)
기존 RAG코드에서 데이터와 질문만 한국어로 바꿨을때보다는 조금 개선된 것 같았지만 아쉬운 결과이긴 했다!
다른 방법은 없을지 좀 더 시도해볼 예정이다.
