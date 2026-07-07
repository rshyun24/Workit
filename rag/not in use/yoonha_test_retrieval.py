from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

client = QdrantClient(host="localhost", port=6333)
model  = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

query  = "지체상금 한도가 설정되지 않은 계약 조항"

# Dense + Sparse 동시 추출
output = model.encode([query], return_dense=True, return_sparse=True)
dense_vector = output["dense_vecs"][0].tolist()

# sparse: token_str → token_id 변환 + 중복 합산
id_to_weight: dict[int, float] = {}
for token_str, weight in output["lexical_weights"][0].items():
    token_id = model.tokenizer.convert_tokens_to_ids(token_str)
    if isinstance(token_id, int):
        id_to_weight[token_id] = id_to_weight.get(token_id, 0.0) + float(weight)

indices = list(id_to_weight.keys())
values  = list(id_to_weight.values())

# Hybrid RRF 검색
response = client.query_points(
    collection_name="law_kb",
    prefetch=[
        Prefetch(query=dense_vector,                                 limit=20, using="dense"),
        Prefetch(query=SparseVector(indices=indices, values=values), limit=20, using="sparse"),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
)

print(f"쿼리: {query}\n")
for p in response.points:
    chunk_id   = p.payload.get("chunk_id", "")
    chunk_text = p.payload.get("chunk_text", "")[:80]
    print(f"[{p.score:.4f}] {chunk_id} | {chunk_text}")