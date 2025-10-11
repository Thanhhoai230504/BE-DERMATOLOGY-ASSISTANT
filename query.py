import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model sentence-transformers
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load parquet đã tạo
df = pd.read_parquet('./qa_with_embeddings_data_all-MiniLM-L6-v2.parquet')

def search_similar_embeddings(query_embedding, df, top_k=3, threshold=0.6):
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    embedding_matrix = np.array(df['embedding'].tolist(), dtype=np.float32)
    similarities = cosine_similarity(query_vector, embedding_matrix)[0]

    df_with_similarity = df.copy()
    df_with_similarity['similarity'] = similarities

    result = (
        df_with_similarity[df_with_similarity['similarity'] >= threshold]
        .sort_values(by='similarity', ascending=False)
        .head(top_k)
    )
    return result[['question', 'answers', 'similarity']]

# Ví dụ câu hỏi
question_example = "Tôi bị mụn với mức độ medium. Hãy đưa ra khuyến nghị điều trị phù hợp?"

# Embedding local thay vì gọi Gemini
question_embedding = embedder.encode(question_example)

# Tìm tài liệu liên quan
retrieval_docs = search_similar_embeddings(
    query_embedding=question_embedding,
    df=df,
    top_k=5,
    threshold=0.85
)

# Gộp thành document
document = "\n\n".join(
    f"Câu hỏi: {row['question']}\nTrả lời: {row['answers']}"
    for _, row in retrieval_docs.iterrows()
)

prompt = f"""
Bạn là một trợ lý ảo chuyên hỗ trợ người dùng tìm hiểu thông tin về bệnh da liễu.
Dựa vào câu hỏi người dùng và các tài liệu tham khảo dưới đây, hãy đưa ra câu trả lời chính xác, ngắn gọn, đúng chuyên môn.
Nếu tài liệu không đủ để trả lời, hãy nói "Tôi chưa có đủ thông tin chính xác để trả lời câu hỏi này."
<question>{question_example}</question>
<document>{document}</document>
"""

print("Context:\n", document)
