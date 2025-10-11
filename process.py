import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model sentence-transformers
embedder = SentenceTransformer("all-MiniLM-L6-v2")

file_path = './data.xlsx'  # Make sure this file is in the specified path

try:
    df = pd.read_excel(file_path, usecols=['question','answers'])
    df['question'] = df['question'].astype(str)
    df['answers'] = df['answers'].astype(str)
    
    df['combined_qa'] = "Question: " + df['question'].astype(str)
    print(df['combined_qa'][0])

    # Tạo embeddings local (FREE)
    embeddings = embedder.encode(df['combined_qa'].tolist(), convert_to_numpy=True)

    df['embedding'] = embeddings.tolist()

    # Lưu lại file parquet
    df.to_parquet('./qa_with_embeddings_data_all-MiniLM-L6-v2.parquet', index=False)
    print("✅ Đã thêm embedding (SentenceTransformers) vào DataFrame và lưu file thành công.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng kiểm tra đường dẫn.")
except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc file Excel: {e}")
