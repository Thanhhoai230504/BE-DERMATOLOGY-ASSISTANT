
# from fastapi.middleware.cors import CORSMiddleware 
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai

# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, StreamingResponse
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
# from tensorflow.keras.models import load_model
# import io
# from PIL import Image
# from ultralytics import YOLO
# import cv2
# # ========== Cấu hình API Key cho Gemini ==========
# # API_KEY = "AIzaSyBoLSKrZTfRs6V1cEkkFS0ttlossjGvXlA"
# API_KEY = "AIzaSyDi3lNXZVCxGjzwijG_lWdnfT3ZAabdoMA"
# genai.configure(api_key=API_KEY)
# model_name = "models/embedding-001"
# genai_model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
# # Load sẵn mô hình YOLO
# yolo_model = YOLO("./best.pt")

# # ========== Load dữ liệu ==========
# df = pd.read_parquet('./qa_with_embeddings_data.parquet')

# # ========== Khởi tạo FastAPI ==========
# app = FastAPI()

# # ========== Bật CORS cho React frontend ==========
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # ⚠️ Thay đổi nếu frontend bạn chạy ở domain khác
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["X-Num-Acne", "X-Acne-Ratio", "X-Severity"]
# )
# # Load model và thông số
# model_path = r"./efficientnetb4_finetuned.keras"
# model = load_model(model_path)
# class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# IMG_SIZE = (380, 380)

# # Hàm tiền xử lý ảnh
# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize(IMG_SIZE)
#     img = img_to_array(img)
#     img = effnet_preprocess(img)
#     return np.expand_dims(img, axis=0)
# # ========== Request Schema ==========
# class Question(BaseModel):
#     question: str

# # ========== Tìm kiếm câu hỏi tương đồng ==========
# def search_similar_embeddings(query_embedding: list | np.ndarray, df: pd.DataFrame, top_k: int = 3, threshold: float = 0.6) -> pd.DataFrame:
#     query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
#     embedding_matrix = np.array(df['embedding'].tolist(), dtype=np.float32)
#     similarities = cosine_similarity(query_vector, embedding_matrix)[0]
#     df_with_similarity = df.copy()
#     df_with_similarity['similarity'] = similarities
#     result = (
#         df_with_similarity[df_with_similarity['similarity'] >= threshold]
#         .sort_values(by='similarity', ascending=False)
#         .head(top_k)
#     )
#     return result[['question', 'answers', 'similarity']]

# # Hàm phân loại mức độ mụn
# def classify_severity(num_acne, acne_ratio):
#     if num_acne == 0:
#         return "none"
#     elif num_acne <= 5 and acne_ratio < 2:
#         return "low"
#     elif num_acne <= 15 or acne_ratio < 5:
#         return "medium"
#     else:
#         return "high"
    

# # Hàm xử lý nhận diện và trả về kết quả
# def analyze_image_bytes(image_bytes, model):
#     # Đọc ảnh từ bytes
#     image_array = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

#     # Predict
#     results = model(img)
#     boxes = results[0].boxes.xyxy.cpu().numpy()

#     num_acne = len(boxes)
#     total_area = img.shape[0] * img.shape[1]

#     acne_area = 0
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box)
#         area = (x2 - x1) * (y2 - y1)
#         acne_area += area
#         # Vẽ bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     acne_ratio = acne_area / total_area * 100
#     severity = classify_severity(num_acne, acne_ratio)

#     return img, num_acne, acne_ratio, severity

# # API nhận ảnh, xử lý và trả về kết quả
# @app.post("/detect-acne")
# async def detect_acne(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         img, num_acne, acne_ratio, severity = analyze_image_bytes(image_bytes, yolo_model)

#         print(f" Số nốt mụn: {num_acne}, Diện tích mụn: {acne_ratio:.2f}%, Mức độ: {severity}")

#         # Encode lại ảnh đã vẽ bbox thành JPEG
#         success, img_encoded = cv2.imencode(".jpg", img)
#         if not success:
#             raise ValueError("Lỗi encode ảnh")

#         img_bytes = img_encoded.tobytes()

#         return StreamingResponse(
#             io.BytesIO(img_bytes),
#             media_type="image/jpeg",
#             headers={
#                 "X-Num-Acne": str(num_acne),
#                 "X-Acne-Ratio": f"{acne_ratio:.2f}",
#                 "X-Severity": severity
#             }
#         )

#     except Exception as e:
#         print(f"⚠️ Lỗi xử lý ảnh: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})


# # ========== Xử lý câu hỏi ========== (Gemini Q&A)
# @app.post("/ask")
# async def receive_question(data: Question):
#     question = data.question

#     result = genai.embed_content(
#         model=model_name,
#         content=question,
#         task_type="SEMANTIC_SIMILARITY"
#     )
#     question_embedding = result['embedding']

#     retrieval_docs = search_similar_embeddings(
#         query_embedding=question_embedding, df=df, top_k=5, threshold=0.85
#     )

#     document = "\n\n".join(
#         f"Câu hỏi: {row['question']}\nTrả lời: {row['answers']}"
#         for _, row in retrieval_docs.iterrows()
#     )

#     prompt = f"""
#     Bạn là một trợ lý ảo chuyên hỗ trợ người dùng tìm hiểu thông tin về bệnh da liễu.
#     Dựa vào câu hỏi người dùng và các tài liệu tham khảo dưới đây, hãy đưa ra câu trả lời chính xác, ngắn gọn, đúng chuyên môn.
#     Nếu tài liệu không đủ để trả lời, hãy nói "Tôi chưa có đủ thông tin chính xác để trả lời câu hỏi này."
#     <question>{question}</question>
#     <document>{document}</document>
#     """

#     try:
#         response = genai_model.generate_content(prompt)
#         return {"answer": response.text}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={
#             "error": str(e),
#             "answer": "Hệ thống hiện tại đang quá tải hoặc gặp lỗi. Vui lòng thử lại sau vài phút."
#         })


# # API nhận ảnh và trả về dự đoán
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         img = preprocess_image(image_bytes)

#         pred = model.predict(img, verbose=0)
#         class_idx = np.argmax(pred)
#         class_label = class_names[class_idx]
#         confidence = float(np.max(pred) * 100)

#         return {
#             "prediction": class_label,
#             "confidence": f"{confidence:.2f}"
#         }

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


import os
import io
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from pytorch_lightning import LightningModule
from ultralytics import YOLO
import google.generativeai as genai

# ========== Cấu hình API Key cho Gemini ==========
# API_KEY = "AIzaSyBoLSKrZTfRs6V1cEkkFS0ttlossjGvXlA"
API_KEY = "AIzaSyDi3lNXZVCxGjzwijG_lWdnfT3ZAabdoMA"
genai.configure(api_key=API_KEY)
model_name = "models/embedding-001"
genai_model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
# Load sẵn mô hình YOLO
yolo_model = YOLO("./best.pt")

# ========== Load dữ liệu ==========
df = pd.read_parquet('./qa_with_embeddings_data.parquet')

# ========== Khởi tạo FastAPI ==========
app = FastAPI()

# ========== Bật CORS cho React frontend ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ⚠️ Thay đổi nếu frontend bạn chạy ở domain khác
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Num-Acne", "X-Acne-Ratio", "X-Severity"]
)
# === Cấu hình ===
model_ckpt = "vit-best.ckpt"
label_map = {
    0: "akiec",  # Actinic keratoses
    1: "bcc",    # Basal cell carcinoma
    2: "bkl",    # Benign keratosis-like lesions
    3: "df",     # Dermatofibroma
    4: "mel",    # Melanoma
    5: "nv",     # Melanocytic nevi
    6: "vasc"    # Vascular lesions
}

# === Feature extractor & Transform giống khi train ===
extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std),
])

# === Mô hình Lightning giống lúc train ===
class ViTClassifier(LightningModule):
    def __init__(self, num_labels=7):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=num_labels)

    def forward(self, x):
        return self.model(x).logits

# === Load mô hình ViT đã train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTClassifier.load_from_checkpoint(model_ckpt)
model.to(device)
model.eval()
# ========== Request Schema ==========
class Question(BaseModel):
    question: str

# ========== Tìm kiếm câu hỏi tương đồng ==========
def search_similar_embeddings(query_embedding: list | np.ndarray, df: pd.DataFrame, top_k: int = 3, threshold: float = 0.6) -> pd.DataFrame:
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

# Hàm phân loại mức độ mụn
def classify_severity(num_acne, acne_ratio):
    if num_acne == 0:
        return "none"
    elif num_acne <= 5 and acne_ratio < 2:
        return "low"
    elif num_acne <= 15 or acne_ratio < 5:
        return "medium"
    else:
        return "high"
    

# Hàm xử lý nhận diện và trả về kết quả
def analyze_image_bytes(image_bytes, model):
    # Đọc ảnh từ bytes
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Predict
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    num_acne = len(boxes)
    total_area = img.shape[0] * img.shape[1]

    acne_area = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        acne_area += area
        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    acne_ratio = acne_area / total_area * 100
    severity = classify_severity(num_acne, acne_ratio)

    return img, num_acne, acne_ratio, severity

# API nhận ảnh, xử lý và trả về kết quả
@app.post("/detect-acne")
async def detect_acne(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img, num_acne, acne_ratio, severity = analyze_image_bytes(image_bytes, yolo_model)

        print(f" Số nốt mụn: {num_acne}, Diện tích mụn: {acne_ratio:.2f}%, Mức độ: {severity}")

        # Encode lại ảnh đã vẽ bbox thành JPEG
        success, img_encoded = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Lỗi encode ảnh")

        img_bytes = img_encoded.tobytes()

        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={
                "X-Num-Acne": str(num_acne),
                "X-Acne-Ratio": f"{acne_ratio:.2f}",
                "X-Severity": severity
            }
        )

    except Exception as e:
        print(f"⚠️ Lỗi xử lý ảnh: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ========== Xử lý câu hỏi ========== (Gemini Q&A)
@app.post("/ask")
async def receive_question(data: Question):
    question = data.question

    result = genai.embed_content(
        model=model_name,
        content=question,
        task_type="SEMANTIC_SIMILARITY"
    )
    question_embedding = result['embedding']

    retrieval_docs = search_similar_embeddings(
        query_embedding=question_embedding, df=df, top_k=5, threshold=0.85
    )

    document = "\n\n".join(
        f"Câu hỏi: {row['question']}\nTrả lời: {row['answers']}"
        for _, row in retrieval_docs.iterrows()
    )

    prompt = f"""
    Bạn là một trợ lý ảo chuyên hỗ trợ người dùng tìm hiểu thông tin về bệnh da liễu.
    Dựa vào câu hỏi người dùng và các tài liệu tham khảo dưới đây, hãy đưa ra câu trả lời chính xác, ngắn gọn, đúng chuyên môn.
    Nếu tài liệu không đủ để trả lời, hãy nói "Tôi chưa có đủ thông tin chính xác để trả lời câu hỏi này."
    <question>{question}</question>
    <document>{document}</document>
    """

    try:
        response = genai_model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "answer": "Hệ thống hiện tại đang quá tải hoặc gặp lỗi. Vui lòng thử lại sau vài phút."
        })


# API nhận ảnh và trả về dự đoán
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            pred_label = label_map[pred_class]
            confidence = float(np.max(probs) * 100)

        return {
            "prediction": pred_label,
            "class_index": pred_class,
            "confidence": f"{confidence:.2f}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})