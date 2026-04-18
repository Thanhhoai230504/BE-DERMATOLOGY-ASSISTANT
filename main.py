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
from transformers import ViTImageProcessor, ViTForImageClassification
from pytorch_lightning import LightningModule
from ultralytics import YOLO
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ========== Cấu hình API Key cho Gemini (bắt buộc set GEMINI_API_KEY khi deploy) ==========
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
model_name = "models/embedding-001"
genai_model = genai.GenerativeModel("models/gemini-2.0-flash-exp") if API_KEY else None

MODELS_REPO = os.environ.get("DERMATOLOGY_MODELS_REPO", "thanhhoai23/dermatology-models")
QA_PARQUET_NAME = "qa_with_embeddings_data_all-MiniLM-L6-v2.parquet"


def _resolve_qa_parquet_path() -> str:
    override = os.environ.get("QA_PARQUET_PATH")
    if override and os.path.isfile(override):
        return override
    if os.path.isfile(QA_PARQUET_NAME):
        return QA_PARQUET_NAME
    hf_name = os.environ.get("QA_PARQUET_HF_FILENAME", QA_PARQUET_NAME)
    try:
        return hf_hub_download(repo_id=MODELS_REPO, filename=hf_name)
    except Exception as e:
        raise RuntimeError(
            f"Không tìm thấy dữ liệu Q&A parquet. Cách xử lý: (1) copy {QA_PARQUET_NAME} vào image, "
            f"(2) set QA_PARQUET_PATH=/đường/dẫn/file.parquet, hoặc (3) upload file lên HF repo "
            f"'{MODELS_REPO}' (filename={hf_name}) — repo private cần HF_TOKEN."
        ) from e

# Load sẵn mô hình YOLO
try:
    yolo_model_path = hf_hub_download(repo_id=MODELS_REPO, filename="best.pt")
except Exception as e:
    yolo_model_path = "./best.pt"
yolo_model = YOLO(yolo_model_path)

# Khởi tạo model embedding local
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Load dữ liệu ==========
df = pd.read_parquet(_resolve_qa_parquet_path())

# ========== Khởi tạo FastAPI ==========
app = FastAPI()

# ========== Bật CORS cho React frontend ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Num-Acne", "X-Acne-Ratio", "X-Severity"]
)
# === Cấu hình ===
try:
    model_ckpt = hf_hub_download(repo_id=MODELS_REPO, filename="vit-best.ckpt")
except Exception as e:
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
extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
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
    if not genai_model:
        return JSONResponse(
            status_code=503,
            content={
                "error": "GEMINI_API_KEY chưa được cấu hình trên server.",
                "answer": "Hệ thống Q&A chưa sẵn sàng. Vui lòng cấu hình biến môi trường GEMINI_API_KEY.",
            },
        )

    question = data.question

    question_embedding = embedder.encode(question).tolist()

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