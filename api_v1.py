import os
import io
import asyncio
import httpx
import torch
import timm
import numpy as np
import albumentations as A
import cv2  # 用于 fallback 的 resize，如果没有 cv2 会尝试用 PIL

from typing import List
from pydantic import BaseModel
from albumentations.pytorch import ToTensorV2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,APIRouter
from contextlib import asynccontextmanager

# ================= 1. 全局配置 =================
# 请核对你的模型路径
MODEL_FOLDER = "./model_all"
NETWORK_TYPE = "resnet50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 预处理类 (防报错独立版) =================
# 优先尝试从 train.py 导入，如果失败则使用内置的备用逻辑
try:
    from train import ResizeWithPad
    print("✅ Successfully imported ResizeWithPad from train.py")
except ImportError:
    print("⚠️ 'train.py' not found. Using standalone ResizeWithPad fallback.")
    
    class ResizeWithPad:
        """
        备用缩放类：保持长宽比缩放，并填充黑边 (Letterbox)
        """
        def __init__(self, target_shape):
            self.target_h, self.target_w = target_shape

        def __call__(self, image, **kwargs):
            h, w = image.shape[:2]
            scale = min(self.target_h / h, self.target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 使用 cv2 缩放
            try:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            except:
                # 如果没有cv2，用PIL作为最后兜底
                pil_img = Image.fromarray(image)
                resized = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))

            # 计算填充
            delta_h = self.target_h - new_h
            delta_w = self.target_w - new_w
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            # 填充黑边
            new_image = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            return new_image

# ================= 3. 数据模型定义 =================
class UrlBatchRequest(BaseModel):
    urls: List[str]  # 接收 JSON: {"urls": ["http...", "http..."]}

# ================= 4. 生命周期 (加载模型) =================
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Server starting on {DEVICE}...")
    
    # --- 加载类别 ---
    class_file = os.path.join(MODEL_FOLDER, "class_names.txt")
    if not os.path.exists(class_file):
        raise RuntimeError(f"❌ Class file missing: {class_file}")
    
    with open(class_file, "r") as f:
        class_names = f.read().splitlines()
    
    # --- 加载模型 ---
    print(f"🔄 Loading {NETWORK_TYPE}...")
    model = timm.create_model(NETWORK_TYPE, pretrained=False, num_classes=len(class_names))
    model.to(DEVICE)
    
    # 优先加载最佳权重
    weights_path = os.path.join(MODEL_FOLDER, "best_model_params.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_FOLDER, "trained_model.pth")
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Weights loaded: {os.path.basename(weights_path)}")
    else:
        raise RuntimeError(f"❌ No model weights found in {MODEL_FOLDER}")

    # --- 定义转换 ---
    transform = A.Compose([
        A.Lambda(image=ResizeWithPad((320, 320))), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    ml_models["model"] = model
    ml_models["classes"] = class_names
    ml_models["transform"] = transform
    
    yield
    
    ml_models.clear()
    torch.cuda.empty_cache()
    print("🛑 Server shutting down.")

app = FastAPI(lifespan=lifespan, title="Font Classifier API")

router_v1 = APIRouter(prefix="/v1")

# ================= 5. 核心逻辑函数 =================
def bytes_to_tensor(content, transform):
    """将图片字节流转为预处理后的Tensor"""
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    return transform(image=img_np)["image"]

def batch_inference(tensors, model, class_names):
    """批量推理并返回结果列表"""
    if not tensors: return []
    
    batch_input = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        logits = model(batch_input)
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, 1)
        
    results = []
    for i in range(len(preds)):
        results.append({
            "prediction": class_names[preds[i].item()],
            "confidence": round(confs[i].item(), 4)
        })
    return results

# ================= 6. API 接口定义 =================

@router_v1.post("/predict_urls")
async def predict_urls(request: UrlBatchRequest):
    """
    【推荐】通过 JSON 批量识别 URL
    输入: { "urls": ["http://a.com/1.jpg", "http://b.com/2.jpg"] }
    """
    if "model" not in ml_models: raise HTTPException(500, "Model loading...")
    
    urls = [u for u in request.urls if u.strip()]
    if not urls: raise HTTPException(400, "Empty url list")

    print(f"🌐 Downloading {len(urls)} URLs...")

    # 并发下载
    async def fetch(client, url):
        try:
            resp = await client.get(url, follow_redirects=True, timeout=10.0)
            return (resp.content if resp.status_code==200 else None, url, None)
        except Exception as e:
            return (None, url, str(e))

    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, url) for url in urls]
        downloads = await asyncio.gather(*tasks)

    # 处理下载结果
    valid_tensors = []
    map_indices = [] # 记录有效图片在原列表中的位置
    final_res = [{"url": u, "status": "failed", "error": "unknown"} for u in urls]

    transform = ml_models["transform"]

    for i, (data, url, err) in enumerate(downloads):
        if data:
            try:
                tensor = bytes_to_tensor(data, transform)
                valid_tensors.append(tensor)
                map_indices.append(i)
                final_res[i]["status"] = "success"
                final_res[i]["error"] = None
            except Exception as e:
                final_res[i]["error"] = f"Image Error: {e}"
        else:
            final_res[i]["error"] = f"Download Error: {err}"

    # 推理
    if valid_tensors:
        preds = batch_inference(valid_tensors, ml_models["model"], ml_models["classes"])
        for idx, pred in zip(map_indices, preds):
            final_res[idx].update(pred)

    return {"total": len(urls), "results": final_res}

@router_v1.post("/predict_files")
async def predict_files(files: List[UploadFile] = File(...)):
    """
    通过 Form-Data 批量上传本地文件
    """
    if "model" not in ml_models: raise HTTPException(500, "Model loading...")
    
    valid_tensors = []
    file_names = []
    
    transform = ml_models["transform"]

    print(f"📂 Receiving {len(files)} files...")
    for file in files:
        try:
            content = await file.read()
            if len(content) > 0:
                tensor = bytes_to_tensor(content, transform)
                valid_tensors.append(tensor)
                file_names.append(file.filename)
        except Exception as e:
            print(f"Skipping {file.filename}: {e}")

    if not valid_tensors:
        return {"count": 0, "msg": "No valid images."}

    preds = batch_inference(valid_tensors, ml_models["model"], ml_models["classes"])
    
    # 合并文件名和结果
    results = []
    for name, pred in zip(file_names, preds):
        results.append({"filename": name, **pred})

    return {"count": len(results), "results": results}
    
app.include_router(router_v1)
# ================= 7. 启动入口 =================
if __name__ == "__main__":
    import uvicorn
    # 这里的 reload=False 很重要，避免重复加载模型
    uvicorn.run("api:app", host="0.0.0.0", port=6006, reload=False)