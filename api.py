import os
import io
import asyncio
import httpx
import torch
import torch.nn as nn
import timm
import numpy as np
import albumentations as A
import cv2
import uvicorn

from typing import List
from pydantic import BaseModel
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import models
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from contextlib import asynccontextmanager

# === 引入 V2 特有的 transforms ===
# 确保 transforms.py 在同级目录下
try:
    from transforms import build_transforms
except ImportError:
    print("❌ 警告: 找不到 transforms.py，V2 接口可能无法正常工作")

# ================= 1. 全局配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- V1 配置 ---
V1_MODEL_FOLDER = "./model_all"
V1_NETWORK_TYPE = "resnet50"

# --- V2 配置 ---
V2_CHECKPOINT_PATH = "./checkpoints/best.pt"
V2_PAD_FILL = "black"

# --- 全局资源池 ---
# 结构: { "v1": {model, transform, ...}, "v2": {model, transform, ...} }
global_resources = {}

# ================= 2. V1 专用工具类 (ResizeWithPad) =================
class ResizeWithPad:
    def __init__(self, target_shape):
        self.target_h, self.target_w = target_shape

    def __call__(self, image, **kwargs):
        h, w = image.shape[:2]
        scale = min(self.target_h / h, self.target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        try:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except:
            pil_img = Image.fromarray(image)
            resized = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))

        delta_h = self.target_h - new_h
        delta_w = self.target_w - new_w
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_image = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return new_image

class UrlBatchRequest(BaseModel):
    urls: List[str]

# ================= 3. V2 专用工具函数 (Build Model) =================
def build_model_v2(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"V2不支持的模型：{model_name}")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def softmax(x):
    return torch.softmax(x, dim=1)

# ================= 4. 生命周期管理 (同时加载 V1 和 V2) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 服务启动中，设备: {DEVICE}")
    
    # ---------------- 加载 V1 资源 ----------------
    print("--- [V1] 正在加载模型 ---")
    try:
        v1_res = {}
        # 1. 加载类别
        class_file = os.path.join(V1_MODEL_FOLDER, "class_names.txt")
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                v1_res["classes"] = f.read().splitlines()
        
            # 2. 加载模型
            model_v1 = timm.create_model(V1_NETWORK_TYPE, pretrained=False, num_classes=len(v1_res["classes"]))
            model_v1.to(DEVICE)
            
            w_path = os.path.join(V1_MODEL_FOLDER, "best_model_params.pt")
            if not os.path.exists(w_path):
                w_path = os.path.join(V1_MODEL_FOLDER, "trained_model.pth")
            
            if os.path.exists(w_path):
                ckpt = torch.load(w_path, map_location=DEVICE)
                model_v1.load_state_dict(ckpt)
                model_v1.eval()
                v1_res["model"] = model_v1
                print(f"✅ [V1] 模型加载成功: {os.path.basename(w_path)}")
            else:
                print("❌ [V1] 找不到模型权重文件")
            
            # 3. 定义 V1 Transform
            v1_res["transform"] = A.Compose([
                A.Lambda(image=ResizeWithPad((320, 320))), 
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            global_resources["v1"] = v1_res
        else:
             print(f"❌ [V1] 找不到 class_names.txt，跳过 V1 加载")
    except Exception as e:
        print(f"❌ [V1] 加载失败: {e}")


    # ---------------- 加载 V2 资源 ----------------
    print("--- [V2] 正在加载模型 ---")
    try:
        v2_res = {}
        if os.path.isfile(V2_CHECKPOINT_PATH):
            ckpt = torch.load(V2_CHECKPOINT_PATH, map_location=DEVICE)
            model_name = ckpt.get("model", "resnet18")
            num_classes = ckpt.get("num_classes")
            classes = ckpt.get("classes", [str(i) for i in range(num_classes)])
            config = ckpt.get("config", {})
            
            # 构建模型
            model_v2 = build_model_v2(model_name, num_classes)
            model_v2.load_state_dict(ckpt["state_dict"])
            model_v2.to(DEVICE)
            model_v2.eval()
            
            # 构建 Transform
            pad_fill_color = (0, 0, 0) if V2_PAD_FILL == "black" else (255, 255, 255)
            use_imagenet_norm = bool(config.get("pretrained", 1))
            _, val_tfms = build_transforms(
                pad_fill=pad_fill_color, 
                image_size_hw=(100, 600), 
                use_imagenet_norm=use_imagenet_norm
            )
            
            v2_res["model"] = model_v2
            v2_res["classes"] = classes
            v2_res["transform"] = val_tfms
            global_resources["v2"] = v2_res
            print(f"✅ [V2] 模型加载成功: {V2_CHECKPOINT_PATH}")
        else:
             print(f"❌ [V2] 找不到 checkpoint: {V2_CHECKPOINT_PATH}")
    except Exception as e:
        print(f"❌ [V2] 加载失败: {e}")

    print(">>> 所有模型加载流程结束 <<<")
    yield
    global_resources.clear()
    torch.cuda.empty_cache()
    print("🛑 服务已关闭")


app = FastAPI(lifespan=lifespan, title="Unified Font API")

# ================= 5. V1 路由定义 =================
router_v1 = APIRouter(prefix="/v1", tags=["V1 Multi-class Classification)"])

def v1_bytes_to_tensor(content, transform):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    return transform(image=img_np)["image"]

def v1_batch_inference(tensors, model, class_names):
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

@router_v1.post("/predict_urls")
async def predict_urls_v1(request: UrlBatchRequest):
    res = global_resources.get("v1")
    if not res or "model" not in res: raise HTTPException(500, "V1 Model not loaded")
    
    urls = [u for u in request.urls if u.strip()]
    if not urls: raise HTTPException(400, "Empty url list")

    async def fetch(client, url):
        try:
            resp = await client.get(url, follow_redirects=True, timeout=10.0)
            return (resp.content if resp.status_code==200 else None, url, None)
        except Exception as e:
            return (None, url, str(e))

    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, url) for url in urls]
        downloads = await asyncio.gather(*tasks)

    valid_tensors = []
    map_indices = []
    final_res = [{"url": u, "status": "failed", "error": "unknown"} for u in urls]

    for i, (data, url, err) in enumerate(downloads):
        if data:
            try:
                tensor = v1_bytes_to_tensor(data, res["transform"])
                valid_tensors.append(tensor)
                map_indices.append(i)
                final_res[i]["status"] = "success"
                final_res[i]["error"] = None
            except Exception as e:
                final_res[i]["error"] = f"Image Error: {e}"
        else:
            final_res[i]["error"] = f"Download Error: {err}"

    if valid_tensors:
        preds = v1_batch_inference(valid_tensors, res["model"], res["classes"])
        for idx, pred in zip(map_indices, preds):
            final_res[idx].update(pred)

    return {"total": len(urls), "results": final_res}

@router_v1.post("/predict_files")
async def predict_files_v1(files: List[UploadFile] = File(...)):
    res = global_resources.get("v1")
    if not res or "model" not in res: raise HTTPException(500, "V1 Model not loaded")
    
    valid_tensors = []
    file_names = []
    
    for file in files:
        try:
            content = await file.read()
            if len(content) > 0:
                tensor = v1_bytes_to_tensor(content, res["transform"])
                valid_tensors.append(tensor)
                file_names.append(file.filename)
        except Exception as e:
            print(f"Skipping {file.filename}: {e}")

    if not valid_tensors:
        return {"count": 0, "msg": "No valid images."}

    preds = v1_batch_inference(valid_tensors, res["model"], res["classes"])
    results = [{"filename": n, **p} for n, p in zip(file_names, preds)]
    return {"count": len(results), "results": results}


# ================= 6. V2 路由定义 =================
router_v2 = APIRouter(prefix="/v2", tags=["V2 (ResNet18)"])

@router_v2.post("/predict")
async def predict_v2(file: UploadFile = File(...)):
    res = global_resources.get("v2")
    if not res or "model" not in res: raise HTTPException(500, "V2 Model not loaded")
    
    model = res["model"]
    classes = res["classes"]
    transform = res["transform"]

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image Error: {e}")

    try:
        x = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform Error: {e}")

    with torch.no_grad():
        logits = model(x)
        probs = softmax(logits)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_idx].item())
        pred_cls = classes[pred_idx] if classes and pred_idx < len(classes) else str(pred_idx)

    return {
        "filename": file.filename,
        "pred_class": pred_cls,
        "pred_index": pred_idx,
        "confidence": round(pred_prob, 6)
    }

# ================= 7. 主程序 =================
app.include_router(router_v1)
app.include_router(router_v2)

if __name__ == "__main__":
    # 使用同一个入口，端口 6006
    uvicorn.run(app, host="0.0.0.0", port=6006)