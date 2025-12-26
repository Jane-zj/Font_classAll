import io
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
import uvicorn

# 引入原本的 transforms
from transforms import build_transforms

# ================= 配置区域 =================
CHECKPOINT_PATH = "./checkpoints/best.pt"
PAD_FILL = "black"  # 填充颜色
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

# 全局资源字典，用于存储加载好的模型和 Transform
global_resources = {}


def build_model(model_name: str, num_classes: int):
    """
    构建模型骨架
    """
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"不支持的模型：{model_name}")
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def softmax(x):
    return torch.softmax(x, dim=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动/关闭时的生命周期管理
    """
    # 1. 检查 Checkpoint 是否存在
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到 checkpoint：{CHECKPOINT_PATH}")
    
    print(f"[{DEVICE}] 正在加载模型: {CHECKPOINT_PATH} ...")
    
    # 2. 加载 Checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model_name = ckpt.get("model", "resnet18")
    num_classes = ckpt.get("num_classes")
    classes = ckpt.get("classes", [str(i) for i in range(num_classes)])
    config = ckpt.get("config", {})
    
    # 3. 初始化并加载模型权重
    model = build_model(model_name, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()  # 设置为评估模式
    
    # 4. 初始化 Transform
    pad_fill_color = (0, 0, 0) if PAD_FILL == "black" else (255, 255, 255)
    use_imagenet_norm = bool(config.get("pretrained", 1))
    
    _, val_tfms = build_transforms(
        pad_fill=pad_fill_color, 
        image_size_hw=(100, 600), 
        use_imagenet_norm=use_imagenet_norm
    )
    
    # 5. 存入全局变量
    global_resources["model"] = model
    global_resources["classes"] = classes
    global_resources["transform"] = val_tfms
    
    print(">>> 模型加载完成，服务已启动 <<<")
    yield
    # 服务关闭时清理
    global_resources.clear()
    print(">>> 服务已关闭，资源释放 <<<")


# 初始化 APP，绑定生命周期
app = FastAPI(lifespan=lifespan)

# 创建 V2 路由
router_v2 = APIRouter(prefix="/v2", tags=["Image Prediction"])


@router_v2.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    上传图片进行预测
    """
    model = global_resources.get("model")
    classes = global_resources.get("classes")
    transform = global_resources.get("transform")
    
    if not model:
        raise HTTPException(status_code=500, detail="模型未加载")

    # 1. 读取并转换图片
    try:
        contents = await file.read()
        # 即使上传的是 PNG/RGBA，也强制转为 RGB，防止通道数不对
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片文件无效: {e}")

    # 2. 预处理
    try:
        x = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预处理失败: {e}")

    # 3. 推理
    with torch.no_grad():
        logits = model(x)
        probs = softmax(logits)[0]
        
        # 获取最大概率的索引和值
        pred_idx = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_idx].item())
        
        # 映射类别名称
        if classes and pred_idx < len(classes):
            pred_cls = classes[pred_idx]
        else:
            pred_cls = str(pred_idx)

    # 4. 返回 JSON
    return {
        "filename": file.filename,
        "pred_class": pred_cls,
        "pred_index": pred_idx,
        "confidence": round(pred_prob, 6)
    }


# 注册路由
app.include_router(router_v2)


if __name__ == "__main__":
    # 启动服务：Host 0.0.0.0 允许外部访问，端口 6006
    uvicorn.run("api:app", host="0.0.0.0", port=6006)