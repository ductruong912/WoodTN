import torch
import torchvision
from torch import nn
from torchvision.models import efficientnet_v2_s
from pathlib import Path

# Thiết bị
device = torch.device("cpu")
print("✅ Using device:", device)

# Load model EfficientNetV2-S
model = efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(1280, 16)  # 16 lớp phân loại

# Load trọng số đã huấn luyện
pth_path = r"C:\TNT\WoodAI\save_model\best_model_epoch10.pth"
model.load_state_dict(torch.load(pth_path, map_location=device))
model.to(device)
model.eval()
print("Loaded model.")

# Tạo dummy input (batch size = 1)
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# Đường dẫn lưu file ONNX
onnx_path = "best_model_epoch10.onnx"

# Export sang ONNX
torch.onnx.export(
    model,                      # model PyTorch
    dummy_input,                # đầu vào mẫu
    onnx_path,                  # đường dẫn lưu
    export_params=True,         # lưu trọng số
    opset_version=11,           # phiên bản ONNX
    do_constant_folding=True,   # tối ưu hằng số
    input_names=['input'],      # tên input
    output_names=['output'],    # tên output
    dynamic_axes={              # cho phép kích thước batch thay đổi
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Exported model to ONNX format at: {onnx_path}")
