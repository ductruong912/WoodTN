import torch
import timm
import torch.nn as nn

# --- Cấu hình ---
pth_path = r"C:\Users\xuant\Documents\GitHub\WoodTN\checkpoints\mvitv2_best_acc.pth"
num_classes = 16
img_size = 224
onnx_path = "mvitv2_best_acc.onnx"

# --- Load model ---
model = timm.create_model("mvitv2_small", pretrained=False)

# --- Fix head nếu là Sequential ---
if isinstance(model.head, nn.Sequential):
    in_features = model.head[-1].in_features
    model.head[-1] = nn.Linear(in_features, num_classes)
else:
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

# --- Load checkpoint ---
ckpt = torch.load(pth_path, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

dummy_input = torch.randn(1, 3, img_size, img_size)

torch.onnx.export(
    model,                          # model PyTorch
    dummy_input,                     # input dummy
    onnx_path,                       # tên file xuất ra
    input_names=['input'],           # tên input
    output_names=['output'],         # tên output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=17,                # opset mới để support MViTv2
    do_constant_folding=True         # fold hằng số để tối ưu
)

print(f"✅ Model đã được export sang ONNX: {onnx_path}")
