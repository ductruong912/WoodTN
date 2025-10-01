import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 3. Dataset + Dataloader
# ===============================
test_dir = r"C:\WoodAI\Go_that"
testset = datasets.ImageFolder(test_dir, transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

class_names = testset.classes
print("Danh sách lớp:", class_names)

# ===============================
# 4. Load model
# ===============================
model_path = r"C:\WoodAI\save_model\Go_v4_epoch_14.pth"
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# ===============================
# 5. Dự đoán trên toàn bộ testset
# ===============================
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===============================
# 6. Metrics
# ===============================
# Classification report (precision, recall, f1)
report = classification_report(all_labels, all_preds,
                               target_names=class_names,
                               output_dict=True)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Report thành DataFrame
report_df = pd.DataFrame(report).transpose()

# ===============================
# 7. Xuất ra Excel
# ===============================
os.makedirs(r"C:\WoodAI\results", exist_ok=True)
output_file = r"C:\WoodAI\results\Go v4 epoch 14.xlsx"

with pd.ExcelWriter(output_file) as writer:
    cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
    report_df.to_excel(writer, sheet_name="Classification_Report")

print(f"✅ Kết quả đã lưu vào: {output_file}")
