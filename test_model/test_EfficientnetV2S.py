import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 16
    model_path = r"C:\WoodAI\save_model\Go_v4_epoch_14.pth"
    test_dir = r"C:\WoodAI\Go_that"
    output_xlsx = r"C:\WoodAI\results\best_model_epoch14.xlsx"

    # transforms
    test_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # dataset
    testset = datasets.ImageFolder(test_dir, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=0  # ⚠ num_workers=0 an toàn cho Windows
    )
    class_names = testset.classes
    print("Danh sách lớp:", class_names)

    # model
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # test
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # save excel
    df_report = pd.DataFrame(report).transpose()
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    with pd.ExcelWriter(output_xlsx) as writer:
        df_report.to_excel(writer, sheet_name="Classification_Report")
        df_cm.to_excel(writer, sheet_name="Confusion_Matrix")

    print(f"✅ Saved results to {output_xlsx}")

if __name__ == "__main__":
    main()
