import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Load model đã train
model = models.efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, 16) 
model.load_state_dict(torch.load(r"C:\WoodAI\save_model\Go_v2_epoch_10.pth")) 
model = model.to("cuda")

# Freeze backbone, chỉ fine-tune head
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Transform
fine_transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.03),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])

# Label mapping đúng class gốc trong model
label_map = {
    0: 5,   # keo_tai_tuong -> class index 5
    1: 9    # mo_tuyen_quang -> class index 9
}

class CustomMappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, self.label_map[label]

    def __len__(self):
        return len(self.dataset)

# Load và mapping dataset
fine_dataset = datasets.ImageFolder(r"C:\WoodAI\data\data_fine_tune", transform=fine_transform)
mapped_dataset = CustomMappedDataset(fine_dataset, label_map)
fine_loader = DataLoader(mapped_dataset, batch_size=16, shuffle=True, num_workers=4)

# Fine-tune
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    loop = tqdm(fine_loader, desc=f"Fine-tune Epoch {epoch+1}")
    for inputs, labels in loop:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(fine_dataset)
    epoch_acc = running_corrects.double() / len(fine_dataset)
    print(f"Epoch {epoch+1}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}")

# Lưu model
torch.save(model.state_dict(), r"C:\WoodAI\save_model\Go_v2_finetuned.pth")
print("Model fine-tuned and saved.")
