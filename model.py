import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns

# Load dataset
file_path = 'D:/Python-Learning/Capstone Project I/Dataset/HAM10000_metadata.csv'
ham10000_metadata = pd.read_csv(file_path)

# Define image folder path
image_folder = 'D:/Python-Learning/Capstone Project I/archive/images'

# Encode labels
ham10000_metadata['label'] = ham10000_metadata['dx'].astype('category').cat.codes
num_classes = ham10000_metadata['label'].nunique()

# Split data
train_df, val_df = train_test_split(ham10000_metadata, test_size=0.2, random_state=4)

# Apply Random Oversampling on the training dataset
ros = RandomOverSampler(random_state=42)
train_resampled, train_labels_resampled = ros.fit_resample(train_df, train_df['label'])

# Custom Dataset class with augmentation for minority class
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None, augment=False):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform
        self.augment = augment
        self.aug_transform = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(30),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])
        self.valid_indices = self._filter_valid_indices()

    def _filter_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.dataframe)):
            row = self.dataframe.iloc[idx]
            image_name = row['image_id'] + '.jpg'
            image_path = os.path.join(self.image_folder, image_name)
            if os.path.exists(image_path):
                valid_indices.append(idx)
            else:
                print(f"Error loading image: {image_path}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.dataframe.iloc[actual_idx]
        image_name = row['image_id'] + '.jpg'
        label = row['label']
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Error loading image that should exist: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = Image.fromarray(image)

        if self.augment:
            image = self.aug_transform(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = SkinLesionDataset(train_resampled, image_folder, transform=transform, augment=True)
val_dataset = SkinLesionDataset(val_df, image_folder, transform=transform)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# Define the model (ResNet-50)
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def main():
    # Initialize model, loss function, and optimizer
    model = ResNet50(num_classes=num_classes).cuda()
    class_weights = torch.tensor([1.0] * num_classes, dtype=torch.float).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training and validation loop
    num_epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_aucs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Calculate ROC AUC
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_true_bin = label_binarize(val_true, classes=range(num_classes))
        val_preds_bin = label_binarize(val_preds.argmax(axis=1), classes=range(num_classes))

        # Compute micro-average ROC AUC score
        roc_auc = roc_auc_score(val_true_bin, val_preds, multi_class='ovr', average='micro')
        val_aucs.append(roc_auc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val AUC: {roc_auc:.4f}')

        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), 'resnet50_skin_classifier.pth')

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

    # Plot accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.close()

    # Plot ROC AUC curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), val_aucs, label='Validation ROC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.title('Validation ROC AUC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_auc_curve.png')
    plt.close()

    # Test the model on the validation set
    test_model(model, val_loader)

def test_model(model, val_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Classification Report
    print("Classification Report:")
    report = classification_report(all_labels, all_predictions, output_dict=True)
    print(pd.DataFrame(report).transpose())

    # Plot classification report
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
    plt.title('Classification Report')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('classification_report.png')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(cm.shape[0]), yticklabels=np.arange(cm.shape[1]))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    main()