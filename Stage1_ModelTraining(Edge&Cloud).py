#Stage 1: Initial edge and cloud model training are performed on the training set independently
#==========================================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import pickle

from Dataset_class import ASDDataset
from EdgeModel_class import EdgeModel, EdgeNode
from CloudModel_class import CloudModel

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Method 1: A lightweight edge model trainer
def train_edge_model_simple(model, train_loader, val_loader, num_epochs= 15, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3) 
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    return best_model_state, best_val_acc

# Method 2: A sophisticated cloud model trainer
def train_resnet_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
    
    return best_model_state, best_val_acc

# Method 3: Orchestrate simple edge model training and validation
def train_edge_model_simple_cv(dataset):
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.3, random_state=42, 
        stratify=[dataset[i][1] for i in range(len(dataset))]
    )

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ASDDataset(dataset.root_dir, transform=transform_train)
    val_dataset = ASDDataset(dataset.root_dir, transform=transform_val)
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)
    
    model = EdgeModel().to(device)
    
    best_model_state, best_val_acc = train_edge_model_simple(
        model, train_loader, val_loader, num_epochs=15, learning_rate=0.0001
    )
    
    print(f"Edge Model Accuracy: {best_val_acc:.2f}%")
    return best_model_state, best_val_acc

# Method 4: Manage k-fold cross-validation for robust cloud model training
def train_cloud_model_kfold(dataset, k=3):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Enhanced data preprocessing
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"\nTraining ResNet101 with {k}-fold cross-validation")
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k} for ResNet101")
        
        train_dataset_fold = ASDDataset(dataset.root_dir, transform=transform_train)
        val_dataset_fold = ASDDataset(dataset.root_dir, transform=transform_val)
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset_fold, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(val_dataset_fold, batch_size=32, sampler=val_sampler)
        
        model = CloudModel().to(device)
        
        best_model_state, best_val_acc = train_resnet_model(
            model, train_loader, val_loader, num_epochs=30, learning_rate=0.0001
        )
        
        fold_results.append(best_val_acc)
        best_models.append(best_model_state)

    # Select the best model from the folds
    best_fold_idx = np.argmax(fold_results)
    best_cloud_model = best_models[best_fold_idx]
    mean_accuracy = np.mean(fold_results)
    
    print(f"ResNet101 Mean Accuracy: {mean_accuracy:.2f}% (±{np.std(fold_results):.2f}%)")
    print(f"Best Fold Accuracy: {fold_results[best_fold_idx]:.2f}%")
    
    return best_cloud_model, mean_accuracy

def main():
    training_data_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\datasets\\training_set"

    if not os.path.exists(training_data_path):
        print(f"Error: Training data path {training_data_path} does not exist!")
        return
    
    # Create save directory
    os.makedirs("saved_models_exp1", exist_ok=True)
    
    # Load traning set
    train_dataset = ASDDataset(training_data_path)
    
    # Perform edge model training
    print("=" * 60)
    print("Training Edge Model (Simplified MobileNetV2)")
    print("=" * 60)
    edge_best_state, edge_accuracy = train_edge_model_simple_cv(train_dataset)
    
    # Save edge model's parameters after training
    torch.save(edge_best_state, "saved_models_exp1/best_edge_model.pth")
    print(f"\nEdge Model Training Complete! Accuracy: {edge_accuracy:.2f}%")
    
    # Perform cloud model training
    print("\n" + "=" * 60)
    print("Training Cloud Model Ensemble (ResNet + AdaBoost)")
    print("=" * 60)
    cloud_best_state, cloud_accuracy = train_cloud_model_kfold(train_dataset, k=3)
    
    # Save cloud model's parameters after training
    torch.save(cloud_best_state, "saved_models_exp1/best_cloud_model.pth")
    
    print(f"\nCloud Model Training Complete! Ensemble Accuracy: {cloud_accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    print(f"Edge Model (Simplified MobileNetV2):  {edge_accuracy:.2f}%")
    print(f"Cloud Model (ResNet Ensemble):        {cloud_accuracy:.2f}%")


if __name__ == "__main__":
    main()