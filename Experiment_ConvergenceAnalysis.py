import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os   
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
import shutil

from Dataset_class import ASDDataset
from EdgeModel_class import EdgeModel, EdgeNode
from CloudModel_class import CloudModel

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clone the file associated with the pre-trained edge and cloud model to the specified directory
def copy_pretrained_models():
    os.makedirs("load_models_experiment3", exist_ok=True)
    edge_mode_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\saved_models_exp1\\best_edge_model.pth"
    cloud_model_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\saved_models_exp1\\best_cloud_model.pth"

    if os.path.exists(edge_mode_path): 
        shutil.copy(edge_mode_path, "load_models_experiment3/")
    if os.path.exists(cloud_model_path):
        shutil.copy(cloud_model_path, "load_models_experiment3/")


# Class CloudEdgeFramework: build a cloud-edge collaborative inference framework
class CloudEdgeFramework:
    def __init__(self, edge_model_path, cloud_model_path, num_edge_nodes=3):
        self.edge_model_path = edge_model_path
        self.cloud_model_path = cloud_model_path
        self.num_edge_nodes = num_edge_nodes

        # Load pre-trained model weights
        edge_state_dict = torch.load(self.edge_model_path, map_location=device)
        cloud_state_dict = torch.load(self.cloud_model_path, map_location=device)
        
        # Create edge nodes
        self.edge_nodes = []
        for i in range(self.num_edge_nodes):
            self.edge_nodes.append(EdgeNode(i, edge_state_dict))
        
        # Create the cloud model 
        self.cloud_model = CloudModel().to(device)
        self.cloud_model.load_state_dict(cloud_state_dict)
        self.cloud_model.eval()
        
        self.shared_edge_params = edge_state_dict
        self.current_node_idx = 0

    def reset_models(self):
        edge_state_dict = torch.load(self.edge_model_path, map_location=device)

        self.edge_nodes = []
        for i in range(self.num_edge_nodes):
            self.edge_nodes.append(EdgeNode(i, edge_state_dict))
        
        self.current_node_idx = 0
              
        
    def compute_confidence(self, probs):
        """Compute Confidence"""
        asd_prob = probs[:, 1]  # probability of ASD
        non_asd_prob = probs[:, 0]  # probability of Non-ASD
        confidence = torch.abs(asd_prob - non_asd_prob)
        return confidence
    
    def edge_inference(self, images):
        node = self.edge_nodes[self.current_node_idx]
        self.current_node_idx = (self.current_node_idx + 1) % len(self.edge_nodes)
        
        probs = node.predict(images)
        confidence = self.compute_confidence(probs)
        predictions = torch.argmax(probs, dim=1)
        return probs, confidence, predictions
    
    def cloud_inference(self, images, temperature=3.0):
        with torch.no_grad():
            logits = self.cloud_model(images)
            soft_probs = F.softmax(logits / temperature, dim=1)
            predictions = torch.argmax(soft_probs, dim=1)
        return soft_probs, predictions
    

    
    def collaborative_inference(self, images, threshold):
        edge_probs, confidence, edge_preds = self.edge_inference(images)
        
        upload_mask = confidence < threshold
        upload_count = upload_mask.sum().item()
        
        final_predictions = edge_preds.clone()
        
        if upload_count > 0:
            uploaded_images = images[upload_mask]
            cloud_probs, cloud_preds = self.cloud_inference(uploaded_images)
            final_predictions[upload_mask] = cloud_preds
        
        return final_predictions
    
    def simulate_single_round_training(self, round_images, round_labels,threshold):
        for node in self.edge_nodes:
            node.model.train()

        optimizer = optim.Adam(self.edge_nodes[0].model.parameters(), lr=0.0001)
        
        nodes_upload_info = []
        
        for node_idx in range(3):
            start_idx = node_idx * 16
            end_idx = min(start_idx + 16, len(round_images))
            if start_idx >= len(round_images):
                break

            node_images = round_images[start_idx:end_idx]
            node_labels = round_labels[start_idx:end_idx]
            
            if len(node_images) == 0:
                continue
            
            batch_images = torch.stack(node_images).to(device)
            batch_labels = torch.tensor(node_labels).to(device)

            edge_logits = self.edge_nodes[node_idx].model(batch_images)
            edge_probs = F.softmax(edge_logits, dim=1)
            confidence = self.compute_confidence(edge_probs)
            edge_preds = torch.argmax(edge_probs, dim=1)
            
            upload_mask = confidence < threshold
            upload_count = upload_mask.sum().item()
            
            node_upload_data = {
                'node_idx': node_idx,
                'has_upload': False,
                'uploaded_images': None,
                'uploaded_labels': None,
                'edge_preds': None,
                'cloud_probs': None
            }
            
            if upload_count > 0:
                uploaded_images = batch_images[upload_mask]
                uploaded_labels = batch_labels[upload_mask]
                
                cloud_probs, cloud_preds = self.cloud_inference(uploaded_images, temperature=3.0)
                
                node_upload_data.update({
                    'has_upload': True,
                    'uploaded_images': uploaded_images,
                    'uploaded_labels': uploaded_labels,
                    'edge_preds': edge_preds[upload_mask],
                    'cloud_probs': cloud_probs,
                    'cloud_preds': cloud_preds
                })
            
            nodes_upload_info.append(node_upload_data)

        round_losses = []
        
        for node_data in nodes_upload_info:
            if node_data['has_upload']:
                node_idx = node_data['node_idx']
                uploaded_images = node_data['uploaded_images']
                uploaded_labels = node_data['uploaded_labels']
                edge_preds = node_data['edge_preds']
                cloud_probs = node_data['cloud_probs']
                cloud_preds = node_data['cloud_preds']
                
                logits = self.edge_nodes[node_idx].model(uploaded_images)
                edge_soft_probs = F.softmax(logits / 3.0, dim=1)
                
                consistent_mask = (edge_preds == cloud_preds)
                
                node_total_loss = 0
                node_loss_count = 0
                
                if consistent_mask.sum() > 0:
                    consistent_edge_probs = edge_soft_probs[consistent_mask]
                    consistent_cloud_probs = cloud_probs[consistent_mask]
                    kl_loss = F.kl_div(
                        torch.log(consistent_edge_probs + 1e-8), 
                        consistent_cloud_probs, 
                        reduction='sum'
                    )
                    node_total_loss += kl_loss
                    node_loss_count += consistent_mask.sum().item()
                        
                if (~consistent_mask).sum() > 0:
                    inconsistent_edge_probs = edge_soft_probs[~consistent_mask]
                    inconsistent_cloud_probs = cloud_probs[~consistent_mask]
                    inconsistent_labels = uploaded_labels[~consistent_mask]

                    kl_loss = F.kl_div(
                        torch.log(inconsistent_edge_probs + 1e-8), 
                        inconsistent_cloud_probs, 
                        reduction='sum'
                    )
                    
                    ce_loss = F.cross_entropy(
                        logits[~consistent_mask], 
                        inconsistent_labels,
                        reduction='sum'
                    )
                    
                    node_total_loss += (0.5*kl_loss + 0.5*ce_loss)
                    node_loss_count += (~consistent_mask).sum().item()
                
                if node_loss_count > 0:
                    node_avg_loss = node_total_loss / node_loss_count
                    round_losses.append(node_avg_loss)
        
        if len(round_losses) > 0:
            total_round_loss = sum(round_losses) / len(round_losses)
            
            optimizer.zero_grad()
            total_round_loss.backward()
            optimizer.step()
            
            updated_state = self.edge_nodes[0].model.state_dict()
            for other_node in self.edge_nodes[1:]:
                other_node.model.load_state_dict(updated_state)
        
        for node in self.edge_nodes:
            node.model.eval()
        
        return

    def evaluate_on_test_set(self, test_loader, paradigm, threshold):
        all_predictions = []
        all_labels = []
        
        self.current_node_idx = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if paradigm == 'edge':
                _, _, predictions = self.edge_inference(images)
            elif paradigm == 'confidence':
                predictions = self.collaborative_inference(images, threshold)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy

    def convergence_analysis_experiment(self, sim_dataset, test_loader, thresholds):
        results = {}
        
        for thre in thresholds:
            print(f"\nTesting different thresholds: {thre}")
            racc_history = []
            
            self.reset_models()
            
            total_samples_in_dataset = len(sim_dataset)
            indices = torch.randperm(total_samples_in_dataset).tolist()
            
            for round_idx in range(0, total_samples_in_dataset, 48):
                round_indices = indices[round_idx:round_idx + 48]
                
                print(f"  Round {round_idx // 48 + 1}/{(total_samples_in_dataset + 47) // 48}")
                
                round_images = []
                round_labels = []
                for idx in round_indices:
                    img, label = sim_dataset[idx]
                    round_images.append(img)
                    round_labels.append(label)
                
                if len(round_images) == 0:
                    break

                self.simulate_single_round_training(round_images, round_labels, thre)
                

                edge_acc = self.evaluate_on_test_set(test_loader, 'edge', thre)
                collaborative_acc = self.evaluate_on_test_set(test_loader, 'confidence', thre)
                
                if collaborative_acc > 0:
                    racc = edge_acc / collaborative_acc
                
                racc_history.append(racc)
                print(f"    Edge Acc: {edge_acc:.4f}, Collaborative Acc: {collaborative_acc:.4f}, rAcc: {racc:.4f}")
            
            results[thre] = racc_history
        
        return results


def main():
    copy_pretrained_models()
    
    # Dataset path
    simulation_data_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\datasets\\simulation_set"
    testing_data_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\datasets\\testing_set"
    
    # Data pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    sim_dataset = ASDDataset(simulation_data_path, transform=transform)
    test_dataset = ASDDataset(testing_data_path, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    print("Starting Convergence Analysis Experiment...")
    print(f"Simulation dataset size: {len(sim_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    framework = CloudEdgeFramework(
        "load_models_experiment3/best_edge_model.pth",
        "load_models_experiment3/best_cloud_model.pth"
    )
    
    thresholds_set = [0.1, 0.2, 0.3]
    
    convergence_results = framework.convergence_analysis_experiment(
        sim_dataset, test_loader, thresholds_set
    )
    

    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS RESULTS SUMMARY")
    print("="*60)
    
    for th, racc_history in convergence_results.items():
        print(f"Threshold {th}:")
        print(f"  Initial rAcc: {racc_history[0]:.4f}")
        print(f"  Final rAcc: {racc_history[-1]:.4f}")
        print(f"  Max rAcc: {max(racc_history):.4f}")
        print(f"  Min rAcc: {min(racc_history):.4f}")
    
    
if __name__ == "__main__":
    main()