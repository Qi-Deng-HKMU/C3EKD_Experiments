import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import os   
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import shutil
import time

from Dataset_class import ASDDataset
from EdgeModel_class import EdgeModel, EdgeNode
from CloudModel_class import CloudModel

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clone the file associated with the pre-trained edge and cloud model to the specified directory
def copy_pretrained_models():
    os.makedirs("load_models_experiment2", exist_ok=True)
    edge_mode_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\saved_models_exp1\\best_edge_model.pth"
    cloud_model_path = "C:\\Users\\DENG Qi\\Desktop\\Experiment\\saved_models_exp1\\best_cloud_model.pth"

    if os.path.exists(edge_mode_path): 
        shutil.copy(edge_mode_path, "load_models_experiment2/")
    if os.path.exists(cloud_model_path):
        shutil.copy(cloud_model_path, "load_models_experiment2/")

# Class DelayCalculator: used to calculate time delay
class DelayCalculator:
    def __init__(self):
        self.image_size_kb = 30  
        self.image_size_bits = self.image_size_kb * 8 * 1024  # convert into bits
        
        # Bandwidth and fixed delay configuration
        self.device_to_edge_bandwidth_mbps = 20  # 20Mbps
        self.edge_to_cloud_bandwidth_mbps = 100  # 100Mbps
        self.device_to_edge_fixed_delay_ms = 5   # 5ms fixed delay
        self.edge_to_cloud_fixed_delay_ms = 20   # 20ms fixed delay
        
    def calculate_transmission_delay(self, num_images, from_device_to_edge=True):
        """Calculate transmission delay (ms)"""
        if from_device_to_edge:
            # edge devices to edge servers
            bandwidth_bps = self.device_to_edge_bandwidth_mbps * 1_000_000  # convert into bps
            transmission_time_s = (self.image_size_bits * num_images) / bandwidth_bps
            transmission_time_ms = transmission_time_s * 1000
            total_delay_ms = transmission_time_ms + self.device_to_edge_fixed_delay_ms
        else:
            # edge servers to the cloud server
            bandwidth_bps = self.edge_to_cloud_bandwidth_mbps * 1_000_000  # convert into bps
            transmission_time_s = (self.image_size_bits * num_images) / bandwidth_bps
            transmission_time_ms = transmission_time_s * 1000
            total_delay_ms = transmission_time_ms + self.edge_to_cloud_fixed_delay_ms
        
        return total_delay_ms


# Class CloudEdgeFramework: construct the cloud-edge collaborative inference framework
class CloudEdgeFramework:
    def __init__(self, edge_model_path, cloud_model_path, num_edge_nodes=3):
        self.edge_model_path = edge_model_path
        self.cloud_model_path = cloud_model_path
        
        # Load pre-trained model weights
        edge_state_dict = torch.load(edge_model_path, map_location=device)
        cloud_state_dict = torch.load(cloud_model_path, map_location=device)
        
        # Create edge nodes
        self.edge_nodes = []
        for i in range(num_edge_nodes):
            self.edge_nodes.append(EdgeNode(i, edge_state_dict))
        
        # Create the cloud model 
        self.cloud_model = CloudModel().to(device)
        self.cloud_model.load_state_dict(cloud_state_dict)
        self.cloud_model.eval()
        
        self.shared_edge_params = edge_state_dict
        self.current_node_idx = 0 
        self.delay_calculator = DelayCalculator()
        
    def compute_confidence(self, probs):
        """Compute Confidence"""
        asd_prob = probs[:, 1]      # probability of ASD
        non_asd_prob = probs[:, 0]  # probability of Non-ASD
        confidence = torch.abs(asd_prob - non_asd_prob)
        return confidence
    
    def edge_inference(self, images):
        """Edge inference"""
        node = self.edge_nodes[self.current_node_idx]
        self.current_node_idx = (self.current_node_idx + 1) % len(self.edge_nodes)
        
        start_time = time.time()
        probs = node.predict(images)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        confidence = self.compute_confidence(probs)
        predictions = torch.argmax(probs, dim=1)
        return probs, confidence, predictions, inference_time_ms
    
    def cloud_inference(self, images, temperature=3.0):
        """Cloud inference"""
        start_time = time.time()
        with torch.no_grad():
            logits = self.cloud_model(images)
            soft_probs = F.softmax(logits / temperature, dim=1)
            predictions = torch.argmax(soft_probs, dim=1)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        return soft_probs, predictions, inference_time_ms
    
    def collaborative_inference(self, images, threshold):
        """Cloud-edge collaboratie inference"""
        num_images = len(images)

        # Time delay from edge devices to edge servers
        device_to_edge_delay = self.delay_calculator.calculate_transmission_delay(
            num_images, from_device_to_edge=True
        )

        edge_probs, confidence, edge_preds, edge_inference_time = self.edge_inference(images)
        
        upload_mask = confidence < threshold
        upload_count = upload_mask.sum().item()
        
        final_predictions = edge_preds.clone()
        total_delay = device_to_edge_delay + edge_inference_time
        
        if upload_count > 0:
            # Time delay from edge servers to the cloud server
            edge_to_cloud_delay = self.delay_calculator.calculate_transmission_delay(
                upload_count, from_device_to_edge=False
            )

            uploaded_images = images[upload_mask]
            cloud_probs, cloud_preds, cloud_inference_time = self.cloud_inference(uploaded_images)
            final_predictions[upload_mask] = cloud_preds

            total_delay += edge_to_cloud_delay + cloud_inference_time
        
        upload_ratio = upload_count / len(images)
        
        return final_predictions, upload_ratio, total_delay


    def calculate_collaborative_no_update_delay(self, sim_dataset, threshold):
        """Calculate the time delay of cloud-edge collaboration framework without edge model updates"""
        total_time_delay = 0
        total_upload_count = 0
        total_samples = len(sim_dataset)
        
        indices = list(range(total_samples))
        
        for round_idx in range(0, total_samples, 48):
            round_indices = indices[round_idx:round_idx + 48]
            
            for node_idx in range(3):
                start_idx = node_idx * 16
                end_idx = min(start_idx + 16, len(round_indices))
                if start_idx >= len(round_indices):
                    break
                    
                node_indices = round_indices[start_idx:end_idx]
                
                node_images = []
                for idx in node_indices:
                    img, _ = sim_dataset[idx]
                    node_images.append(img)
                
                if len(node_images) == 0:
                    continue
                
                batch_images = torch.stack(node_images).to(device)
                num_images = len(batch_images)

                device_to_edge_delay = self.delay_calculator.calculate_transmission_delay(
                    num_images, from_device_to_edge=True
                )
                
                start_time = time.time()
                edge_probs = self.edge_nodes[node_idx].predict(batch_images)
                end_time = time.time()
                edge_inference_time = (end_time - start_time) * 1000

                confidence = self.compute_confidence(edge_probs)
                upload_mask = confidence < threshold
                upload_count = upload_mask.sum().item()
                total_upload_count += upload_count
                
                batch_delay = device_to_edge_delay + edge_inference_time
                
                if upload_count > 0:
                    edge_to_cloud_delay = self.delay_calculator.calculate_transmission_delay(
                        upload_count, from_device_to_edge=False
                    )
                    
                    uploaded_images = batch_images[upload_mask]
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.cloud_model(uploaded_images)
                    end_time = time.time()
                    cloud_inference_time = (end_time - start_time) * 1000
                    
                    batch_delay += edge_to_cloud_delay + cloud_inference_time
                
                total_time_delay += batch_delay
        
        upload_proportion = total_upload_count / total_samples if total_samples > 0 else 0
        avg_time_delay = round(total_time_delay / total_samples, 3)
        
        return upload_proportion, avg_time_delay

    def simulate_communication_rounds(self, sim_dataset, learning_rate=0.0001, threshold=0.7):
        """Complete simulation round"""

        for node in self.edge_nodes:
            node.model.train()

        optimizer = optim.Adam(self.edge_nodes[0].model.parameters(), lr=learning_rate)
        
        total_upload_count = 0
        total_samples = 0

        total_time_delay = 0  

        total_samples_in_dataset = len(sim_dataset)
        indices = torch.randperm(total_samples_in_dataset).tolist()
        
        for round_idx in range(0, total_samples_in_dataset, 48):
            round_indices = indices[round_idx:round_idx + 48]

            nodes_upload_info = []

            for node_idx in range(3):
                start_idx = node_idx * 16
                end_idx = min(start_idx + 16, len(round_indices))
                if start_idx >= len(round_indices):
                    break
                    
                node_indices = round_indices[start_idx:end_idx]
                
                node_images = []
                node_labels = []
                for idx in node_indices:
                    img, label = sim_dataset[idx]
                    node_images.append(img)
                    node_labels.append(label)
                
                if len(node_images) == 0:
                    continue
                
                batch_images = torch.stack(node_images).to(device)
                batch_labels = torch.tensor(node_labels).to(device)

                num_images = len(batch_images) 
                
                device_to_edge_delay = self.delay_calculator.calculate_transmission_delay(
                num_images, from_device_to_edge=True)

                start_time = time.time()
                edge_logits = self.edge_nodes[node_idx].model(batch_images)
                end_time = time.time()
                edge_inference_time = (end_time - start_time) * 1000  # convert into ms

                edge_probs = F.softmax(edge_logits, dim=1)
                confidence = self.compute_confidence(edge_probs)
                edge_preds = torch.argmax(edge_probs, dim=1)
                
                # Selective upload
                upload_mask = confidence < threshold
                upload_count = upload_mask.sum().item()
                total_upload_count += upload_count
                total_samples += len(batch_images)
                
                batch_delay = device_to_edge_delay + edge_inference_time

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

                    edge_to_cloud_delay = self.delay_calculator.calculate_transmission_delay(
                        upload_count, from_device_to_edge=False
                    )  

                    cloud_probs, cloud_preds, cloud_inference_time = self.cloud_inference(uploaded_images, temperature=3.0)
                    batch_delay += edge_to_cloud_delay + cloud_inference_time

                    node_upload_data.update({
                        'has_upload': True,
                        'uploaded_images': uploaded_images,
                        'uploaded_labels': uploaded_labels,
                        'edge_preds': edge_preds[upload_mask],
                        'cloud_probs': cloud_probs,
                        'cloud_preds': cloud_preds
                    })
                
                nodes_upload_info.append(node_upload_data)
                total_time_delay += batch_delay  

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
        
        upload_proportion = total_upload_count / total_samples if total_samples > 0 else 0

        for node in self.edge_nodes:
            node.model.eval()

        avg_time_delay = round(total_time_delay/2880.0, 3)

        return upload_proportion, avg_time_delay
    
    def calculate_pure_edge_delay(self, sim_dataset):
        total_time_delay = 0
        total_samples = len(sim_dataset)
        

        indices = list(range(total_samples))
        
        for round_idx in range(0, total_samples, 48):
            round_indices = indices[round_idx:round_idx + 48]
            
            for node_idx in range(3):
                start_idx = node_idx * 16
                end_idx = min(start_idx + 16, len(round_indices))
                if start_idx >= len(round_indices):
                    break
                    
                node_indices = round_indices[start_idx:end_idx]
                
                node_images = []
                for idx in node_indices:
                    img, _ = sim_dataset[idx]
                    node_images.append(img)
                
                if len(node_images) == 0:
                    continue
                
                batch_images = torch.stack(node_images).to(device)
                num_images = len(batch_images)
                

                device_to_edge_delay = self.delay_calculator.calculate_transmission_delay(
                    num_images, from_device_to_edge=True
                )
                

                start_time = time.time()
                _ = self.edge_nodes[node_idx].predict(batch_images)
                end_time = time.time()
                edge_inference_time = (end_time - start_time) * 1000
                
                batch_delay = device_to_edge_delay + edge_inference_time
                total_time_delay += batch_delay

        avg_time_delay = round(total_time_delay/2880.0, 3)
        return avg_time_delay

    def calculate_pure_cloud_delay(self, sim_dataset):

        total_time_delay = 0
        total_samples = len(sim_dataset)
        

        indices = list(range(total_samples))
        
        for round_idx in range(0, total_samples, 48):
            round_indices = indices[round_idx:round_idx + 48]
            
            for node_idx in range(3):
                start_idx = node_idx * 16
                end_idx = min(start_idx + 16, len(round_indices))
                if start_idx >= len(round_indices):
                    break
                    
                node_indices = round_indices[start_idx:end_idx]
                
                node_images = []
                for idx in node_indices:
                    img, _ = sim_dataset[idx]
                    node_images.append(img)
                
                if len(node_images) == 0:
                    continue
                
                batch_images = torch.stack(node_images).to(device)
                num_images = len(batch_images)
                

                device_to_edge_delay = self.delay_calculator.calculate_transmission_delay(
                    num_images, from_device_to_edge=True
                )
                

                edge_to_cloud_delay = self.delay_calculator.calculate_transmission_delay(
                    num_images, from_device_to_edge=False
                )
                

                start_time = time.time()
                with torch.no_grad():
                    _ = self.cloud_model(batch_images)
                end_time = time.time()
                cloud_inference_time = (end_time - start_time) * 1000
                
                batch_delay = device_to_edge_delay + edge_to_cloud_delay + cloud_inference_time
                total_time_delay += batch_delay
        
        avg_time_delay = round(total_time_delay/2880.0, 3)
        return avg_time_delay

def evaluate_paradigm_test_only(framework, test_loader, paradigm, threshold_):
    all_predictions = []
    all_labels = []
        

    framework.current_node_idx = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
            
        if paradigm == 'edge':
            _, _, predictions, _ = framework.edge_inference(images)
                
        elif paradigm == 'cloud':
            _, predictions, _ = framework.cloud_inference(images)
                
        elif paradigm in  ['confidence_0.1', 'confidence_0.2', 'confidence_0.3','confidence_no_update_0.2']:
            predictions, _, _ = framework.collaborative_inference(
                images, threshold=threshold_
            )
            
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy


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
    
    # Load the datasets
    sim_dataset = ASDDataset(simulation_data_path, transform=transform)
    test_dataset = ASDDataset(testing_data_path, transform=transform)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    print("Starting Paradigm Comparison Experiments...")
    print("\n" + "="*50)
    print("Paradigm Comparison Experiment")
    print("="*50)

    paradigms = ['edge', 'cloud', 'confidence_0.1', 'confidence_0.2', 'confidence_0.3', 'confidence_no_update_0.2']
    paradigm_names = ['Edge 1 + Edge 2 + Edge 3', 'Pure Cloud', 'CCEKL0.1', 'CCEKL0.2', 'CCEKL0.3', 'CCEKL_noupdate0.2']
    
    results = []
    
    for i, paradigm in enumerate(paradigms):
        print(f"Evaluating {paradigm_names[i]}...")
        
        framework = CloudEdgeFramework(
            "load_models_experiment2/best_edge_model.pth",
            "load_models_experiment2/best_cloud_model.pth"
        )
        
        threshold = 0   # initialize the value of threshold into 0
        if paradigm == 'confidence_0.1':
            upload_prop_sim, time_delay_sim = framework.simulate_communication_rounds(sim_dataset, threshold=0.1)
            threshold = 0.1

        elif paradigm == 'confidence_0.2':
            upload_prop_sim, time_delay_sim = framework.simulate_communication_rounds(sim_dataset, threshold=0.2)
            threshold = 0.2

        elif paradigm == 'confidence_0.3':
            upload_prop_sim, time_delay_sim = framework.simulate_communication_rounds(sim_dataset, threshold=0.3)
            threshold = 0.3

        elif paradigm == 'confidence_no_update_0.2':
            upload_prop_sim, time_delay_sim = framework.calculate_collaborative_no_update_delay(sim_dataset, threshold=0.2)
            threshold = 0.2

        elif paradigm == 'edge':
            upload_prop_sim = 0.0  # no uploads
            time_delay_sim = framework.calculate_pure_edge_delay(sim_dataset)

        else:  # pure cloud
            upload_prop_sim = 1.0  # all upload
            time_delay_sim = framework.calculate_pure_cloud_delay(sim_dataset)

        accuracy = evaluate_paradigm_test_only(
            framework, test_loader, paradigm, threshold
        )
        
        results.append({
            'paradigm': paradigm_names[i],
            'upload_proportion': upload_prop_sim, 
            'accuracy': accuracy,  
            'time_delay_ms': time_delay_sim})  
        
        print(f"{paradigm_names[i]}:")
        print(f"  Upload Proportion: {upload_prop_sim:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Time Delay: {time_delay_sim:.2f} ms")
        print()
    
if __name__ == "__main__":
    main()