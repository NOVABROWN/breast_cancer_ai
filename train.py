# import argparse
# import collections
# import torch
# import numpy as np
# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from parse_config import ConfigParser
# from trainer import Trainer
# from utils import prepare_device


# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)

# def main(config):
#     logger = config.get_logger('train')

#     # setup data_loader instances
#     data_loader = config.init_obj('data_loader', module_data)
#     valid_data_loader = data_loader.split_validation()

#     # build model architecture, then print to console
#     model = config.init_obj('arch', module_arch)
#     logger.info(model)

#     # prepare for (multi-device) GPU training
#     device, device_ids = prepare_device(config['n_gpu'])
#     model = model.to(device)
#     if len(device_ids) > 1:
#         model = torch.nn.DataParallel(model, device_ids=device_ids)

#     # get function handles of loss and metrics
#     criterion = getattr(module_loss, config['loss'])
#     metrics = [getattr(module_metric, met) for met in config['metrics']]

#     # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
#     lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

#     trainer = Trainer(model, criterion, metrics, optimizer,
#                       config=config,
#                       device=device,
#                       data_loader=data_loader,
#                       valid_data_loader=valid_data_loader,
#                       lr_scheduler=lr_scheduler)

#     trainer.train()


# if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='PyTorch Template')
#     args.add_argument('-c', '--config', default=None, type=str,
#                       help='config file path (default: None)')
#     args.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     args.add_argument('-d', '--device', default=None, type=str,
#                       help='indices of GPUs to enable (default: all)')

#     # custom cli options to modify configuration from default values given in json file.
#     CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
#     options = [
#         CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
#         CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
#     ]
#     config = ConfigParser.from_args(args, options)
#     main(config)


"""
Simplified training script for Breast Cancer Detection
Usage: python train.py --data_dir <path_to_organized_data> --epochs 20
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy

def train_model(data_dir, save_path, epochs=20, batch_size=32, lr=0.001):
    """Train breast cancer classification model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load datasets
    print(f"\nLoading datasets from: {data_dir}")
    image_datasets = {
        x: datasets.ImageFolder(f'{data_dir}/{x}', data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=0)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"\nDataset Info:")
    print(f"  Classes: {class_names}")
    print(f"  Training images: {dataset_sizes['train']}")
    print(f"  Validation images: {dataset_sizes['val']}")
    
    # Model (using ResNet18 for faster training)
    print("\nInitializing model...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: benign, malignant
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 70)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 70)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize():5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'  *** New best model! Accuracy: {best_acc:.4f} ***')
    
    print("\n" + "=" * 70)
    print(f'Training complete!')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load and save best model
    model.load_state_dict(best_model_wts)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'best_acc': best_acc
    }, save_path)
    
    print(f'\nBest model saved to: {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train breast cancer detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to organized dataset')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.save_path, args.epochs, args.batch_size, args.lr)
