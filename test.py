# import argparse
# import torch
# from tqdm import tqdm
# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from parse_config import ConfigParser


# def main(config):
#     logger = config.get_logger('test')

#     # setup data_loader instances
#     data_loader = getattr(module_data, config['data_loader']['type'])(
#         config['data_loader']['args']['data_dir'],
#         batch_size=512,
#         shuffle=False,
#         validation_split=0.0,
#         training=False,
#         num_workers=2
#     )

#     # build model architecture
#     model = config.init_obj('arch', module_arch)
#     logger.info(model)

#     # get function handles of loss and metrics
#     loss_fn = getattr(module_loss, config['loss'])
#     metric_fns = [getattr(module_metric, met) for met in config['metrics']]

#     logger.info('Loading checkpoint: {} ...'.format(config.resume))
#     checkpoint = torch.load(config.resume)
#     state_dict = checkpoint['state_dict']
#     if config['n_gpu'] > 1:
#         model = torch.nn.DataParallel(model)
#     model.load_state_dict(state_dict)

#     # prepare model for testing
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()

#     total_loss = 0.0
#     total_metrics = torch.zeros(len(metric_fns))

#     with torch.no_grad():
#         for i, (data, target) in enumerate(tqdm(data_loader)):
#             data, target = data.to(device), target.to(device)
#             output = model(data)

#             #
#             # save sample images, or do something with output here
#             #

#             # computing loss, metrics on test set
#             loss = loss_fn(output, target)
#             batch_size = data.shape[0]
#             total_loss += loss.item() * batch_size
#             for i, metric in enumerate(metric_fns):
#                 total_metrics[i] += metric(output, target) * batch_size

#     n_samples = len(data_loader.sampler)
#     log = {'loss': total_loss / n_samples}
#     log.update({
#         met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
#     })
#     logger.info(log)


# if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='PyTorch Template')
#     args.add_argument('-c', '--config', default=None, type=str,
#                       help='config file path (default: None)')
#     args.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     args.add_argument('-d', '--device', default=None, type=str,
#                       help='indices of GPUs to enable (default: all)')

#     config = ConfigParser.from_args(args)
#     main(config)


"""
Test script for Breast Cancer Detection
Usage: python test.py --model_path best_model.pth --test_dir <path_to_test_folder>
"""
import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def test_model(model_path, test_dir, batch_size=32):
    """Test the trained model and print detailed metrics"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', ['benign', 'malignant'])
    print(f"Classes: {class_names}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print(f"\nLoading test data from: {test_dir}")
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Test images: {len(test_dataset)}")
    
    # Testing
    print("\n" + "=" * 70)
    print("Running inference on test set...")
    print("=" * 70)
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    # Results
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    print(f"\n{'-'*70}")
    print("Confusion Matrix:")
    print(f"{'-'*70}")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n              Predicted")
    print(f"              {class_names[0]:>10s} {class_names[1]:>10s}")
    print(f"Actual")
    for i, cls in enumerate(class_names):
        print(f"{cls:>10s}    {cm[i][0]:>5d}      {cm[i][1]:>5d}")
    
    print(f"\n{'-'*70}")
    print("Classification Report:")
    print(f"{'-'*70}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test breast cancer detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test dataset folder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    test_model(args.model_path, args.test_dir, args.batch_size)