import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from voc_dataset import VOCDataset
from sklearn.manifold import TSNE

# Define class names and colors
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# Configurations
batch_size = 32
image_size = (224, 224)
num_samples = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'checkpoint-model-epoch50.pth'

# Load dataset and create a subset
test_dataset = VOCDataset(split='test', size=image_size[0])
random.seed(42)
indices = random.sample(range(len(test_dataset)), num_samples)
subset = Subset(test_dataset, indices)
subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load trained model
model = torch.load(checkpoint_path, map_location=device)
print(f"Loaded model onto {device} from checkpoint!")
model.eval()

# Feature extraction function
def extract_features(model, loader):
    features, gt_classes = [], []
    print("Extracting features from sampled images...")
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            batch_features = model(images).squeeze()
            features.append(batch_features.cpu().numpy())
            gt_classes.append(labels.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(gt_classes, axis=0)

# Compute colors for visualization
def compute_colors(gt_classes):
    colors = []
    for label_vector in gt_classes:
        active_classes = np.where(label_vector > 0)[0]
        colors.append(np.mean(CLASS_COLORS[active_classes], axis=0) if len(active_classes) > 1 else CLASS_COLORS[active_classes[0]])
    return np.array(colors)

# Apply t-SNE and visualize
def plot_tsne(features, colors):
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 9))
    plt.style.use('seaborn-darkgrid')
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, s=30, alpha=0.8, edgecolor='k')
    
    # Legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
                                   markerfacecolor=CLASS_COLORS[i], markersize=10, markeredgecolor='k')
                       for i, cls in enumerate(CLASS_NAMES)]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), title="Classes", fontsize='small')
    
    plt.title("t-SNE of Image Features for PASCAL Test Set", fontsize=15)
    plt.xlabel("t-SNE Dim 1", fontsize=12)
    plt.ylabel("t-SNE Dim 2", fontsize=12)
    plt.tight_layout()
    plt.savefig('g_tsne_resnet_ft.png')

# Run feature extraction and visualization
features, gt_classes = extract_features(model, subset_loader)
image_colors = compute_colors(gt_classes)
plot_tsne(features, image_colors)