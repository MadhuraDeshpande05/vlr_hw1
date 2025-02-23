import torch
from torchvision import datasets, transforms
import random
import numpy as np
from PIL import Image
from voc_dataset import VOCDataset
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision
from train_q2 import ResNet

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))  # Assign distinct colors to each class
CLASS_COLORS = plt.cm.tab20(np.arange(20))
batch_size = 32
image_size = (224, 224)
num_samples = 1000

test_dataset = VOCDataset(split='test', size=image_size[0])  

random.seed(42)
indices = random.sample(range(len(test_dataset)), num_samples)  
subset = Subset(test_dataset, indices)  
subset_loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=2)  

# model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# model = torch.nn.Sequential(*list(model.children())[:-1])  

checkpoint_path = '/home/ubuntu/vlr_hw1/checkpoint-resnet18-epoch50.pth' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(checkpoint_path, weights_only = False, map_location=device)
print("Loaded model onto {} from checkpoint!".format('cuda' if next(model.parameters()).is_cuda else 'cpu'))
model.eval()

# Extract features and ground truth labels
features = []
gt_classes = []

print("Extracting features from sampled images...")
with torch.no_grad():
    for images, labels, _ in subset_loader:
        images = images.to(device)
        batch_features = model(images).squeeze()  # (batch_size, feature_dim)
        features.append(batch_features.cpu().numpy())
        gt_classes.append(labels.cpu().numpy())

# Combine features and labels into arrays
features = np.concatenate(features, axis=0)  # Shape: (num_samples, feature_dim)
gt_classes = np.concatenate(gt_classes, axis=0)  # Shape: (num_samples, NUM_CLASSES)

# Compute colors for images based on GT classes
image_colors = []
for label_vector in gt_classes:
    active_classes = np.where(label_vector > 0)[0]  # Find active class indices
    if len(active_classes) == 1:
        # Single class: use corresponding color
        image_colors.append(CLASS_COLORS[active_classes[0]])
    else:
        # Multiple classes: compute mean color
        mean_color = np.mean(CLASS_COLORS[active_classes], axis=0)
        image_colors.append(mean_color)

image_colors = np.array(image_colors)



# Apply t-SNE for 2D projection
print("Computing t-SNE projection...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
features_2d = tsne.fit_transform(features)


# Plot t-SNE results
plt.figure(figsize=(16, 10))
plt.style.use('ggplot')  # Replace with a valid Matplotlib style
print("i am here")

scatter = plt.scatter(
    features_2d[:, 0], 
    features_2d[:, 1], 
    c=image_colors, 
    s=30, 
    alpha=0.95, 
     # Add edge color to make points stand out
)

# Add legend for classes
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=class_name,
               markerfacecolor=CLASS_COLORS[cls_id], markersize=10, alpha=0.8, markeredgecolor='k')
    for cls_id, class_name in enumerate(CLASS_NAMES)
]
plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.05, 1), title="Classes", fontsize='small')

# Add labels and title
plt.title("t-SNE Question 2", fontsize=16)
plt.xlabel("t-SNE Feature Similarity Axis 1", fontsize=8)
plt.ylabel("t-SNE Feature Similarity Axis 2", fontsize=8)
plt.tight_layout()

# Show the plot
plt.show()

plt.savefig('me_tsne_resnet_ft.png')
