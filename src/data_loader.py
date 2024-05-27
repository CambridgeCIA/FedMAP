import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import gdown

def load_data_synthetic(path, label):
    """
    Load and preprocess synthetic data from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        label (str): Name of the label column.
        
    Returns:
        tuple: Tuple containing trainloader and testloader.
    """
    data = pd.read_csv(path)
    
    X = data.drop(columns=[label])
    y = data[label]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) 

    # Convert the data to PyTorch tensors 
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader

class Office31Dataset(Dataset):
    """Custom dataset for Office-31."""
    
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = os.path.join(root_dir, domain, 'images')
        self.transform = transform
        self.images = []
        self.labels = []
        
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Provided path '{self.root_dir}' does not exist or is not accessible.")
        
        self.classes = sorted(os.listdir(self.root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    
def load_image_data(id):
    """
    Load and preprocess image data from the Office-31 dataset.
    
    Args:
        id (int): Client ID to determine the domain.
        
    Returns:
        tuple: Tuple containing train_dataloader and val_dataloader.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Define the data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Set the path to the Office-31 dataset directory
    dataset_dir = './data/public/office-31'
    
    # Determine the domain based on the client ID
    if id == 0 or id == 3:
        domain = 'amazon'
    elif id == 1 or id == 4:
        domain = 'dslr'
    else:
        domain = 'webcam'
        
    print(f"Loading data for domain: DIR {dataset_dir} Domain {domain} Client {id}")

    # Create the full dataset instance
    full_dataset = Office31Dataset(dataset_dir, domain, transform=None)  # Transforms applied later

    # Split the dataset into training and validation sets
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Custom wrapper to apply transformations
    class TransformingDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __getitem__(self, index):
            img, label = self.dataset[index]
            img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.dataset)

    # Wrap datasets with corresponding transformations
    train_dataset = TransformingDataset(train_dataset, data_transforms['train'])
    val_dataset = TransformingDataset(val_dataset, data_transforms['val'])

    # Create the data loaders
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader
