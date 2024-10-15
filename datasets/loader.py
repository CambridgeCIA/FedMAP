import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, random_split
from PIL import Image
import os
from torchvision import transforms

def load_individual_dataset(client_id):
    data = pd.read_csv(f'./datasets/synthetic/partitions_{client_id}.csv')    
   
    LABEL = 'ClassCategory_0'
   
    X = data.drop(columns=[LABEL])
    y = data[LABEL]

    validation_ratio = 0.3

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_ratio, random_state=42)

    # Apply scaling
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) 

    # Convert the Pandas DataFrames to PyTorch tensors 
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader

class Office31Dataset(Dataset):
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
    
    id = int(id)
    dataset_dir = './datasets/public/office-31'
    remove_ds_store(dataset_dir)
    if id == 0:
        domain = 'amazon'
    elif id == 1:
        domain = 'dslr'
    else:
        domain = 'webcam'
        
    print(f"Loading data for domain: {domain}")

    # Create the full dataset instance
    full_dataset = Office31Dataset(dataset_dir, domain, transform=None)  # We will apply transforms later

    # Split the dataset into training and validation sets
    total_images = len(full_dataset)
    train_size = int(0.7 * total_images)
    val_size = total_images - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Applying transforms using a custom wrapper that applies transformation
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

    # Create the dataloaders
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader, val_dataloader

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        # Check if '.DS_Store' is in the files list
        if '.DS_Store' in files:
            ds_store_path = os.path.join(root, '.DS_Store')
            try:
                # Remove the .DS_Store file
                os.remove(ds_store_path)
                print(f"Removed: {ds_store_path}")
            except Exception as e:
                print(f"Error removing {ds_store_path}: {e}")
