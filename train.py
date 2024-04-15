import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import wandb
import argparse
from XMP_model import XMP

wandb.init(project="XMP_train", name = "XMP")

parser = argparse.ArgumentParser(description='Set training details')

# 512
# python train.py -scen 1 -size 512 -name XMP_scen1_512 -gpu 3 -emb_dim 128 -depth 3 -sm_cnn_size 11 -sm_dim 256 -sm_token_size 4 -lg_cnn_size 11 -lg_dim 512 -lg_token_size 16
# python train.py -scen 2 -size 512 -name XMP_scen2_512 -gpu 3 -emb_dim 512 -depth 2 -sm_cnn_size 27 -sm_dim 256 -sm_token_size 16 -lg_cnn_size 35 -lg_dim 512 -lg_token_size 32
# python train.py -scen 3 -size 512 -name XMP_scen3_512 -gpu 1 -emb_dim 512 -depth 2 -sm_cnn_size 27 -sm_dim 256 -sm_token_size 32 -lg_cnn_size 43 -lg_dim 512 -lg_token_size 128
# python train.py -scen 4 -size 512 -name XMP_scen4_512 -gpu 1 -emb_dim 512 -depth 2 -sm_cnn_size 19 -sm_dim 256 -sm_token_size 16 -lg_cnn_size 35 -lg_dim 512 -lg_token_size 64
# python train.py -scen 5 -size 512 -name XMP_scen5_512 -gpu 3 -emb_dim 512 -depth 3 -sm_cnn_size 3 -sm_dim 128 -sm_token_size 16 -lg_cnn_size 19 -lg_dim 256 -lg_token_size 64
# python train.py -scen 6 -size 512 -name XMP_scen6_512 -gpu 3 -emb_dim 512 -depth 3 -sm_cnn_size 3 -sm_dim 256 -sm_token_size 16 -lg_cnn_size 19 -lg_dim 512 -lg_token_size 64

# 4k
# python train.py -scen 1 -size 4k -name XMP_scen1_4k -gpu 3 -emb_dim 512 -depth 3 -sm_cnn_size 19 -sm_dim 128 -sm_token_size 64 -lg_cnn_size 19 -lg_dim 512 -lg_token_size 256
# python train.py -scen 2 -size 4k -name XMP_scen2_4k -gpu 3 -emb_dim 256 -depth 2 -sm_cnn_size 11 -sm_dim 256 -sm_token_size 128 -lg_cnn_size 11 -lg_dim 512 -lg_token_size 512
# python train.py -scen 3 -size 4k -name XMP_scen3_4k -gpu 3 -emb_dim 512 -depth 2 -sm_cnn_size 19 -sm_dim 128 -sm_token_size 128 -lg_cnn_size 35 -lg_dim 256 -lg_token_size 512
# python train.py -scen 4 -size 4k -name XMP_scen4_4k -gpu 2 -emb_dim 512 -depth 4 -sm_cnn_size 27 -sm_dim 256 -sm_token_size 128 -lg_cnn_size 27 -lg_dim 512 -lg_token_size 512
# python train.py -scen 5 -size 4k -name XMP_scen5_4k -gpu 3 -emb_dim 512 -depth 2 -sm_cnn_size 3 -sm_dim 128 -sm_token_size 128 -lg_cnn_size 19 -lg_dim 512 -lg_token_size 512
# python train.py -scen 6 -size 4k -name XMP_scen6_4k -gpu 3 -emb_dim 512 -depth 2 -sm_cnn_size 19 -sm_dim 128 -sm_token_size 64 -lg_cnn_size 35 -lg_dim 256 -lg_token_size 256

parser.add_argument("-scen", "--training_scenario", help="choose dataset scenario", type=int, required=True)
parser.add_argument("-size", "--data_size", help="choose data-block size", type=str, required=True) # 512 or 4k
parser.add_argument("-name", "--checkpoint_name", help="checkpoint_name", type=str, required=True)
parser.add_argument("-gpu", "--gpu_id", help="gpu_id", type=int, required=True)
parser.add_argument("-emb_dim", "--value_embedding", help="value_embedding", type=int, required=True) # default : 512
parser.add_argument("-depth", "--multi_scale_Performer_depth", help="multi_scale_Performer_depth", type=int, required=True) # 2, 3, 4
parser.add_argument("-sm_cnn_size", "--small_cnn_kernel_size", help="small_cnn_kernel_size", type=int, required=True)
parser.add_argument("-sm_dim", "--small_token_dimension", help="small_token_dimension", type=int, required=True)
parser.add_argument("-sm_token_size", "--small_token_size", help="small_token_size", type=int, required=True)
parser.add_argument("-lg_cnn_size", "--large_cnn_kernel_size", help="large_cnn_kernel_size", type=int, required=True)
parser.add_argument("-lg_dim", "--large_token_dimension", help="large_token_dimension", type=int, required=True)
parser.add_argument("-lg_token_size", "--large_token_size", help="large_token_size", type=int, required=True)
parser.add_argument("-epochs", "--total_epochs", help="total_epochs", type=int, default=100)
parser.add_argument("-lr", "--first_lr", help="first_lr", type=float, default=0.001)
parser.add_argument("-batch_size", "--dataset_batch_size", help="dataset_batch_size", type=int, default=128)
parser.add_argument("-patience", "--early_stopping", help="early_stopping", type=int, default=11)

args = parser.parse_args()

training_scenario = args.training_scenario
data_size = args.data_size
checkpoint = args.checkpoint_name
gpu_id = args.gpu_id
emb_dim = args.value_embedding
depth = args.multi_scale_Performer_depth
small_cnn_kernel_size = args.small_cnn_kernel_size
small_token_dimension = args.small_token_dimension
small_token_size = args.small_token_size
large_cnn_kernel_size = args.large_cnn_kernel_size
large_token_dimension = args.large_token_dimension
large_token_size = args.large_token_size
epochs = args.total_epochs
first_lr = args.first_lr
batch_size = args.dataset_batch_size
early_stopping = args.early_stopping

wandb.config.update(args)

scenario_name = data_size + '_' + str(training_scenario)
basic_path = './' + scenario_name + '/'

if data_size=='512':
    print('=========== data_size = 512 ===========')
    seq_size = 512
elif data_size=='4k':
    print('=========== data_size = 4096 ===========')
    seq_size = 4096
else:
    raise ValueError('=========== Invalid data size! ===========')

num_class_list = [75,11,25,5,2,2]
num_classes = num_class_list[training_scenario-1]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fifty_dataset(Dataset):
    def __init__(self, data, label):
        self.x_data = torch.from_numpy(data.copy()).long()
        self.y_data = torch.from_numpy(label.copy()).long()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

def train_model(epochs, model, criterion, optimizer, scheduler ,train_loader, val_loader, patience=10):
    best_val_loss = float('inf')
    best_accuracy = float(0)
    epochs_without_improvement = 0

    for epoch in range(epochs):
        running_correct = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            pred_out = outputs.argmax(dim=1, keepdim=True)
            running_correct += pred_out.eq(labels.view_as(pred_out)).sum().item()

        running_loss /= len(train_loader.dataset)
        running_accuracy = 100 * running_correct / len(train_loader.dataset)

        # Evaluate the model on the validation set
        print('Evaluation start')
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)

        print(f'Epoch: {epoch + 1}, Training Loss: {running_loss:.4f}, Training Accuracy: {running_accuracy:.2f}%')
        print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            print('Saving the model...')
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model

            name_v1 = './best_model/'+ checkpoint + '_v1.pth'
            name_v2 = './best_model/'+ checkpoint + '_v2.pth'

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,}, name_v1)

            torch.save(model, name_v2)

        else:
            epochs_without_improvement += 1

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy

        print(f'Epoch: {epoch + 1}, Best Accuracy: {best_accuracy:.2f}%')
        print('========================================================')

        # Check if early stopping criteria are met
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch + 1} epochs without improvement.')
            break

        wandb.log({
        "Train Loss": running_loss,
        "Train Acc": running_accuracy,
        "Validation Loss": val_loss,
        "Validation Acc": val_accuracy
        })

        scheduler.step(val_loss)
    print('Finished Training')
    return model

def evaluate_model(model, criterion, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100 * correct / len(val_loader.dataset)
    return val_loss, accuracy

# Load datasets
train_path = basic_path + 'train.npz'
val_path = basic_path + 'val.npz'

train_np_data = np.load(train_path)
val_np_data = np.load(val_path)

train_data = Fifty_dataset(train_np_data['x'], train_np_data['y'])
val_data = Fifty_dataset(val_np_data['x'], val_np_data['y'])

# Dataloaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

# Set up the model, loss criterion, and optimizer
model = XMP(
    seq_len = seq_size,
    emb_dim = emb_dim,
    num_classes = num_classes,
    depth = depth,
    sm_cnn_kernel_size = small_cnn_kernel_size,
    sm_dim = small_token_dimension,
    sm_token_size = small_token_size,
    lg_cnn_kernel_size = large_cnn_kernel_size,
    lg_dim = large_token_dimension,
    lg_token_size = large_token_size,
    ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=first_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

model = train_model(epochs, model, criterion, optimizer, scheduler, train_loader, val_loader, patience=early_stopping)

wandb.finish()
