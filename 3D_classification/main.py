import torch
import argparse, os

from torch.utils.data import DataLoader
from vit_pytorch.vit_3d import ViT
from models.vivit import Vivit
from models.vit3d_llama import ViTLLama
from models.vivit_llama import VivitLLama
from sklearn.metrics import roc_auc_score
import numpy as np


import medmnist
from medmnist import INFO

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--mode', type=str, default="train", help='train or test')
parser.add_argument('--checkpoint', type=str, default="./output/best_checkpoint.pth", help='checkpoint path')
parser.add_argument('--dataset', type=str, default="organmnist3d", help='dataset name')
parser.add_argument('--model', type=str, default="vit", help='model name')
parser.add_argument('--llm_mode', type=str, default="llm_in", help='llm residual name')
parser.add_argument('--llm_grad', type=bool, default=False, help='llm grad required')
parser.add_argument('--cuda', type=int, default=6, help='cuda id')

args = parser.parse_args()

print('dataset:', args.dataset)
print('model:', args.model)

device = torch.device('cuda', args.cuda)

parameters_map = {
    "organmnist3d": 11,
    "nodulemnist3d": 2,
    "fracturemnist3d": 3,
    "adrenalmnist3d": 2,
    "vesselmnist3d": 2,
    "synapsemnist3d": 2
}

data_flag = args.dataset
model_name = args.model
num_classes = parameters_map[data_flag]
llm_mode = args.llm_mode

save_path = f"./output/{model_name}/{data_flag}/"
if "llm" in model_name:
    if args.llm_grad:
        save_path = f"./output/{model_name}/{llm_mode}/require_grad/{data_flag}/"
    else:
        save_path = f"./output/{model_name}/{llm_mode}/{data_flag}/"

os.makedirs(save_path, exist_ok=True)

if model_name == "vit":
    v = ViT(
        image_size = 28,          # image size
        frames = 28,               # number of frames
        image_patch_size = 4,     # image patch size
        frame_patch_size = 4,      # frame patch size
        num_classes = num_classes,
        dim = 1024,
        depth = 8,
        heads = 8,
        mlp_dim = 1024,
        dropout = 0,
        emb_dropout = 0
    ).to(device)
elif model_name == "vit_llm":
    v = ViTLLama(
        image_size = 28,          # image size
        frames = 28,               # number of frames
        image_patch_size = 4,     # image patch size
        frame_patch_size = 4,      # frame patch size
        num_classes = num_classes,
        dim = 1024,
        depth = 8,
        heads = 8,
        mlp_dim = 1024,
        dropout = 0,
        emb_dropout = 0,
        model_name=llm_mode,
        requires_grad=args.llm_grad
    ).to(device)
elif model_name == "vit_small":
    v = ViT(
        image_size = 28,          # image size
        frames = 28,               # number of frames
        image_patch_size = 4,     # image patch size
        frame_patch_size = 4,      # frame patch size
        num_classes = num_classes,
        dim = 512,
        depth = 8,
        heads = 8,
        mlp_dim = 512,
        dropout = 0,
        emb_dropout = 0
    ).to(device)
elif model_name == "vit_llm_small":
    v = ViTLLama(
        image_size = 28,          # image size
        frames = 28,               # number of frames
        image_patch_size = 4,     # image patch size
        frame_patch_size = 4,      # frame patch size
        num_classes = num_classes,
        dim = 512,
        depth = 8,
        heads = 8,
        mlp_dim = 512,
        dropout = 0,
        emb_dropout = 0,
        model_name=llm_mode,
        requires_grad=args.llm_grad
    ).to(device)
elif model_name == "vit_large":
    v = ViT(
        image_size = 28,          # image size
        frames = 28,               # number of frames
        image_patch_size = 4,     # image patch size
        frame_patch_size = 4,      # frame patch size
        num_classes = num_classes,
        dim = 2048,
        depth = 8,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0,
        emb_dropout = 0
    ).to(device)
elif model_name == "vit_llm_large":
    v = ViTLLama(
        image_size = 28,          # image size
        frames = 28,               # number of frames
        image_patch_size = 4,     # image patch size
        frame_patch_size = 4,      # frame patch size
        num_classes = num_classes,
        dim = 2048,
        depth = 8,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0,
        emb_dropout = 0,
        model_name=llm_mode,
        requires_grad=args.llm_grad
    ).to(device)
elif model_name == "vivit":
    v = Vivit(
        image_size=28,  # image size
        frames=28,  # number of frames
        image_patch_size=4,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=num_classes,
        dim=1024,
        spatial_depth=8,  # depth of the spatial transformer
        temporal_depth=8,  # depth of the temporal transformer
        heads=8,
        mlp_dim=1024
    ).to(device)
elif model_name == "vivit_llm":
    v = VivitLLama(
        image_size=28,  # image size
        frames=28,  # number of frames
        image_patch_size=4,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=num_classes,
        dim=1024,
        spatial_depth=8,  # depth of the spatial transformer
        temporal_depth=8,  # depth of the temporal transformer
        heads=8,
        mlp_dim=1024,
        model_name=llm_mode,
        requires_grad=args.llm_grad
    ).to(device)
elif model_name == "vivit_small":
    v = Vivit(
        image_size=28,  # image size
        frames=28,  # number of frames
        image_patch_size=4,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=num_classes,
        dim=512,
        spatial_depth=4,  # depth of the spatial transformer
        temporal_depth=4,  # depth of the temporal transformer
        heads=8,
        mlp_dim=512
    ).to(device)
elif model_name == "vivit_llm_small":
    v = VivitLLama(
        image_size=28,  # image size
        frames=28,  # number of frames
        image_patch_size=4,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=num_classes,
        dim=512,
        spatial_depth=4,  # depth of the spatial transformer
        temporal_depth=4,  # depth of the temporal transformer
        heads=8,
        mlp_dim=512,
        model_name = llm_mode,
        requires_grad=args.llm_grad
    ).to(device)
elif model_name == "vivit_large":
    v = Vivit(
        image_size=28,  # image size
        frames=28,  # number of frames
        image_patch_size=4,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=num_classes,
        dim=2048,
        spatial_depth=8,  # depth of the spatial transformer
        temporal_depth=8,  # depth of the temporal transformer
        heads=8,
        mlp_dim=2048
    ).to(device)
elif model_name == "vivit_llm_large":
    v = VivitLLama(
        image_size=28,  # image size
        frames=28,  # number of frames
        image_patch_size=4,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=num_classes,
        dim=2048,
        spatial_depth=16,  # depth of the spatial transformer
        temporal_depth=16,  # depth of the temporal transformer
        heads=8,
        mlp_dim=2048,
        model_name = llm_mode,
        requires_grad=args.llm_grad
    ).to(device)


num_params = sum(p.numel() for p in v.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

model_size = sum(p.numel() * p.element_size() for p in v.parameters() if p.requires_grad)
print(f"Model size: {model_size / (1024 * 1024)} MB")

download = True

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(v.parameters(), lr=1e-5) # 5


criterion = torch.nn.CrossEntropyLoss()
best_accuracy = 0
best_auc = 0

if args.mode == "train":
    with open(save_path + "log.txt", "a") as f:
        for epoch in range(100):
            for step, batch in enumerate(train_loader):
                video, target = batch
                target = target.to(device)
                one_hot = torch.zeros(target.size(0), num_classes, dtype=torch.float32, device=device)
                one_hot.scatter_(1, target, 1)  # Use scatter_ to set indices to 1

                video_expand = video.float().repeat(1, 3, 1, 1, 1).to(device)  # (batch, channels, frames, height, width)
                preds = v(video_expand)
                loss = criterion(preds, one_hot)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']
                if step % 10 == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Learning Rate: {current_lr}")

            total_images, total_correct = 0, 0
            all_predictions, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    video, labels = batch
                    video_expand = video.float().repeat(1, 3, 1, 1, 1).to(device)  # (batch, channels, frames, height, width)
                    outputs = v(video_expand)

                    _, predicted = torch.max(outputs, 1)
                    total_images += labels.size(0)
                    total_correct += (predicted == labels.squeeze().to(device)).sum().item()

                    probabilities = torch.softmax(outputs, dim=1)
                    all_predictions.extend(probabilities.cpu().numpy())
                    all_labels.extend(labels.numpy())

            accuracy = total_correct / total_images

            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels).squeeze()
            auc = 0
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_predictions[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr')

            # Save the latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': v.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy
            }, save_path + 'latest_checkpoint.pth')

            # Save the best checkpoint based on validation accuracy
            if accuracy >= best_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': v.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path + 'best_acc_checkpoint.pth')

            if auc >= best_auc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': v.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path + 'best_auc_checkpoint.pth')

            best_accuracy = max(best_accuracy, accuracy)
            best_auc = max(best_auc, auc)

            f.write(f'Epoch: {epoch}, Validation Accuracy: {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}, AUC: {auc:.4f}, best AUC: {best_auc:.4f}\n')
            print(f'Epoch: {epoch}, Validation Accuracy: {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}, AUC: {auc:.4f}, best AUC: {best_auc:.4f}')

else:
    checkpoint = torch.load(args.checkpoint)
    v.load_state_dict(checkpoint['model_state_dict'])
    total_images, total_correct = 0, 0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            video, labels = batch
            video_expand = video.float().repeat(1, 3, 1, 1, 1).to(device)  # (batch, channels, frames, height, width)
            outputs = v(video_expand)

            _, predicted = torch.max(outputs, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels.squeeze().to(device)).sum().item()

            probabilities = torch.softmax(outputs, dim=1)
            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = total_correct / total_images
    print(f'Test Accuracy: {accuracy:.4f}')

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels).squeeze()
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_predictions[:,1])
        print(f'AUC: {auc:.4f}')
    else:
        ovr_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr')
        ovo_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovo')
        print(f'ovr AUC: {ovr_auc:.4f}, ovo AUC: {ovo_auc:.4f}')
