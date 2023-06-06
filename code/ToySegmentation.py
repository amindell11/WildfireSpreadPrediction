# %%
import wandb
wandb.login()

# %%z
import math
import random
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
torch.set_grad_enabled(False);

#import intel_extension_for_pytorch
import logging
import sys
from pathlib import Path
import os
import time

from torch import optim
from tqdm import tqdm
os.environ['WANDB_NOTEBOOK_NAME'] = 'ToySegmentation.ipynb'
device = torch.device("cuda") 

import gc

gc.collect()

torch.cuda.empty_cache()


# %%
def get_dataloader(is_train, batch_size, slice=5):
    transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    sub_dataset = torch.utils.data.Subset(trainset, indices=range(0, len(trainset), slice))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size, shuffle=True if is_train else False, pin_memory=True, num_workers=2)
    
    return loader

# %%
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
class demodetr(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(10, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
    
        # propagate through the transformer
        srcin = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        batch_size_internal = srcin.shape[1]
        tgtin = self.query_pos.unsqueeze(1).repeat(1, batch_size_internal, 1)
        

        h = self.transformer(srcin,
                             tgtin).transpose(0, 1)
        
        ph = self.linear_class(h)

        #l = ph.softmax(2)

        keep = ph.max(-1)[0]

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': ph, 
                'pred_label_strengths': keep,
                'pred_boxes': self.linear_bbox(h).sigmoid()}

# %%
def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

# %%
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size))])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

# %%
# Launch 5 experiments, trying different dropout rates
for _ in range(1):
    # 🐝 initialise a wandb run
    wandb.init(
        project="ToySegment",
        config={
            "epochs": 100,
            "batch_size": 200,
            "lr": 1e-6
            })
    
    # Copy your config 
    config = wandb.config

    # Get the data
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
    
    # A simple MLP model
    model = demodetr(num_classes=10)
    model.to(device)
    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

   # Training
    example_ct = 0
    step_ct = 0
    for epoch in range(config.epochs):
        model.train()
        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)
            #labels = labels.float()
            
            outputs = model(images)
            
            train_loss = loss_func(outputs["pred_label_strengths"], labels)
            train_loss.requires_grad_()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            example_ct += len(images)
            metrics = {"train/train_loss": train_loss, 
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       "train/example_ct": example_ct}
            
            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)
                
            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # If you had a test set, this is how you could log it as a Summary metric

    # 🐝 Close your wandb run 
    wandb.finish()
