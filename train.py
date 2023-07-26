import os
import tqdm
import argparse
import configparser
import wandb

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from utils import print_tensor

device = 'cuda'

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(model, train_loader, config):
    train_config = config["Train"]
    if train_config.getboolean("wandb"):
        wandb.init(project="resnet50 STL10")
    train_loader = sample_data(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.getfloat("lr"))
    start_step = train_config.get("start_step", 0)
    pbar = tqdm.trange(train_config.getint("steps"), dynamic_ncols=True, initial=start_step)
    for step in pbar:
        img, gt_bbox, gt_onehot_cat_id = next(train_loader)
        img = img.to(device)
        gt_onehot_cat_id = gt_onehot_cat_id.to(device)

        if train_config.getboolean("debug"):
            print_tensor(img)
            print_tensor(gt_onehot_cat_id)
            sample = make_grid(img, normalize=(0, 1))
            save_image(sample, "sample.png")

        pred = model(img, gt_bbox)
        pred = nn.functional.softmax(pred, dim=-1)
        y = nn.functional.one_hot(y, 10).float()

        loss = criterion(y, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % train_config.getint("log_step") == 0:
            pbar.set_description(f"loss:{loss.item():.6f}")
            if train_config.getboolean("wandb"):
                wandb.log({"loss": loss.item()})

        if step != 0 and step % train_config.getint("save_step") == 0:
            id = train_config.get("id")
            torch.save({
                "model" : model.state_dict(),
                "step" : step
                }, f"checkpoints/{id}/{step}.pt")
            
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--config", default="config.ini")
    argparser.add_argument("--id", required=True)
    argparser.add_argument("--load")
    argparser.add_argument("--wandb", action="store_true", default=False)

    args = argparser.parse_args()

    os.makedirs(f"checkpoints/{args.id}", exist_ok=True)

    config = configparser.ConfigParser()
    config.read(args.config)

    # model
    model_config = config["Model"]
    model_name = model_config.get("model").lower()
    if model_name == "resnet50":
        from resnet50 import ResNet50
        class_num = model_config.getint("class_num")
        model = ResNet50(class_num)
    elif model_name == "torch_resnet50":
        from torch_resnet import resnet50
        model = resnet50()
    elif model_name == "faster_rcnn":
        from faster_rcnn import FasterRCNN
        model = FasterRCNN(backbone="vgg16", n_classes=90)
    else:
        raise ValueError(f"No such model: {model_name}")
    model = model.to(device)
    if args.load:
        ckpt = torch.load(args.load)
        model.load_state_dict(ckpt["model"])
        config["Train"]["start_step"] = ckpt["start_step"]
    
    # data
    data_config = config["Data"]
    data_name = data_config.get("dataset").lower()
    # ImageNet
    if data_name == "imagenet":
        from data import ImageNetDataset
        transform = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dataset = ImageNetDataset("/data/imagenet", split="train", transform=transform)
    
    # STL10
    elif data_name == "stl10":
        from torchvision.datasets import STL10
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = STL10("/data/STL10", split="train", download=True,
                               transform=transform)
    
    # COCO
    elif data_name == "coco":
        from data import COCO
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        train_dataset = COCO("/data/coco", split="train", size=224, transform=transform)
        
    else:
        raise ValueError(f"No such data: {data_name}")
    

    # data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config.get("batch_size")),
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )

    config["Train"]["id"] = args.id
    config["Train"]["wandb"] = str(args.wandb)
    
    train(model, train_loader=train_loader, config=config)
