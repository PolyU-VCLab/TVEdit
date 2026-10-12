from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import random
import time
import numpy as np
from datasets import load_dataset
import torch.distributed as dist
from .data import TVEditDataset
from lightning.pytorch.strategies import DDPStrategy
from .model import TVEditModel
from .callbacks import TrainingCallback
from datetime import timedelta

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank
def get_global_rank():
    """获取全局进程排名（多机多卡中唯一）"""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    test_config = config["test"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)
        
    if 'use_offset_noise' not in config.keys():
        config['use_offset_noise'] = False

    # Initialize dataset and dataloader
    
    if training_config["dataset"]["type"] == "edit_drag":
        dataset = TVEditDataset(
            os.path.join(training_config["dataset"]["datapath"],training_config["dataset"]["name"]),
            datapath=training_config["dataset"]["datapath"],
            condition_size=(512,512),
            target_size=training_config["train_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
          
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )
    
    # Initialize model
    trainable_model = TVEditModel(
        model_id=config["model_path"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_offset_noise=config["use_offset_noise"],
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config,test_config=test_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks, 
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=True,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        num_nodes=config["num_nodes"],
        devices=config["devices"],
        sync_batchnorm=True,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=2))
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}",exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
