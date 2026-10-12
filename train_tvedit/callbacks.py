import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
# import cv2
import torch
import os
from datetime import datetime
import json
try:
    import wandb
except ImportError:
    wandb = None



class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {},test_config :dict = {}):
        self.run_name, self.training_config = run_name, training_config
        self.test_config = test_config
        self.test_size = test_config['test_size']
        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving weights"
            )
            pl_module.save_encoder(
                f"{self.save_path}/{self.run_name}/weight_ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}",
                f"lora_{self.total_steps}",
                batch["condition_type"][                0            ],  # Use the condition type from the current batch
            )
    def point2disk(self,points,H,W):
        points = torch.round(points) # points [n,2]
        # bs = points.shape[0]
        disk_map = torch.zeros((H, W)).long().to(points.device)

        if len(points) == 0:
            return disk_map.unsqueeze(0)

        # Assign values to disk map
        idx = torch.arange(len(points)).to(points.device) + 1
        disk_map[points[:, 1].long(), points[:, 0].long()] = idx
        disk_map = disk_map.unsqueeze(0)
        return disk_map
    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type,
    ):
        
        file_name = [
        ]
        

        
        pl_module.qwen_pipe.transformer.eval()
        pl_module.point_encoder.eval()
        pl_module.controlnet.eval()
        for i, name in enumerate(file_name):
            with open(os.path.join(name,"points.json"), 'r') as f:
                data = json.load(f)
            with open(os.path.join(name,"instruction.json"), 'r') as f:
                instruction = json.load(f)
            points = np.array(data['points'])
            instruction = instruction['instruction']
            num_p = points.shape[0]
            
            test_size = self.test_size

            test_image = Image.open(os.path.join(name,"original.jpg"))
            
            points = remap_coordinates(points,test_image.size,test_size)
            test_image = test_image.resize((test_size,test_size))

            points = torch.tensor(points)
            points = points.view(-1,2,2)
            src_points = points[:10,0,:]
            tgt_points = points[:10,1,:]
            src_disk = self.point2disk(src_points,test_size,test_size)
            tgt_disk = self.point2disk(tgt_points,test_size,test_size)
    

            prompt_ = instruction

            image = pl_module.qwen_pipe(
                prompt=prompt_,
                image=test_image,
                negative_prompt=" ",
                true_cfg_scale=3,
                num_inference_steps=50,
                generator=torch.Generator("cpu").manual_seed(666),
                point_encoder=pl_module.point_encoder, 
                src_disk=src_disk.unsqueeze(0),
                tgt_disk=tgt_disk.unsqueeze(0),
                controlnet = pl_module.casc
            ).images[0]

            image_with_point = Image.open(os.path.join(name,"image_with_points.jpg"))
            image_with_point = image_with_point.resize((test_size,test_size))
            vis_image = visualize_and_concat(image_with_point,image,src_points=src_points.numpy(),tgt_points=tgt_points.numpy())
            image_name=name.split('/')[-1]
            vis_image.save(os.path.join(save_path, f'{self.total_steps}-{image_name}.jpg'))
        
        pl_module.qwen_pipe.transformer.eval()
        pl_module.point_encoder.train()
        pl_module.controlnet.train()
 
 

def remap_coordinates(coordinates, original_size, test_size,inplace=False):


    coordinates = coordinates.astype(np.float64)
    coordinates[:, 0] /= original_size[0]  
    coordinates[:, 0] *= test_size
    coordinates[:, 1] /= original_size[1]  
    coordinates[:, 1] *= test_size
    return coordinates
def visualize_and_concat(
    original_image,
    generated_image,
    src_points,
    tgt_points,
    point_radius=5,
):
    from PIL import ImageDraw
    
    original_vis = original_image.copy()
    generated_vis = generated_image.copy()
    

    src_color = (255, 0, 0)   
    tgt_color = (0, 0, 255)   
    

    draw_generated = ImageDraw.Draw(generated_vis)
    
    for idx in range(len(src_points)):
        src_x, src_y = int(src_points[idx][0]), int(src_points[idx][1])
        tgt_x, tgt_y = int(tgt_points[idx][0]), int(tgt_points[idx][1])

        draw_generated.ellipse(
            [src_x - point_radius, src_y - point_radius, 
             src_x + point_radius, src_y + point_radius],
            fill=src_color
        )
        draw_generated.ellipse(
            [tgt_x - point_radius, tgt_y - point_radius, 
             tgt_x + point_radius, tgt_y + point_radius],
            fill=tgt_color
        )
    
    width, height = original_vis.size
    concat_image = Image.new('RGB', (width * 2, height))
    concat_image.paste(original_vis, (width, 0))
    concat_image.paste(generated_vis, (0, 0))
    
    return concat_image