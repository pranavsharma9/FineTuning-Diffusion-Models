from torchvision import transforms
from diffusers import DDPMPipeline
from diffusers import DDIMScheduler
from datasets import load_dataset
from matplotlib import pyplot as plt
import torch, torchvision
import torch.nn.functional as F
import wandb
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from fastcore.script import call_parse

@call_parse
def train(
    image_size = 256,
    batch_size = 16,
    grad_accumulation_steps = 2,
    num_epochs = 1,
    start_model = "google/ddpm-celebahq-256",
    dataset_name = "Skiittoo/cartoon-faces",
    device = 'cuda',
    model_save_name = 'cartoonface_ddpm',
    wandb_project = 'cartoonddpmfacenew',
    log_samples_every = 10,
    save_model_every = 500,
    ):
        
    wandb.init(project=wandb_project, config=locals())
    image_pipe = DDPMPipeline.from_pretrained(start_model);
    image_pipe.to(device)
    
    dir=start_model+"/scheduler"
    sampling_scheduler = DDIMScheduler.from_config(dir)
    sampling_scheduler.set_timesteps(num_inference_steps=100)

    dataset = load_dataset(dataset_name, split="train")
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def transf(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}
        
    dataset.set_transform(transf)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optim = torch.optim.AdamW(image_pipe.unet.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            clean_img = batch['images'].to(device)

            noise = torch.randn(clean_img.shape).to(clean_img.device)
            bs = clean_img.shape[0]

            timesteps = torch.randint(0, image_pipe.scheduler.num_train_timesteps, (bs,), device=clean_img.device).long()

            noisy_images = image_pipe.scheduler.add_noise(clean_img, noise, timesteps)

            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)

            wandb.log({'loss':loss.item()})
            loss.backward()

            if (step+1)%grad_accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                
            if (step+1)%log_samples_every == 0:
                x = torch.randn(8, 3, 256, 256).to(device) 
                for i, t in tqdm(enumerate(sampling_scheduler.timesteps)):
                    model_input = sampling_scheduler.scale_model_input(x, t)
                    with torch.no_grad():
                        noise_pred = image_pipe.unet(model_input, t)["sample"]
                    x = sampling_scheduler.step(noise_pred, t, x).prev_sample
                grid = torchvision.utils.make_grid(x, nrow=4)
                im = grid.permute(1, 2, 0).cpu().clip(-1, 1)*0.5 + 0.5
                im = Image.fromarray(np.array(im*255).astype(np.uint8))
                wandb.log({'Sample generations': wandb.Image(im)})

            if (step+1)%save_model_every == 0:
                image_pipe.save_pretrained(model_save_name+f'step_{step+1}')

        scheduler.step()

    image_pipe.save_pretrained(model_save_name)
    wandb.finish()
