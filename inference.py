import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(None)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(None)

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline =  DDPMPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
        device=device
    )

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    batch_size = 50
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(num_inference_steps=args.num_inference_steps, 
                                  batch_size=batch_size, 
                                  class_labels=classes, 
                                  generator=generator)
            all_images.append(gen_images)
    else:
        # generate 5000 images
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline(num_inference_steps=args.num_inference_steps, 
                                  batch_size=batch_size, 
                                  generator=generator)
            all_images.append(gen_images)
    all_images = torch.cat(all_images, dim=0)
    all_images = (all_images * 255).clamp(0, 255).to(torch.uint8)
    # TODO: load validation images as reference batch
    val_dir = args.val_dir
    val_images = []
    for fname in os.listdir(val_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(val_dir, fname)).convert('RGB')
            img = img.resize((args.unet_in_size, args.unet_in_size))
            img = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0) / 255.0
            val_images.append(img)
    val_images = torch.cat(val_images, dim=0).to(device)
    
    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    from torchmetrics.image.fid import FrechetInceptionDistance, InceptionScore
    
    # TODO: compute FID and IS
    # FID: real then fake
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True).to(device)
    fid.update(val_images, real=True)
    fid.update(all_images.float() / 255.0, real=False)
    fid_score = fid.compute().item()

    # IS: only on generated images
    inception.update(all_images.float() / 255.0)
    is_mean, is_std = inception.compute()
    
    # output to log
    logger.info(f"FID score: {fid_score:.4f}")
    logger.info(f"Inception Score: mean={is_mean:.4f}, std={is_std:.4f}")


if __name__ == '__main__':
    main()
