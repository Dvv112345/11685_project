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

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint, save_best_checkpoint, load_checkpoint, visualize_batch, save_last_checkpoint

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="config file used to specify parameters")

    # data 
    parser.add_argument("--data_dir", type=str, default='./data/imagenet100_128x128/train', help="data folder") 
    parser.add_argument("--is_cifar_10", type = str2bool, default = 1, help = 'use cifar 10') #jinyi
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes in dataset")


    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')
    parser.add_argument("--use_val", type = str2bool, default = 1, help = "use val") #jinyi
    
    # ddpm
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="ddpm variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="clip sample range")
    parser.add_argument("--optimizer_type", type=str, default='AdamW', help="Optimizer")
    parser.add_argument("--scheduler_type", type=str, default='CosineAnnealingLR', help="LR scheduler")
    
    # unet
    parser.add_argument("--unet_in_size", type=int, default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")
    
    # vae
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm")
    parser.add_argument("--pretrained_vae", type = str, default = "pretrained/model.ckpt", help = "pretrained vae path")
    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference")
    
    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference")
    
    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path for inference")
    parser.add_argument("--load_model_pth", type = str, default= None, help = "load previous model")
    parser.add_argument("--resume", type = str2bool, default=False, help = "train from previous checkpoint")
    parser.add_argument("--previous_run_id", type = str, default = None, help = "wandb previous run id")
    # distributed setting, jinyi
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    parser.add_argument("--rank", type=int, default=0, help="process rank (for distributed)")
    parser.add_argument("--world_size", type=int, default=1, help="world size (for distributed)")
    
    # first parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args
    
    
def main():
    
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup distributed initialize and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    
    # setup dataset
    logger.info("Creating dataset")
    # TODO: use transform to normalize your images to [-1, 1]
    # TODO: you can also use horizontal flip
    P_TRANSFORM = [0.1,0.1,0.1] # Crop, Rotation, Color
    transform = transforms.Compose([
        # 1. Random Flip 
        transforms.RandomHorizontalFlip(),
        
        # 2. Random Crop/Resize (Use RandomResizedCrop for a strong spatial augmentation)
        transforms.RandomApply(
            [transforms.RandomResizedCrop(
                size=(args.image_size, args.image_size), 
                scale=(0.8, 1.0) # Scale factor to zoom in/out
            )], 
            p=P_TRANSFORM[0]
        ),
        
        # 3. Random Rotation
        transforms.RandomApply(
            [transforms.RandomRotation(
                degrees=10, 
                expand=False
            )], 
            p=P_TRANSFORM[1]
        ),
        
        # 4. Color/Brightness/Contrast Jitter
        transforms.RandomApply(
            [transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.05
            )],
            p=P_TRANSFORM[2]
        ),

        # 5. Final Transformations (Always applied)
        # Ensure the image is resized to the target size after all random spatial transforms
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # TOOD: use image folder for your train dataset
    if args.is_cifar_10:
        train_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=False,  # set to True if CIFAR-10 is not yet downloaded
            transform=transform
        )
    else:   
        train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    
    # TODO: setup dataloader
    sampler = None 
    if args.distributed:
        # TODO: distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    # TODO: shuffle
    shuffle = False if sampler else True
    # TODO dataloader

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Example call:
    visualize_batch(train_loader, num_images=8, norm_range='-1_to_1')

        # --- Validation dataset / loader (use CIFAR10 test split as val by default) ---
    val_loader = None
    if args.use_val:
        if args.is_cifar_10:
            val_dataset = datasets.CIFAR10(
                root=args.data_dir,
                train=False,   # test split -> use as validation
                download=False,
                transform=transform  # evaluation transform; you may want a simpler transform
            )
        else:
            # If have a separate val folder, replace this with ImageFolder
            val_dataset = datasets.ImageFolder(root=args.data_dir.replace("train", "val"), transform=transform)

        val_sampler = None
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False if val_sampler else False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        val_loader = None

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size 
    args.total_batch_size = total_batch_size
    
    # setup experiment folder
    os.makedirs(args.output_dir, exist_ok=True)
    if args.run_name is None:
        args.run_name = f"exp-{len(os.listdir(args.output_dir))}"
    else:
        args.run_name = f"exp-{len(os.listdir(args.output_dir))}-{args.run_name}"
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, "checkpoints")
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
   
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    # scheduler = DDPMScheduler(args.num_train_timesteps)
    
    # NOTE: this is for latent DDPM 
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        # NOTE: do not change this
        vae.init_from_ckpt(args.pretrained_vae)
        vae.eval()

    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: 
        class_embedder = ClassEmbedder(args.num_classes, args.unet_ch)
        
    # send to device
    unet = unet.to(device)
    # scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
    
    # TODO: setup optimizer
    # Setup Scheduler based on args.optimizer_type
    from torch.optim import Adam, AdamW, SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
    OPTIMIZER_MAP = {
        'AdamW': AdamW,
        'Adam': Adam,
        'SGD': SGD,
    }
    optimizer_class = OPTIMIZER_MAP.get(args.optimizer_type, AdamW) # Default to AdamW if not found
    # Common arguments for all optimizers
    optimizer_kwargs = {
        'lr': args.learning_rate,
        # Only include weight_decay for applicable optimizers
        'weight_decay': args.weight_decay if args.optimizer_type in ['AdamW', 'Adam', 'SGD'] else 0.0
    }
    # Add SGD-specific args if needed
    if optimizer_class is SGD:
        optimizer_kwargs['momentum'] = 0.9
    optimizer = optimizer_class(unet.parameters(), **optimizer_kwargs)
    # TODO: setup scheduler
    # Setup Scheduler based on args.scheduler_type
    SCHEDULER_MAP = {
        'CosineAnnealingLR': CosineAnnealingLR,
        'Step': StepLR,
        'Plateau': ReduceLROnPlateau,
        'None': None,
    }
    scheduler_class = SCHEDULER_MAP.get(args.scheduler_type, CosineAnnealingLR) # Default to CosineAnnealing
    if scheduler_class is None:
        scheduler = None
    elif scheduler_class is CosineAnnealingLR:
        # Requires T_max (total number of steps/epochs for the annealing cycle)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=max(1, args.num_epochs) 
        )
        print("Using Cosine Annealing LR.")
    elif scheduler_class is StepLR:
        # Requires step_size and gamma
        scheduler = StepLR(
            optimizer, 
            step_size=max(1, args.num_epochs // 3), 
            gamma=0.1
        )
    elif scheduler_class is ReduceLROnPlateau:
        # Requires monitoring a metric (needs to be stepped in the training loop)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', # Monitor loss (min) or metric (max)
            factor=0.5, 
            patience=5
        )
    # scheduler = scheduler.to(device)
    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    #  setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae
    # TODO: setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(args.num_train_timesteps, args.num_inference_steps, args.beta_start, args.beta_end, args.beta_schedule, args.variance_type, args.prediction_type, args.clip_sample, args.clip_sample_range)
    else:
        scheduler_wo_ddp = DDPMScheduler(args.num_train_timesteps, args.num_inference_steps, args.beta_start, args.beta_end, args.beta_schedule, args.variance_type, args.prediction_type, args.clip_sample, args.clip_sample_range)
    scheduler_wo_ddp.to(device)
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(unet, scheduler = scheduler, vae=vae_wo_ddp, class_embedder=class_embedder_wo_ddp, optimizer=optimizer, epoch = start_epoch, checkpoint_path=args.load_model_pth)
        print("Model Load Successfully, model:", args.load_model_pth)

    # TODO: setup evaluation pipeline
    # NOTE: this pipeline is not differentiable and only for evaluatin
    pipeline = DDPMPipeline(unet=unet_wo_ddp, 
                            scheduler=scheduler_wo_ddp, 
                            vae=vae_wo_ddp, 
                            class_embedder=class_embedder_wo_ddp)
    
    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
    
    # start tracker
    if is_primary(args):
        if args.resume:
            wandb_logger = wandb.init(
                project='ddpm',
                name=args.run_name,
                id=args.previous_run_id,
                resume='must'
            )
        else:
            wandb_logger = wandb.init(
                project='ddpm', 
                name=args.run_name, 
                config=vars(args))
    
    # Start training    
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=start_epoch * num_update_steps_per_epoch,
        disable=not is_primary(args)
    )
    best_val_m = 1e6
    # training
    for epoch in range(start_epoch, args.num_epochs):
        
        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        
        loss_m = AverageMeter()
        
        # TODO: set unet and scheduelr to train
        unet.train()

        
        # TODO: finish this
        for step, (images, labels) in enumerate(train_loader):
            
            batch_size = images.size(0)
            # TODO: send to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            
            # NOTE: this is for latent DDPM 
            if vae is not None:
                # use vae to encode images as latents
                enc = None
                enc = vae.encode(images)
                if isinstance(enc, torch.Tensor):
                    images = enc
                elif isinstance(enc, (list, tuple)) and len(enc) > 0 and isinstance(enc[0], torch.Tensor):
                    images = enc[0]
                elif hasattr(enc, 'latent') and isinstance(enc.latent, torch.Tensor):
                    images = enc.latent
                else:
                    print("Use Images.")
                    images = images
                # NOTE: do not change  this line, this is to ensure the latent has unit std
                images = images * 0.1845
            
            # TODO: zero grad optimizer
            optimizer.zero_grad()
            
            # NOTE: this is for CFG
            if class_embedder is not None:
                # TODO: use class embedder to get class embeddings
                class_emb = class_embedder(labels)
            else:
                # NOTE: if not cfg, set class_emb to None
                class_emb = None
            
            # TODO: sample noise 
            noise = torch.randn_like(images, device=device)   
            
            # TODO: sample timestep t
            timesteps = torch.randint(0, args.num_train_timesteps, (batch_size,), device=device).long()
            
            # TODO: add noise to images using scheduler
            noisy_images = scheduler_wo_ddp.add_noise(images, noise, timesteps)
            
            # TODO: model prediction
            model_pred = unet(noisy_images, timesteps, class_emb)
            
            if args.prediction_type == 'epsilon':
                target = noise 
            
            # TODO: calculate loss
            loss = F.mse_loss(model_pred, target)
            
            # record loss
            loss_m.update(loss.item())
            
            # backward and step 
            loss.backward()
            # TODO: grad clip
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            
            # TODO: step your optimizer
            optimizer.step()
            progress_bar.update(1)

        if not args.scheduler_type == "Plateau":
            scheduler.step()
        else:
            scheduler.step(loss)

        
            
        # logger: only update when finish one epoch
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg})")
        metrics = {'loss': loss_m.avg}
    
        # validation
        # send unet to evaluation mode
        unet.eval()        
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)

        # add validation, Jinyi
        # -------------------------
        # Validation (compute val loss)
        # -------------------------
        max_val_batches = 100
        val_m = AverageMeter()
        if val_loader is not None:
            unet.eval()
            with torch.no_grad():
                # max_val_batches = args.max_val_batches
                for vstep, (vimages, vlabels) in enumerate(val_loader):
                    if max_val_batches > 0 and vstep >= max_val_batches:
                        break

                    vbatch = vimages.size(0)
                    vimages = vimages.to(device, non_blocking=True)
                    vlabels = vlabels.to(device, non_blocking=True)

                    # encode with VAE if using latent ddpm
                    if vae is not None:
                        enc = vae.encode(vimages)
                        if isinstance(enc, torch.Tensor):
                            vimages_latent = enc
                        elif isinstance(enc, (list, tuple)) and len(enc) > 0 and isinstance(enc[0], torch.Tensor):
                            vimages_latent = enc[0]
                        elif hasattr(enc, 'latent') and isinstance(enc.latent, torch.Tensor):
                            vimages_latent = enc.latent
                        else:
                            vimages_latent = vimages
                        vimages = vimages_latent * 0.1845

                    # sample noise and timesteps same as training
                    vnoise = torch.randn_like(vimages, device=device)
                    vtimesteps = torch.randint(0, args.num_train_timesteps, (vbatch,), device=device).long()

                    vnoisy = scheduler_wo_ddp.add_noise(vimages, vnoise, vtimesteps)
                    vpred = unet(vnoisy, vtimesteps, None if class_embedder is None else class_embedder(vlabels))

                    if args.prediction_type == 'epsilon':
                        vtarget = vnoise
                    else:
                        vtarget = vnoise  # keep fallback

                    vloss = F.mse_loss(vpred, vtarget)
                    val_m.update(vloss.item(), n=vbatch)

            # log validation loss
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} Validation Loss: {val_m.avg:.6f}")
            if is_primary(args):
                metrics.update({'val_loss': val_m.avg})
                metrics.update({'lr': optimizer.param_groups[0]['lr']})
            # step ReduceLROnPlateau with val loss
            # if args.scheduler_type == "Plateau" and scheduler is not None:
            #     scheduler.step(val_m.avg)
        else:
            logger.info("No validation loader configured (val_loader is None)")

        
        # NOTE: this is for CFG
        if args.use_cfg:
            # random sample 4 classes
            classes = torch.randint(0, args.num_classes, (4,), device=device)
            # TODO: fill pipeline
            gen_images = pipeline(batch_size=4, 
                                    classes=classes, 
                                    generator=generator, 
                                    num_inference_steps=args.num_inference_steps, 
                                    guidance_scale=args.cfg_guidance_scale)
        else:
            # TODO: fill pipeline
            gen_images = pipeline(batch_size=4, 
                                    generator=generator, 
                                    num_inference_steps=args.num_inference_steps)
        if epoch % 20 == 0:
            # create a blank canvas for the grid
            grid_image = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))
            # paste images into the grid
            for i, image in enumerate(gen_images):
                x = (i % 4) * args.image_size
                y = 0
                grid_image.paste(image, (x, y))
        
            # Send to wandb
            if is_primary(args):
                metrics.update({'gen_images': wandb.Image(grid_image)})
        wandb_logger.log(metrics)
                
        # save checkpoint
        if val_m.avg < best_val_m:
            best_val_m = val_m.avg
            if is_primary(args):
                save_best_checkpoint(unet_wo_ddp, scheduler_wo_ddp, vae_wo_ddp, class_embedder, optimizer, epoch, save_dir=save_dir)
        
        save_last_checkpoint(unet_wo_ddp, scheduler_wo_ddp, vae_wo_ddp, class_embedder, optimizer, epoch, save_dir=save_dir)

if __name__ == '__main__':
    main()
