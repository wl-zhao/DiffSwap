import argparse
import os
import cv2
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from einops import rearrange, repeat
import torch.nn.functional as F
import pdb
import torchvision
from PIL import Image
import pdb
import numpy as np
import torch.distributed as dist
import builtins
import datetime
from pathlib import Path
from ldm.data.portrait import Portrait
import json
from torchvision import transforms
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler


import warnings
warnings.filterwarnings("ignore")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
# above is all about distributed training and shouldn't be modified

    
@torch.no_grad()
def perform_swap(self, batch, ckpt, ddim_sampler = None, ddim_steps=200, ddim_eta=0., **kwargs):
    # now we swap the faces
    use_ddim = ddim_steps is not None

    log = dict()
        
    z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                    return_first_stage_outputs=True,
                                    force_c_encode=True,
                                    return_original_cond=True, swap=True)
    N = x.size(0)

    b, h, w = z.shape[0], z.shape[2], z.shape[3] # 64 x 64
    
    for mask_key in ['mask']:
        mask = (1 - batch[mask_key].float())[:, None]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        
            
        with self.ema_scope("Plotting Inpaint"):
            if ddim_sampler is None:
                ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, _ = ddim_sampler.sample(ddim_steps,N,shape,c, eta=ddim_eta, x0=z[:N], mask=mask, verbose=False,**kwargs)
        
        x_samples = self.decode_first_stage(samples.to(self.device))

        gen_imgs = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0).cpu()
        gen_imgs = np.array(gen_imgs * 255).astype(np.uint8)
        gen_imgs = rearrange(gen_imgs, 'b c h w -> b h w c')
        
        # save swapped images
        for j in range(N):
            src = batch['src'][j][:-4]
            save_root = f'swap_res/{ckpt}_{ddim_sampler.tgt_scale}'
            os.makedirs(os.path.join(save_dir, save_root, src), exist_ok=True)
            Image.fromarray(gen_imgs[j]).save(os.path.join(save_dir, save_root, src, batch['target'][j]))

    
if __name__ == '__main__':
    batch_size = 16
    num_workers = 8
    root_dir = 'data/portrait'
    save_dir = root_dir
    
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='checkpoint to load')
    parser.add_argument('--save_img', type=bool, default=False)
    parser.add_argument('--tgt_scale', type=float, default=0.01)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()
    init_distributed_mode(args)
    
    setup_for_distributed(is_main_process())
    device = get_rank()

    # get config
    config_path = 'configs/diffswap/default-project.yaml'
    config = OmegaConf.load(config_path)

    print('ready to build model', config.model)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.checkpoint)
    ckpt = os.path.basename(args.checkpoint).rsplit('.', 1)[0]

    model.eval()
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    
    print('model built')    
    model.module.cond_stage_model.affine_crop = True
    model.module.cond_stage_model.swap = True

    num_tasks = dist.get_world_size()

    dataset = Portrait(root_dir)
    sampler_val = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=device, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    ddim_sampler = DDIMSampler(model.module, tgt_scale=args.tgt_scale)
    
    print('start batch')
    for batch_idx, batch in enumerate(tqdm(dataloader)):        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        perform_swap(model.module, batch, ckpt, ddim_sampler)




