import os
import time
import imageio
import numpy as np
from tqdm import tqdm, trange

from model_helpers import *
from nerf import NeRF
from renderer import Renderer
from encoder import PositionalEncoder


class Model:
    def __init__(self, args, device=None):
        self.start = 0

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        nerf, nerf_fine, optimizer, embedder, embedder_dirs = self.create_nerf(args)

        self.nerf = nerf
        self.nerf_fine = nerf_fine
        self.optimizer = optimizer

        self.global_step = self.start

        self.renderer = Renderer(args)

        # Optimization settings
        self.use_batching = not args.no_batching

        self.lrate_decay = args.lrate_decay
        self.lrate = args.lrate

        self.N_iters = 200000 + 1
        self.N_ray_samples = args.N_rand

        self.chunk = args.chunk

        if not self.use_batching:
            self.precrop_iters = args.precrop_iters
            self.precrop_frac = args.precrop_frac

        # Logging Parameters:
        self.i_weights = args.i_weights
        self.i_print = args.i_print
        self.i_testset = args.i_testset
        self.i_video = args.i_video

    def create_nerf(self, args):
        """Instantiate NeRF's MLP model.
        """
        embedder = PositionalEncoder(args.multires, args.i_embed)

        embedder_dirs = None
        if args.use_viewdirs:
            embedder_dirs = PositionalEncoder(args.multires_views, args.i_embed)

        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]
        nerf_model = NeRF(D=args.netdepth, W=args.netwidth,
                          output_ch=output_ch, skips=skips, embedder=embedder, embedder_dirs=embedder_dirs,
                          use_viewdirs=args.use_viewdirs).to(self.device)
        grad_vars = list(nerf_model.parameters())

        nerf_model_fine = None
        if args.N_importance > 0:
            nerf_model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                                   output_ch=output_ch, skips=skips, embedder=embedder, embedder_dirs=embedder_dirs,
                                   use_viewdirs=args.use_viewdirs).to(self.device)
            grad_vars += list(nerf_model_fine.parameters())

        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        # Load checkpoints
        if args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname)))
                     if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # Reset start point
            self.start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            nerf_model.load_state_dict(ckpt['network_fn_state_dict'])
            if nerf_model_fine is not None:
                nerf_model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        return nerf_model, nerf_model_fine, optimizer, embedder, embedder_dirs

    def sample_rays(self, poses, images, H, W, K, idx=None):
        if idx is None:
            idx = np.linspace(0, len(poses)-1, len(poses), dtype=int)

        # For random ray batching
        # print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        # print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in idx], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        # print('shuffle rays')
        np.random.shuffle(rays_rgb)

        # print('done')
        return rays_rgb

    def ray_batch_from_one_image(self, step_i, pose, target, H, W, K):
        rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)

        if step_i < self.precrop_iters:
            dH = int(H // 2 * self.precrop_frac)
            dW = int(W // 2 * self.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                ), -1)
            if step_i == self.start:
                print(
                    f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {self.precrop_iters}")
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                 -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[self.N_ray_samples], replace=False)  # (N_ray_samples,)
        select_coords = coords[select_inds].long()  # (N_ray_samples, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_ray_samples, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_ray_samples, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_ray_samples, 3)

        return batch_rays, target_s

    def train_on_dataset(self, dataset, basedir, expname, device=None):
        if device is None:
            device = self.device

        i_batch = 0
        i_train, i_val, i_test = dataset.i_split
        H, W, focal = dataset.hwf
        K = dataset.K
        # self.start += 1

        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        rays_rgb = None

        for i in trange(self.start, self.N_iters):
            # Sample random ray batch
            if self.use_batching:
                # Random over all images
                if rays_rgb is None:
                    rays_rgb = self.sample_rays(dataset.poses, dataset.images, H, W, K, idx=i_train)

                    # Move training data to GPU
                    rays_rgb = torch.tensor(rays_rgb, device=device)

                batch = rays_rgb[i_batch:i_batch + self.N_ray_samples]  # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += self.N_ray_samples
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0

            else:
                # Random from one image
                img_i = np.random.choice(i_train)
                target = dataset.images[img_i]
                pose = dataset.poses[img_i, :3, :4]

                target = torch.tensor(target, device=device)
                pose = torch.tensor(pose, device=device)

                batch_rays, target_s = self.ray_batch_from_one_image(i, pose, target, H, W, K)

            # Core optimization loop
            rgb, disp, acc, extras = self.renderer.forward(H, W, K, nerf=self.nerf,
                                                           nerf_fine=self.nerf_fine, chunk=self.chunk, rays=batch_rays,
                                                           near=dataset.near, far=dataset.far, retraw=True,)

            self.optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            self.optimizer.step()

            # update learning rate
            decay_rate = 0.1
            decay_steps = self.lrate_decay * 1000
            new_lrate = self.lrate * (decay_rate ** (self.global_step / decay_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate

            # Logging
            if i % self.i_weights == 0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': self.global_step,
                    'network_fn_state_dict': self.nerf.state_dict(),
                    'network_fine_state_dict': self.nerf_fine.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            if i % self.i_video == 0 and i > 0:
                self.render_video(dataset, basedir, expname, step_i=i)

            if i % self.i_testset == 0 and i > 0:
                self.render_test(dataset, basedir, expname, step_i=i)

            if i % self.i_print == 0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            self.global_step += 1

    @torch.no_grad()
    def render_video(self, dataset, basedir, expname, step_i):
        render_poses = torch.tensor(dataset.render_poses, device=self.device)
        self.renderer.eval()
        rgbs, disps = self.renderer.render_path(render_poses, dataset.hwf, dataset.K, self.chunk,
                                                self.nerf, self.nerf_fine,
                                                near=dataset.near, far=dataset.far,)
        self.renderer.train()
        print('Done, saving', rgbs.shape, disps.shape)
        moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, step_i))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

    @torch.no_grad()
    def render_test(self, dataset, basedir, expname, step_i):
        _, _, i_test = dataset.i_split

        test_images = torch.tensor(dataset.images[i_test], device=self.device)

        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(step_i))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', dataset.poses[i_test].shape)

        render_poses = torch.tensor(dataset.poses[i_test], device=self.device)
        self.renderer.eval()
        self.renderer.render_path(render_poses, dataset.hwf, dataset.K, self.chunk,
                                  self.nerf, self.nerf_fine,
                                  near=dataset.near, far=dataset.far, gt_imgs=test_images, savedir=testsavedir)
        self.renderer.train()

        print('Saved test set')

    @torch.no_grad()
    def render_only(self, dataset, basedir, expname, render_test=False, render_factor=1.):
        print('RENDER ONLY')
        if render_test:
            _, _, i_test = dataset.i_split
            # render_test switches to test poses
            images = dataset.images[i_test]
            render_poses = np.array(dataset.poses[i_test])
        else:
            # Default is smoother render_poses path
            images = None
            render_poses = torch.tensor(dataset.render_poses, device=self.device)

        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}_{:06d}'.format('test' if render_test else 'path', self.start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        self.renderer.eval()
        rgbs, disps = self.renderer.render_path(render_poses, dataset.hwf, dataset.K, self.chunk, self.nerf,
                                            self.nerf_fine, near=dataset.near, far=dataset.far,
                                            gt_imgs=images, savedir=testsavedir, render_factor=render_factor)
        self.renderer.train()
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
        
        imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.max(disps)), fps=30, quality=8)