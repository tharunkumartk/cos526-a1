import os
import time
import imageio
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_helpers import *

DEBUG = False


class Renderer(nn.Module):
    def __init__(self, args):
        super(Renderer, self).__init__()

        # If True, assume a white background.
        self.white_bkgd = args.white_bkgd

        # Number of different times to sample along each ray.
        self.N_samples = args.N_samples
        # Number of additional times to sample along each ray.
        # These samples are only passed to network_fine.
        if args.N_importance is None:
            self.N_importance = 0
        else:
            self.N_importance = args.N_importance

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != "llff" or args.no_ndc:
            print("Not ndc!")
            # If True, represent ray origin, direction in NDC coordinates.
            self.ndc = False
            # If True, sample linearly in inverse depth rather than in depth.
            self.lindisp = args.lindisp
        else:
            self.ndc = True
            self.lindisp = False

        # If True, use viewing direction of a point in space to condition the model.
        self.use_viewdirs = args.use_viewdirs

        self.raw_noise_std = args.raw_noise_std

        # 0 or 1. If non-zero, each ray is sampled at stratified random points in time. Always 0 during test time.
        self.perturb = args.perturb

        self.netchunk = args.netchunk

    def ray_marcher(self, raw, z_vals, rays_d, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        # raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
        def raw2alpha(raw, dists, act_fn=F.relu):
            # ------------------------------------------------------
            # ----- PLEASE FILL IN COMPUTATIONS FOR
            # ----- alpha: [N_rays, N_samples]. Actual alpha values computed from volume density and sampled distance.
            # ------------------------------------------------------

            alpha = 1.0 - torch.exp(-act_fn(raw) * dists)
            return alpha

        device = raw.device

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)],
            -1,
        )  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.0
        # Only add random noise during training
        if self.training and self.raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * self.raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * self.raw_noise_std
                noise = torch.tensor(noise, device=device)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # ------------------------------------------------------
        # ----- PLEASE FILL IN COMPUTATIONS FOR
        # ----- weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        # ----- rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        # ----- depth_map: [num_rays]. Estimated distance to object.
        # ----- disp_map: [num_rays]. Disparity map. Inverse of depth map.
        # ----- acc_map: [num_rays]. Sum of weights along each ray.
        # ------------------------------------------------------

        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [
                        torch.ones((alpha.shape[0], 1), device=device),
                        1.0 - alpha + 1e-10,
                    ],
                    -1,
                ),
                -1,
            )[:, :-1]
        )

        # Calculate weighted color and depth
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1.0 / torch.max(
            torch.full_like(depth_map, 1e-10), depth_map / torch.sum(weights, -1)
        )
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_rays(
        self,
        ray_batch,
        nerf,
        nerf_fine=None,
        retraw=False,
        pytest=False,
    ):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          nerf: function. Model for predicting RGB and density at each point
            in space.
          nerf_fine: "fine" network with same spec as nerf.
          retraw: bool. If True, include model's raw, unprocessed predictions.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        device = ray_batch.device

        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples, device=device)
        if not self.lindisp:
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.N_samples])

        if self.training and self.perturb > 0.0:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand, device=device)

            z_vals = lower + (upper - lower) * t_rand

            perturb = self.perturb
        else:
            perturb = False

        # ------------------------------------------------------
        # ----- PLEASE FILL IN COMPUTATIONS FOR
        # ----- pts: [N_rays, N_samples, 3] 3D point position
        # ------------------------------------------------------

        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]

        # Pass sampled points through an implicit scene representation model
        raw = nerf(pts, viewdirs, self.netchunk)
        rgb_map, disp_map, acc_map, weights, depth_map = self.ray_marcher(
            raw, z_vals, rays_d, pytest=pytest
        )

        if self.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                weights[..., 1:-1],
                self.N_importance,
                det=(perturb == 0.0),
                pytest=pytest,
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = (
                rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            )  # [N_rays, N_samples + N_importance, 3]

            fine_model = nerf if nerf_fine is None else nerf_fine
            # Pass importance sampled points through an implicit scene representation model
            raw = fine_model(pts, viewdirs, self.netchunk)

            rgb_map, disp_map, acc_map, weights, depth_map = self.ray_marcher(
                raw, z_vals, rays_d, pytest=pytest
            )

        ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
        if retraw:
            ret["raw"] = raw
        if self.N_importance > 0:
            ret["rgb0"] = rgb_map_0
            ret["disp0"] = disp_map_0
            ret["acc0"] = acc_map_0
            ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def batchify_rays(
        self, rays_flat, nerf, nerf_fine=None, chunk=1024 * 32, retraw=False
    ):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(
                rays_flat[i : i + chunk], nerf, nerf_fine=nerf_fine, retraw=retraw
            )
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def forward(
        self,
        H,
        W,
        K,
        nerf,
        nerf_fine=None,
        chunk=1024 * 32,
        rays=None,
        c2w=None,
        near=0.0,
        far=1.0,
        c2w_staticcam=None,
        retraw=False,
    ):
        """Render rays
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          K: float. Intrnsic matrix of the camera.
          nerf: function. Model for predicting RGB and density at each point
            in space.
          nerf_fine: "fine" network with same spec as nerf.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
          retraw: bool. If True, include model's raw, unprocessed predictions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """
        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = get_rays(H, W, K, c2w)
        else:
            # use provided ray batch
            rays_o, rays_d = rays

        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if self.ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
            rays_d[..., :1]
        )
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        all_ret = self.batchify_rays(rays, nerf, nerf_fine, chunk, retraw=retraw)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ["rgb_map", "disp_map", "acc_map"]
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path(
        self,
        render_poses,
        hwf,
        K,
        chunk,
        nerf,
        nerf_fine,
        near,
        far,
        gt_imgs=None,
        savedir=None,
        render_factor=0,
    ):

        H, W, focal = hwf

        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor
            focal = focal / render_factor

            K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

        rgbs = []
        disps = []

        for i, c2w in enumerate(tqdm(render_poses)):
            rgb, disp, acc, _ = self.forward(
                H,
                W,
                K,
                nerf,
                nerf_fine=nerf_fine,
                chunk=chunk,
                c2w=c2w[:3, :4],
                near=near,
                far=far,
            )
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())
            if i == 0:
                print(rgb.shape, disp.shape)

            """
            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                print(p)
            """

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, "{:03d}.png".format(i))
                imageio.imwrite(filename, rgb8)

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

        return rgbs, disps
