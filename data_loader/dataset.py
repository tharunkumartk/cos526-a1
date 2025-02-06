import numpy as np

from data_loader.load_llff import load_llff_data
from data_loader.load_deepvoxels import load_dv_data
from data_loader.load_blender import load_blender_data
from data_loader.load_LINEMOD import load_LINEMOD_data


class Data():
    def __init__(self, args):
        # Load data
        K = None
        if args.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                      recenter=True, bd_factor=.75,
                                                                      spherify=args.spherify)
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
            if not isinstance(i_test, list):
                i_test = [i_test]

            if args.llffhold > 0:
                print('Auto LLFF holdout,', args.llffhold)
                i_test = np.arange(images.shape[0])[::args.llffhold]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])

            i_split = [i_train, i_val, i_test]

            print('DEFINING BOUNDS')
            if args.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.

            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)

        elif args.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)

            near = 2.
            far = 6.

            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        elif args.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                        args.testskip)
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')

            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        elif args.dataset_type == 'deepvoxels':

            images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                     basedir=args.datadir,
                                                                     testskip=args.testskip)

            print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)

            hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
            near = hemi_R - 1.
            far = hemi_R + 1.

        else:
            print('Unknown dataset type', args.dataset_type, 'exiting')
            return

        # TODO: Remove after debugging
        i_train, i_val, i_test = i_split
        i_train = i_train[:3]

        self.i_split = i_split
        self.images = images
        self.poses = poses
        self.render_poses = render_poses

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            self.K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])
        else:
            self.K = K

        if args.render_only:
            self.render_poses = np.array(poses[i_test])

        self.hwf = hwf
        self.near = near
        self.far = far
