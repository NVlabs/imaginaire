# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import numpy as np
import torch


class EvalCameraController:
    def __init__(self, voxel, maxstep=128, pattern=0, cam_ang=73, smooth_decay_multiplier=1.0):
        self.voxel = voxel
        self.maxstep = maxstep
        self.camera_poses = []  # ori, dir, up, f
        circle = torch.linspace(0, 2*np.pi, steps=maxstep)
        size = min(voxel.voxel_t.size(1), voxel.voxel_t.size(2)) / 2
        # Shrink the circle a bit.
        shift = size * 0.2
        size = size * 0.8

        if pattern == 0:
            height_history = []
            # Calculate smooth height.
            for i in range(maxstep):
                farpoint = torch.tensor([
                    70,
                    torch.sin(circle[i])*size + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size + voxel.voxel_t.size(2)/2 + shift])
                height_history.append(self._get_height(farpoint[1], farpoint[2], farpoint[0]))

            # Filtfilt
            height_history = self.filtfilt(height_history, decay=0.2*smooth_decay_multiplier)

            for i in range(maxstep):
                farpoint = torch.tensor([
                    70,
                    torch.sin(circle[i])*size + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size + voxel.voxel_t.size(2)/2 + shift])

                farpoint[0] = height_history[i]

                nearpoint = torch.tensor([
                    60,
                    torch.sin(circle[i]+0.5*np.pi)*size*0.5 + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]+0.5*np.pi)*size*0.5 + voxel.voxel_t.size(2)/2 + shift])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(cam_ang/2))  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        elif pattern == 1:
            zoom = torch.linspace(1.0, 0.25, steps=maxstep)
            height_history = []
            for i in range(maxstep):
                farpoint = torch.tensor([
                    90,
                    torch.sin(circle[i])*size + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size + voxel.voxel_t.size(2)/2 + shift])

                height_history.append(self._get_height(farpoint[1], farpoint[2], farpoint[0]))

            height_history = self.filtfilt(height_history, decay=0.2*smooth_decay_multiplier)

            for i in range(maxstep):
                farpoint = torch.tensor([
                    90,
                    torch.sin(circle[i])*size + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size + voxel.voxel_t.size(2)/2 + shift])

                farpoint[0] = height_history[i]

                nearpoint = torch.tensor([
                    60,
                    torch.sin(circle[i]-0.3*np.pi)*size*0.3 + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]-0.3*np.pi)*size*0.3 + voxel.voxel_t.size(2)/2 + shift])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(cam_ang/2)*zoom[i])  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        elif pattern == 2:
            move = torch.linspace(1.0, 0.2, steps=maxstep)
            height_history = []
            for i in range(maxstep):
                farpoint = torch.tensor([
                    90,
                    torch.sin(circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                height_history.append(self._get_height(farpoint[1], farpoint[2], farpoint[0]))

            height_history = self.filtfilt(height_history, decay=0.2*smooth_decay_multiplier)

            for i in range(maxstep):
                farpoint = torch.tensor([
                    90,
                    torch.sin(circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                farpoint[0] = height_history[i]

                nearpoint = torch.tensor([
                    60,
                    torch.sin(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(2)/2 + shift])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(cam_ang/2))  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        elif pattern == 3:
            move = torch.linspace(0.75, 0.2, steps=maxstep)
            height_history = []
            for i in range(maxstep):
                farpoint = torch.tensor([
                    70,
                    torch.sin(-circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(-circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                height_history.append(self._get_height(farpoint[1], farpoint[2], farpoint[0]))

            height_history = self.filtfilt(height_history, decay=0.2*smooth_decay_multiplier)

            for i in range(maxstep):
                farpoint = torch.tensor([
                    70,
                    torch.sin(-circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(-circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                farpoint[0] = height_history[i]

                nearpoint = torch.tensor([
                    60,
                    torch.sin(-circle[i]-0.4*np.pi)*size*0.9*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(-circle[i]-0.4*np.pi)*size*0.9*move[i] + voxel.voxel_t.size(2)/2 + shift])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(cam_ang/2))  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        elif pattern == 4:
            move = torch.linspace(1.0, 0.5, steps=maxstep)
            height_history = []
            for i in range(maxstep):
                farpoint = torch.tensor([
                    90,
                    torch.sin(circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                height_history.append(self._get_height(farpoint[1], farpoint[2], farpoint[0]))

            height_history = self.filtfilt(height_history, decay=0.2*smooth_decay_multiplier)

            for i in range(maxstep):
                farpoint = torch.tensor([
                    90,
                    torch.sin(circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                farpoint[0] = height_history[i]

                nearpoint = torch.tensor([
                    60,
                    torch.sin(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(2)/2 + shift])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(cam_ang/2))  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        # look outward
        elif pattern == 5:
            move = torch.linspace(1.0, 0.5, steps=maxstep)
            height_history = []
            for i in range(maxstep):
                nearpoint = torch.tensor([
                    60,
                    torch.sin(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(2)/2 + shift])

                height_history.append(self._get_height(nearpoint[1], nearpoint[2], nearpoint[0]))

            height_history = self.filtfilt(height_history, decay=0.2*smooth_decay_multiplier)

            for i in range(maxstep):
                nearpoint = torch.tensor([
                    60,
                    torch.sin(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(2)/2 + shift])

                nearpoint[0] = height_history[i]

                farpoint = torch.tensor([
                    60,
                    torch.sin(circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

                cam_ori = self.voxel.world2local(nearpoint)
                cam_dir = self.voxel.world2local(farpoint - nearpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(cam_ang/2))  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))
        # Rise
        elif pattern == 6:
            shift = 0
            lift = torch.linspace(0.0, 200.0, steps=maxstep)
            zoom = torch.linspace(0.8, 1.6, steps=maxstep)
            for i in range(maxstep):
                farpoint = torch.tensor([
                    80+lift[i],
                    torch.sin(circle[i]/4)*size*0.2 + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]/4)*size*0.2 + voxel.voxel_t.size(2)/2 + shift])

                farpoint[0] = self._get_height(farpoint[1], farpoint[2], farpoint[0])

                nearpoint = torch.tensor([
                    65,
                    torch.sin(circle[i]/4+0.5*np.pi)*size*0.1 + voxel.voxel_t.size(1)/2 + shift,
                    torch.cos(circle[i]/4+0.5*np.pi)*size*0.1 + voxel.voxel_t.size(2)/2 + shift])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(73/2)*zoom[i])  # about 24mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))
        # 45deg
        elif pattern == 7:
            rad = torch.tensor([np.deg2rad(45).astype(np.float32)])
            size = 1536
            for i in range(maxstep):
                farpoint = torch.tensor([
                    61+size,
                    torch.sin(rad)*size + voxel.voxel_t.size(1)/2,
                    torch.cos(rad)*size + voxel.voxel_t.size(2)/2])

                nearpoint = torch.tensor([
                    61,
                    voxel.voxel_t.size(1)/2,
                    voxel.voxel_t.size(2)/2])
                cam_ori = self.voxel.world2local(farpoint)
                cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
                cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
                cam_f = 0.5/np.tan(np.deg2rad(19.5/2))  # about 50mm fov

                self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

    def _get_height(self, loc0, loc1, minheight):
        loc0 = int(loc0)
        loc1 = int(loc1)
        height = minheight
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if (loc0+dx) < 0 or (loc0+dx) >= self.voxel.heightmap.shape[0] or (loc1+dy) < 0 or \
                        (loc1+dy) >= self.voxel.heightmap.shape[1]:
                    height = max(height, minheight)
                else:
                    height = max(height, self.voxel.heightmap[loc0+dx, loc1+dy] + 2)
        return height

    def filtfilt(self, height_history, decay=0.2):
        # Filtfilt
        height_history2 = []
        maxstep = len(height_history)
        prev_height = height_history[0]
        for i in range(maxstep):
            prev_height = prev_height - decay
            if prev_height < height_history[i]:
                prev_height = height_history[i]
            height_history2.append(prev_height)
        prev_height = height_history[-1]
        for i in range(maxstep-1, -1, -1):
            prev_height = prev_height - decay
            if prev_height < height_history[i]:
                prev_height = height_history[i]
            height_history2[i] = max(prev_height, height_history2[i])
        return height_history2

    def __len__(self):
        return len(self.camera_poses)

    def __getitem__(self, idx):
        return self.camera_poses[idx]


class TourCameraController:
    def __init__(self, voxel, maxstep=128):
        self.voxel = voxel
        self.maxstep = maxstep
        self.camera_poses = []  # ori, dir, up, f
        circle = torch.linspace(0, 2*np.pi, steps=maxstep//4)
        size = min(voxel.voxel_t.size(1), voxel.voxel_t.size(2)) / 2
        # Shrink the circle a bit
        shift = size * 0.2
        size = size * 0.8

        for i in range(maxstep//4):
            farpoint = torch.tensor([
                70,
                torch.sin(circle[i])*size + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i])*size + voxel.voxel_t.size(2)/2 + shift])

            farpoint[0] = self._get_height(farpoint[1], farpoint[2], farpoint[0])

            nearpoint = torch.tensor([
                60,
                torch.sin(circle[i]+0.5*np.pi)*size*0.5 + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i]+0.5*np.pi)*size*0.5 + voxel.voxel_t.size(2)/2 + shift])
            cam_ori = self.voxel.world2local(farpoint)
            cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
            cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
            cam_f = 0.5/np.tan(np.deg2rad(73/2))  # about 24mm fov

            self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        zoom = torch.linspace(1.0, 0.25, steps=maxstep//4)
        for i in range(maxstep//4):
            farpoint = torch.tensor([
                90,
                torch.sin(circle[i])*size + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i])*size + voxel.voxel_t.size(2)/2 + shift])

            farpoint[0] = self._get_height(farpoint[1], farpoint[2], farpoint[0])

            nearpoint = torch.tensor([
                60,
                torch.sin(circle[i]-0.3*np.pi)*size*0.3 + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i]-0.3*np.pi)*size*0.3 + voxel.voxel_t.size(2)/2 + shift])
            cam_ori = self.voxel.world2local(farpoint)
            cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
            cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
            cam_f = 0.5/np.tan(np.deg2rad(73/2)*zoom[i])  # about 24mm fov

            self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        move = torch.linspace(1.0, 0.2, steps=maxstep//4)
        for i in range(maxstep//4):
            farpoint = torch.tensor([
                90,
                torch.sin(circle[i])*size*move[i] + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i])*size*move[i] + voxel.voxel_t.size(2)/2 + shift])

            farpoint[0] = self._get_height(farpoint[1], farpoint[2], farpoint[0])

            nearpoint = torch.tensor([
                60,
                torch.sin(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i]+0.5*np.pi)*size*0.3*move[i] + voxel.voxel_t.size(2)/2 + shift])
            cam_ori = self.voxel.world2local(farpoint)
            cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
            cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
            cam_f = 0.5/np.tan(np.deg2rad(73/2))  # about 24mm fov

            self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

        lift = torch.linspace(0.0, 200.0, steps=maxstep//4)
        zoom = torch.linspace(0.6, 1.2, steps=maxstep//4)
        for i in range(maxstep//4):
            farpoint = torch.tensor([
                80+lift[i],
                torch.sin(circle[i])*size*0.2 + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i])*size*0.2 + voxel.voxel_t.size(2)/2 + shift])

            farpoint[0] = self._get_height(farpoint[1], farpoint[2], farpoint[0])

            nearpoint = torch.tensor([
                60,
                torch.sin(circle[i]+0.5*np.pi)*size*0.1 + voxel.voxel_t.size(1)/2 + shift,
                torch.cos(circle[i]+0.5*np.pi)*size*0.1 + voxel.voxel_t.size(2)/2 + shift])
            cam_ori = self.voxel.world2local(farpoint)
            cam_dir = self.voxel.world2local(nearpoint - farpoint, is_vec=True)
            cam_up = self.voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)
            cam_f = 0.5/np.tan(np.deg2rad(73/2)*zoom[i])  # about 24mm fov

            self.camera_poses.append((cam_ori, cam_dir, cam_up, cam_f))

    def _get_height(self, loc0, loc1, minheight):
        loc0 = int(loc0)
        loc1 = int(loc1)
        height = minheight
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if (loc0+dx) < 0 or (loc0+dx) >= self.voxel.heightmap.shape[0] or (loc1+dy) < 0 or \
                        (loc1+dy) >= self.voxel.heightmap.shape[1]:
                    height = max(height, minheight)
                else:
                    height = max(height, self.voxel.heightmap[loc0+dx, loc1+dy] + 2)
        return height

    def __len__(self):
        return len(self.camera_poses)

    def __getitem__(self, idx):
        return self.camera_poses[idx]


def rand_camera_pose_birdseye(voxel, border=128):
    r"""Generating random camera pose in the upper hemisphere, in the format of origin-direction-up
    Assuming [Y X Z] coordinate. Y is negative gravity direction.
    The camera pose is converted into the voxel coordinate system so that it can be used directly for rendering
    1. Uniformly sample a point on the upper hemisphere of a unit sphere, as cam_ori.
    2. Set cam_dir to be from cam_ori to the origin
    3. cam_up is always pointing towards sky
    4. move cam_ori to random place according to voxel size
    """
    cam_dir = torch.randn(3, dtype=torch.float32)
    cam_dir = cam_dir / torch.sqrt(torch.sum(cam_dir*cam_dir))
    cam_dir[0] = -torch.abs(cam_dir[0])
    cam_up = torch.tensor([1, 0, 0], dtype=torch.float32)

    # generate camera lookat target
    r = np.random.rand(2)
    r[0] *= voxel.voxel_t.size(1)-border-border
    r[1] *= voxel.voxel_t.size(2)-border-border
    r = r + border
    y = voxel.heightmap[int(r[0]+0.5), int(r[1]+0.5)] + (np.random.rand(1)-0.5) * 5
    cam_target = torch.tensor([y, r[0], r[1]], dtype=torch.float32)
    cam_ori = cam_target - cam_dir * (np.random.rand(1).item() * 100)
    cam_ori[0] = max(voxel.heightmap[int(cam_ori[1]+0.5), int(cam_ori[2]+0.5)]+2, cam_ori[0])
    # Translate to voxel coordinate
    cam_ori = voxel.world2local(cam_ori)
    cam_dir = voxel.world2local(cam_dir, is_vec=True)
    cam_up = voxel.world2local(cam_up, is_vec=True)

    return cam_ori, cam_dir, cam_up


def get_neighbor_height(heightmap, loc0, loc1, minheight, neighbor_size=7):
    loc0 = int(loc0)
    loc1 = int(loc1)
    height = 0
    for dx in range(-neighbor_size//2, neighbor_size//2+1):
        for dy in range(-neighbor_size//2, neighbor_size//2+1):
            if (loc0+dx) < 0 or (loc0+dx) >= heightmap.shape[0] or (loc1+dy) < 0 or (loc1+dy) >= heightmap.shape[1]:
                height = max(height, minheight)
            else:
                height = max(minheight, heightmap[loc0+dx, loc1+dy] + 2)
    return height


def rand_camera_pose_firstperson(voxel, border=128):
    r"""Generating random camera pose in the upper hemisphere, in the format of origin-direction-up
    """
    r = np.random.rand(5)
    r[0] *= voxel.voxel_t.size(1)-border-border
    r[1] *= voxel.voxel_t.size(2)-border-border
    r[0] = r[0] + border
    r[1] = r[1] + border

    y = get_neighbor_height(voxel.heightmap, r[0], r[1], 0) + np.random.rand(1) * 15

    cam_ori = torch.tensor([y, r[0], r[1]], dtype=torch.float32)

    rand_ang_h = r[2] * 2 * np.pi
    cam_target = torch.tensor([0, cam_ori[1]+np.sin(rand_ang_h)*border*r[4], cam_ori[2] +
                              np.cos(rand_ang_h)*border*r[4]], dtype=torch.float32)
    cam_target[0] = get_neighbor_height(voxel.heightmap, cam_target[1],
                                        cam_target[2], 0, neighbor_size=1) - 2 + r[3] * 10

    cam_dir = cam_target - cam_ori

    cam_up = torch.tensor([1, 0, 0], dtype=torch.float32)

    cam_ori = voxel.world2local(cam_ori)
    cam_dir = voxel.world2local(cam_dir, is_vec=True)
    cam_up = voxel.world2local(cam_up, is_vec=True)

    return cam_ori, cam_dir, cam_up


def rand_camera_pose_thridperson(voxel, border=96):
    r = torch.rand(2)
    r[0] *= voxel.voxel_t.size(1)
    r[1] *= voxel.voxel_t.size(2)
    rand_height = 60 + torch.rand(1) * 40
    rand_height = get_neighbor_height(voxel.heightmap, r[0], r[1], rand_height, neighbor_size=5)
    farpoint = torch.tensor([rand_height, r[0], r[1]], dtype=torch.float32)

    r = torch.rand(2)
    r[0] *= voxel.voxel_t.size(1) - border - border
    r[1] *= voxel.voxel_t.size(2) - border - border
    r[0] = r[0] + border
    r[1] = r[1] + border
    rand_height = get_neighbor_height(voxel.heightmap, r[0], r[1], 65, neighbor_size=1) - 5
    nearpoint = torch.tensor([rand_height, r[0], r[1]], dtype=torch.float32)

    cam_ori = voxel.world2local(farpoint)
    cam_dir = voxel.world2local(nearpoint - farpoint, is_vec=True)
    cam_up = voxel.world2local(torch.tensor([1, 0, 0], dtype=torch.float32), is_vec=True)

    return cam_ori, cam_dir, cam_up


def rand_camera_pose_thridperson2(voxel, border=48):
    r = torch.rand(2)
    r[0] *= voxel.voxel_t.size(1) - border - border
    r[1] *= voxel.voxel_t.size(2) - border - border
    r[0] = r[0] + border
    r[1] = r[1] + border
    rand_height = 60 + torch.rand(1) * 40
    rand_height = get_neighbor_height(voxel.heightmap, r[0], r[1], rand_height, neighbor_size=5)
    farpoint = torch.tensor([rand_height, r[0], r[1]], dtype=torch.float32)

    r = torch.rand(2)
    r[0] *= voxel.voxel_t.size(1) - border - border
    r[1] *= voxel.voxel_t.size(2) - border - border
    r[0] = r[0] + border
    r[1] = r[1] + border
    rand_height = get_neighbor_height(voxel.heightmap, r[0], r[1], 65, neighbor_size=1) - 5
    nearpoint = torch.tensor([rand_height, r[0], r[1]], dtype=torch.float32)

    # Random Up vector (tilt a little bit)
    # up = torch.randn(3) * 0.05 # cutoff +-0.1, Tan(10deg) = 0.176
    up = torch.randn(3) * 0.02
    up[0] = 1.0
    up = up / up.norm(p=2)
    cam_ori = voxel.world2local(farpoint)
    cam_dir = voxel.world2local(nearpoint - farpoint, is_vec=True)
    cam_up = voxel.world2local(up, is_vec=True)

    return cam_ori, cam_dir, cam_up


def rand_camera_pose_thridperson3(voxel, border=64):
    r"""Attempting to solve the camera too close to wall problem and the lack of aerial poses."""
    r = torch.rand(2)
    r[0] *= voxel.voxel_t.size(1) - border - border
    r[1] *= voxel.voxel_t.size(2) - border - border
    r[0] = r[0] + border
    r[1] = r[1] + border
    rand_height = 60 + torch.rand(1) * 40
    if torch.rand(1) > 0.8:
        rand_height = 60 + torch.rand(1) * 60
    rand_height = get_neighbor_height(voxel.heightmap, r[0], r[1], rand_height, neighbor_size=7)
    farpoint = torch.tensor([rand_height, r[0], r[1]], dtype=torch.float32)

    r = torch.rand(2)
    r[0] *= voxel.voxel_t.size(1) - border - border
    r[1] *= voxel.voxel_t.size(2) - border - border
    r[0] = r[0] + border
    r[1] = r[1] + border
    rand_height = get_neighbor_height(voxel.heightmap, r[0], r[1], 65, neighbor_size=3) - 5
    nearpoint = torch.tensor([rand_height, r[0], r[1]], dtype=torch.float32)

    # Random Up vector (tilt a little bit)
    # up = torch.randn(3) * 0.05 # cutoff +-0.1, Tan(10deg) = 0.176
    up = torch.randn(3) * 0.02
    up[0] = 1.0
    up = up / up.norm(p=2)
    # print(up)
    cam_ori = voxel.world2local(farpoint)
    cam_dir = voxel.world2local(nearpoint - farpoint, is_vec=True)
    cam_up = voxel.world2local(up, is_vec=True)

    return cam_ori, cam_dir, cam_up


def rand_camera_pose_tour(voxel):
    size = min(voxel.voxel_t.size(1), voxel.voxel_t.size(2)) / 2
    center = [voxel.voxel_t.size(1)/2, voxel.voxel_t.size(2)/2]

    rnd = torch.rand(8)

    rnd_deg = torch.rand(1) * 2 * np.pi
    far_radius = rnd[0]*0.8+0.2
    far_height = rnd[1]*30 + 60
    farpoint = torch.tensor([
        far_height,
        torch.sin(rnd_deg)*size*far_radius + center[0],
        torch.cos(rnd_deg)*size*far_radius + center[1]])

    farpoint[0] = get_neighbor_height(voxel.heightmap, farpoint[1], farpoint[2], farpoint[0], neighbor_size=7)

    near_radius = far_radius * rnd[2]
    near_shift_rad = np.pi*(rnd[3]-0.5)
    near_height = 60 + rnd[4] * 10
    nearpoint = torch.tensor([
        near_height,
        torch.sin(rnd_deg+near_shift_rad)*size*near_radius + center[0],
        torch.cos(rnd_deg+near_shift_rad)*size*near_radius + center[1]])

    # Random Up vector (tilt a little bit)
    # up = torch.randn(3) * 0.05 # cutoff +-0.1, Tan(10deg) = 0.176
    up = torch.randn(3) * 0.02
    up[0] = 1.0
    up = up / up.norm(p=2)
    cam_ori = voxel.world2local(farpoint)
    cam_dir = voxel.world2local(nearpoint - farpoint, is_vec=True)
    cam_up = voxel.world2local(up, is_vec=True)
    cam_f = 0.5/np.tan(np.deg2rad(73/2)*(rnd[5]*0.75+0.25))  # about 24mm fov

    return cam_ori, cam_dir, cam_up, cam_f

# Look from center to outward


def rand_camera_pose_insideout(voxel):
    size = min(voxel.voxel_t.size(1), voxel.voxel_t.size(2)) / 2
    center = [voxel.voxel_t.size(1)/2, voxel.voxel_t.size(2)/2]

    rnd = torch.rand(8)

    rnd_deg = torch.rand(1) * 2 * np.pi
    far_radius = rnd[0]*0.8+0.2
    far_height = rnd[1]*10 + 60
    farpoint = torch.tensor([
        far_height,
        torch.sin(rnd_deg)*size*far_radius + center[0],
        torch.cos(rnd_deg)*size*far_radius + center[1]])

    near_radius = far_radius * rnd[2]
    near_shift_rad = np.pi*(rnd[3]-0.5)
    near_height = 60 + rnd[4] * 30
    nearpoint = torch.tensor([
        near_height,
        torch.sin(rnd_deg+near_shift_rad)*size*near_radius + center[0],
        torch.cos(rnd_deg+near_shift_rad)*size*near_radius + center[1]])

    nearpoint[0] = get_neighbor_height(voxel.heightmap, nearpoint[1], nearpoint[2], nearpoint[0], neighbor_size=7)

    # Random Up vector (tilt a little bit)
    # up = torch.randn(3) * 0.05 # cutoff +-0.1, Tan(10deg) = 0.176
    up = torch.randn(3) * 0.02
    up[0] = 1.0
    up = up / up.norm(p=2)
    cam_ori = voxel.world2local(nearpoint)
    cam_dir = voxel.world2local(farpoint-nearpoint, is_vec=True)
    cam_up = voxel.world2local(up, is_vec=True)
    cam_f = 0.5/np.tan(np.deg2rad(73/2)*(rnd[5]*0.75+0.25))  # about 24mm fov

    return cam_ori, cam_dir, cam_up, cam_f
