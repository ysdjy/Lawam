"""Bounded real renderer/readback gate; run in the simulator Python."""
import argparse
import faulthandler
import json
from pathlib import Path
import numpy as np
import sapien

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--shader', choices=['default', 'rt'], default='default')
p.add_argument('--denoiser', default='none')
p.add_argument('--frames', type=int, default=40)
p.add_argument('--cameras', type=int, default=1)
p.add_argument('--timeout', type=int, default=45)
a = p.parse_args()
faulthandler.enable()
faulthandler.dump_traceback_later(a.timeout, exit=True)
a.output.mkdir(parents=True, exist_ok=True)
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
if a.shader == 'rt':
    sapien.render.set_camera_shader_dir('rt')
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser(a.denoiser)
scene = engine.create_scene()
scene.set_timestep(1 / 250)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([1, 0, -1], [1, 1, 1])
scene.add_ground(0)
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.1]*3, material=[1, 0.1, 0.1])
actor = builder.build_kinematic(name='moving_cube')
cameras = [scene.add_camera(f'gate{i}', 128, 128, 1.0, 0.01, 10) for i in range(a.cameras)]
for camera in cameras: camera.set_pose(sapien.Pose([-1, 0, 0.3]))
rows = []
for i in range(a.frames):
    actor.set_pose(sapien.Pose([0, (i%40-20)*0.006, 0.12]))
    scene.step()
    scene.update_render()
    for camera in cameras: camera.take_picture()
    for camera in cameras:
        rgb = camera.get_picture('Color')[..., :3].copy()
        assert np.isfinite(rgb).all() and rgb.std() > 0.01
    rows.append(dict(frame=i, mean=float(rgb.mean()), std=float(rgb.std())))
    print(json.dumps(rows[-1]), flush=True)
from PIL import Image
Image.fromarray((np.clip(rgb, 0, 1)*255).astype('uint8')).save(a.output/'last_rgb.png')
assert max(r['mean'] for r in rows) > min(r['mean'] for r in rows)
(a.output/'result.json').write_text(json.dumps(dict(status='passed', frames=rows, shader=a.shader, denoiser=a.denoiser), indent=2))
faulthandler.cancel_dump_traceback_later()
