"""Official policy/native controller episode with passive, timestamped traces.

One fixed seed per process; never substitutes seeds or changes success predicates.
"""
import argparse
import faulthandler
import gzip
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def task_args(root, task, condition, output):
    import yaml
    cfg = root/'env_cfg/task_config'
    args = yaml.safe_load((cfg/f'{condition}.yml').read_text())
    embodiments = yaml.safe_load((cfg/'_embodiment_config.yml').read_text())
    assert args['embodiment'] == ['aloha-agilex']
    robot = embodiments['aloha-agilex']['file_path']
    args.update(task_name=task, task_config=condition, ckpt_setting='official_release',
                save_path=str(output), eval_mode=True, save_data=False,
                eval_video_log=False, eval_video_save_dir=None,
                left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True)
    for side in ['left', 'right']:
        args[f'{side}_embodiment_config'] = yaml.safe_load((root/robot/'config.yml').read_text())
    camera = yaml.safe_load((cfg/'_camera_config.yml').read_text())[args['camera']['head_camera_type']]
    args.update(head_camera_h=camera['h'], head_camera_w=camera['w'])
    return args


def snapshot(env, evaluate_predicate=True):
    robot = env.robot
    state = {'eef_world': {}, 'tcp_center_world': {}, 'raw_endlink_world': {},
             'joint_qpos': {}, 'joint_drive_target': {}, 'gripper_command': {},
             'gripper_measured_joint_qpos': {}, 'objects': {}, 'contacts': []}
    for side in ['left', 'right']:
        state['eef_world'][side] = getattr(robot, f'get_{side}_ee_pose')()
        state['tcp_center_world'][side] = getattr(robot, f'get_{side}_tcp_pose')()
        pose = getattr(robot, f'{side}_ee').global_pose
        state['raw_endlink_world'][side] = np.r_[pose.p, pose.q]
        entity = getattr(robot, f'{side}_entity')
        qpos = entity.get_qpos()
        joints = entity.get_active_joints()
        state['joint_qpos'][side] = qpos
        state['joint_drive_target'][side] = getattr(robot, f'get_{side}_arm_jointState')()[:-1]
        state['gripper_command'][side] = getattr(robot, f'get_{side}_gripper_val')()
        state['gripper_measured_joint_qpos'][side] = {
            j[0].get_name(): float(qpos[joints.index(j[0])]) for j in getattr(robot, f'{side}_gripper')}
    # Task actor attributes include the entities referenced by native predicates.
    for name, value in vars(env).items():
        if name.startswith('_') or not hasattr(value, 'get_pose'): continue
        try:
            pose = value.get_pose()
            item = {'pose_world': np.r_[pose.p, pose.q]}
            item['qpos'] = value.get_qpos() if hasattr(value, 'get_qpos') else 'unavailable: no articulation qpos API on this actor'
            item['qlimits'] = value.get_qlimits() if hasattr(value, 'get_qlimits') else 'unavailable: no articulation limits API on this actor'
            state['objects'][name] = item
        except Exception as e:
            state['objects'][name] = {'unavailable': str(e)}
    for c in env.scene.get_contacts():
        if c.points:
            state['contacts'].append({'bodies': [b.entity.name for b in c.bodies],
                'points': [{'position': p.position, 'normal': p.normal,
                            'impulse': p.impulse, 'separation': p.separation} for p in c.points]})
    state['native_success_predicate'] = (bool(env.check_success()) if evaluate_predicate else
        'unavailable: native predicate evaluated after this scene.step; see action_end/terminal')
    state['predicate_constants'] = {k: vars(env)[k] for k in ['start_height', 'object_start_height', 'origin_z', 'table_z_bias'] if k in vars(env)}
    task = type(env).__name__
    details = {}
    if task == 'lift_pot':
        from envs.utils import get_face_prod
        details = dict(pot_z=float(env.pot.get_pose().p[2]),
                       pot_axis_dot=float(get_face_prod(env.pot.get_pose().q,[0,0,1],[0,0,1])),
                       left_contact_point=env.pot.get_contact_point(0), right_contact_point=env.pot.get_contact_point(1))
    elif task == 'hanging_mug':
        mug_frame=env.mug.get_functional_point(0,ret='matrix')
        rack_frame=env.rack.get_functional_point(0,ret='matrix')
        details = dict(mug_functional_point=mug_frame[:3,3], rack_functional_point=rack_frame[:3,3],
                       mug_functional_frame_world=mug_frame, rack_functional_frame_world=rack_frame,
                       rack_middle=(env.rack.get_pose().p+rack_frame[:3,3])/2)
    elif task == 'place_can_basket':
        details = dict(can_contact_table=env.check_actors_contact('071_can','table'),
                       can_contact_basket=env.check_actors_contact('071_can','110_basket'),
                       basket_axis=env.basket.get_pose().to_transformation_matrix()[:3,1],
                       can_basket_l1_distance=float(np.abs(env.can.get_pose().p-env.basket.get_pose().p).sum()))
    elif task == 'stack_blocks_three':
        details = dict(layer2_delta=env.block2.get_pose().p-env.block1.get_pose().p,
                       layer3_delta=env.block3.get_pose().p-env.block2.get_pose().p)
    elif task == 'stack_bowls_three':
        z = sorted(float(getattr(env,f'bowl{i}').get_pose().p[2]) for i in [1,2,3])
        details = dict(sorted_z=z, signed_height_error=np.asarray(z)-np.array([.74,.77,.81])-env.table_z_bias)
    elif task == 'put_object_cabinet':
        details = dict(cabinet_target=env.cabinet.get_functional_point(0),
                       object_lift=float(env.object.get_pose().p[2]-env.origin_z))
    elif task == 'stamp_seal':
        details = dict(seal_target_delta=env.seal.get_pose().p-env.target.get_pose().p,
                       contact_requirement_in_native_predicate=False)
    state['predicate_details'] = details
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--robotwin', type=Path, required=True)
    p.add_argument('--official-code', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='lift_pot')
    p.add_argument('--condition', default='demo_clean', choices=['demo_clean', 'demo_randomized'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--port', type=int, default=10097)
    p.add_argument('--reset-only', action='store_true')
    p.add_argument('--serial-camera', action='store_true')
    a = p.parse_args()
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=False)
    a.robotwin = a.robotwin.resolve(); a.official_code = a.official_code.resolve()
    if a.checkpoint: a.checkpoint = a.checkpoint.resolve()
    sys.path[:0] = [str(a.official_code), str(a.robotwin), str(a.robotwin/'description/utils')]
    os.chdir(a.robotwin)
    faulthandler.enable(); faulthandler.dump_traceback_later(120, repeat=True)
    import torch
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.32)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    env = None; model = None; writer = None; tick = 0; query = 0
    started = time.monotonic()
    row = dict(source='infrastructure_gate' if a.reset_only else 'local_baseline', checkpoint=str(a.checkpoint), task=a.task, condition=a.condition,
               seed=a.seed, episode_id=a.output.name, status='started', success=None,
               episode_length=0, instruction=None, renderer_denoiser='none', replan_steps=36,
               action_execution='unmodified native take_action ee', flow_seed_rule='seed*10000+query_index',
               expert_filter='fixed seed; rejection recorded without replacement', reset_only=a.reset_only)
    row['video_views']='head_camera; all three raw RGB views saved at every policy query'
    row['instrumentation_version']='head_capture_matrix_diagnostics_v4'
    row['cpu_affinity']=sorted(os.sched_getaffinity(0))
    row['serial_camera_readback']=a.serial_camera
    def save(): (a.output/'summary.json').write_text(json.dumps(row, indent=2, default=plain))
    save()
    journal = (a.output/'episode.jsonl').open('w', buffering=1)
    trace = gzip.open(a.output/'physical_trace.jsonl.gz', 'wt', compresslevel=2)
    def event(kind, **values):
        journal.write(json.dumps(dict(event=kind, physics_tick=tick, sim_seconds=tick/250,
                                     wall_monotonic=time.monotonic(), **values), default=plain)+'\n')
    try:
        cls = getattr(importlib.import_module(f'envs.{a.task}'), a.task)
        env = cls()
        native_setup = env.setup_scene
        def setup_scene(**kwargs):
            native_setup(**kwargs)
            import sapien
            sapien.render.set_ray_tracing_denoiser('none')
        env.setup_scene = setup_scene
        args = task_args(a.robotwin, a.task, a.condition, a.output)
        (a.output/'resolved_task_config.json').write_text(json.dumps(args, indent=2))
        from envs.utils.create_actor import UnStableError
        try:
            env.setup_demo(now_ep_num=0, seed=a.seed, is_test=True, **args)
        except UnStableError:
            if a.reset_only: raise
            row.update(status='seed_rejected',expert_seed_valid=False,
                       rejection_reason='native initial-scene stability filter',
                       expert_probe_error=traceback.format_exc())
            event('expert_seed_probe',valid=False,error=row['expert_probe_error'])
            return
        event('reset_complete')
        if not a.reset_only:
            try:
                info = env.play_once()
            except AssertionError as exc:
                if str(exc) != 'target_pose cannot be None for move action.': raise
                # Official _try_seed_once also rejects expert exceptions before policy.
                row.update(status='seed_rejected',expert_seed_valid=False,
                           rejection_reason='native expert could not construct a grasp target',
                           expert_probe_error=traceback.format_exc())
                event('expert_seed_probe',valid=False,error=row['expert_probe_error'])
                return
            valid = bool(env.plan_success and env.check_success())
            event('expert_seed_probe', valid=valid, episode_info=info)
            row['expert_seed_valid'] = valid
            if not valid:
                row['status'] = 'seed_rejected'; return
            env.close_env()
            env.setup_demo(now_ep_num=0, seed=a.seed, is_test=True, **args)
            from generate_episode_instructions import generate_episode_descriptions
            random.seed(a.seed)
            descriptions = generate_episode_descriptions(a.task, [info['info']], 1)
            # Official LaWAM deploy_policy.yml fixes instruction_type=seen for both conditions.
            pool = descriptions[0]['seen']
            instruction = str(pool[np.random.default_rng(a.seed).integers(0, len(pool))])
            env.set_instruction(instruction)
            row['instruction'] = instruction
            from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient
            model = ModelClient(str(a.checkpoint), port=a.port, replan_steps=36, action_ensemble=False)
            predict = model.client.predict_action
            def predict_logged(payload):
                nonlocal query
                flow_seed = a.seed*10000 + query
                payload['flow_seed'] = flow_seed
                event('policy_query', query=query, flow_seed=flow_seed, instruction=instruction,
                      command_index=env.take_action_cnt)
                response = predict(payload)
                event('policy_output', query=query, response=response)
                query += 1
                return response
            model.client.predict_action = predict_logged
            unnormalize = model.unnormalize_actions
            def unnormalize_logged(*args, **kwargs):
                actions = unnormalize(*args, **kwargs)
                event('full_absolute_action_chunk', query=query-1, actions=actions)
                return actions
            model.unnormalize_actions = unnormalize_logged
        if a.serial_camera:
            # Same scene/camera order and RGB conversion; finish each read before next capture.
            cameras=env.cameras
            camera_pairs=[('left_camera',cameras.left_camera),('right_camera',cameras.right_camera)]
            camera_pairs+=list(zip(cameras.static_camera_name,cameras.static_camera_list))
            rgba_cache={}
            def serial_picture():
                rgba_cache.clear()
                for name,camera in camera_pairs:
                    camera.take_picture()
                    rgba_cache[name]={'rgba':(camera.get_picture('Color')*255).clip(0,255).astype('uint8')}
            cameras.update_picture=serial_picture
            cameras.get_rgba=lambda:rgba_cache
        native_step = env.scene.step
        def step():
            nonlocal tick
            native_step(); tick += 1
            # Every physics tick, including realized state and contact impulses.
            trace.write(json.dumps(dict(physics_tick=tick, sim_seconds=tick/250,
                                       command_index=env.take_action_cnt, **snapshot(env,evaluate_predicate=False)), default=plain)+'\n')
        env.scene.step = step
        for side in ['left', 'right']:
            native_plan = getattr(env.robot, f'{side}_plan_path')
            def plan_logged(*args, _side=side, _plan=native_plan, **kwargs):
                result = _plan(*args, **kwargs)
                event('planner_result', side=_side, target=args[0] if args else kwargs,
                      result=result)
                return result
            setattr(env.robot, f'{side}_plan_path', plan_logged)
        import imageio.v2 as imageio
        from PIL import Image
        writer = imageio.get_writer(str(a.output/'video.mp4'), fps=10, codec='libx264', quality=7)
        (a.output/'query_rgb').mkdir()
        while True:
            faulthandler.dump_traceback_later(120, exit=True)
            needs_query = a.reset_only or model.needs_query(task_description=row['instruction'])
            if needs_query:
                obs = env.get_obs()
                rgb = obs['observation']
                images = [rgb[cam]['rgb'] for cam in ['head_camera', 'left_camera', 'right_camera']]
            else:
                # Head-only diagnostic video; policy queries retain all three native cameras.
                # Avoid extra wrist readbacks and randomized-light/RNG updates.
                env.scene.update_render()
                camera=env.cameras.static_camera_list[env.cameras.static_camera_name.index('head_camera')]
                camera.take_picture()
                images=[(camera.get_picture('Color')[:,:,:3]*255).clip(0,255).astype('uint8')]
            assert all(np.isfinite(im).all() and im.std()>2 for im in images)
            writer.append_data(images[0])
            event('video_frame', frame=int(env.take_action_cnt), playback_fps=10,
                  clock='one frame per action command; not physical-time video')
            if needs_query:
                for cam, im in zip(['head', 'left', 'right'], images):
                    Image.fromarray(im).save(a.output/'query_rgb'/f'{query:04d}_{cam}.png')
                event('observation', query=query, state=snapshot(env), raw_endpose=obs['endpose'],
                      rgb_paths=[f'query_rgb/{query:04d}_{cam}.png' for cam in ['head','left','right']])
            if a.reset_only:
                for _ in range(25): step()
                row['status']='reset_rgb_passed'; break
            action = (model.step(model.build_example(row['instruction'], obs), step=env.take_action_cnt)
                      if needs_query else model.step_cached(task_description=row['instruction']))
            assert action.shape == (16,) and np.isfinite(action).all()
            event('action_start', command_index=env.take_action_cnt, executed_action=action)
            env.take_action(action, action_type=model.env_action_type)
            row['episode_length'] = int(env.take_action_cnt)
            event('action_end', command_index=env.take_action_cnt, state=snapshot(env))
            save()
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                terminal_obs=env.get_obs()
                terminal_images=[terminal_obs['observation'][cam]['rgb'] for cam in ['head_camera','left_camera','right_camera']]
                writer.append_data(terminal_images[0])
                for cam,im in zip(['head','left','right'],terminal_images):
                    Image.fromarray(im).save(a.output/f'final_{cam}.png')
                event('terminal_observation', state=snapshot(env), frame=int(env.take_action_cnt))
                row.update(status='completed', success=bool(env.eval_success)); break
    except BaseException:
        row.update(status='system_error', error=traceback.format_exc()); print(row['error'], flush=True)
    finally:
        row.update(physics_ticks=tick, simulated_seconds=tick/250, policy_queries=query,
                   elapsed_seconds=time.monotonic()-started)
        event('episode_end', summary=row)
        if writer: writer.close()
        trace.close(); journal.close(); save()
        if model: model.close()
        if env:
            try: env.close_env()
            except Exception: pass
        faulthandler.cancel_dump_traceback_later()
        print(json.dumps(row, default=plain), flush=True)
    if row['status']=='system_error': raise SystemExit(1)


if __name__ == '__main__': main()
