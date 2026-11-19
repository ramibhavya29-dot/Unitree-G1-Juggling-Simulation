#!/usr/bin/env python3
"""Continuous one-ball hand-to-hand juggling for Unitree G1 in MuJoCo.

This is a GUIDED visual demo, not a physical grasp/contact simulation.
The ball follows a continuous gravity-shaped flight path. During catches it
smoothly blends to the receiving palm; it is NEVER snapped/teleported to it.
The model's apple contact is disabled to prevent rolling off open palms.

Windows PowerShell:
  python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml
  python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --headless --cycles 4
  python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --hold

The GUI loops until the viewer is closed. Ctrl+C stops it from PowerShell.
"""
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

G = 9.81
BALL_RADIUS = 0.04
ROT_W = 0.4


def quat_from_z(n):
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    v = np.cross([0.0, 0.0, 1.0], n)
    w = 1.0 + n[2]
    if w < 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([w, *v])
    return q / np.linalg.norm(q)


def pick_body(spec, side):
    names = [b.name for b in spec.bodies if b.name]
    for pref in ('hand', 'wrist_yaw', 'wrist', 'elbow'):
        match = [n for n in names if side in n.lower() and pref in n.lower()]
        if match:
            return match[-1]
    raise SystemExit(f'No {side} wrist/hand body found. Available bodies: {names}')


def build_model(args):
    spec = mujoco.MjSpec.from_file(args.xml)
    for key in list(spec.keys):
        spec.delete(key)
    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(joint)

    left_body = args.left_body or pick_body(spec, 'left')
    right_body = args.right_body or pick_body(spec, 'right')
    print(f'Hand bodies: left={left_body}  right={right_body}', flush=True)

    for body_name, site_name, normal in (
        (left_body, 'l_palm', args.normal_l),
        (right_body, 'r_palm', args.normal_r),
    ):
        b = spec.body(body_name)
        off = np.asarray(args.palm_offset, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        b.add_site(name=site_name, pos=off, size=[0.012, 0.012, 0.012],
                   rgba=[0.0, 1.0, 0.1, 1.0])
        if not args.no_plate:
            b.add_geom(name=site_name + '_plate',
                       type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.055, 0.004, 0.001],
                       pos=off - normal * 0.004,
                       quat=quat_from_z(normal),
                       rgba=[0.3, 0.3, 0.3, 0.6],
                       contype=0, conaffinity=0)

    apple = spec.worldbody.add_body(name='apple', pos=[0, 0, 2])
    apple.add_freejoint()
    apple.add_geom(name='apple_geom', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[BALL_RADIUS, 0, 0], mass=0.15,
                   rgba=[0.9, 0.1, 0.1, 1],
                   contype=0, conaffinity=0)
    model = spec.compile()
    # Scripted ball tracks exact positions and should not acquire downward speed.
    model.body_gravcomp[model.body('apple').id] = 1.0
    return model


class Hand:
    def __init__(self, model, side, site, normal):
        self.site_id = model.site(site).id
        self.local_normal = np.asarray(normal, dtype=float)
        self.local_normal /= np.linalg.norm(self.local_normal)
        jids = []
        for j in range(model.njnt):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower()
            if side in name and any(s in name for s in ('shoulder', 'elbow', 'wrist')):
                jids.append(j)
        if not jids:
            raise SystemExit(f'Could not find {side} arm joints')
        self.jids = jids
        self.qadr = np.asarray([model.jnt_qposadr[j] for j in jids], dtype=int)
        self.dadr = np.asarray([model.jnt_dofadr[j] for j in jids], dtype=int)

    def pos(self, data):
        return data.site_xpos[self.site_id].copy()

    def normal(self, data):
        return data.site_xmat[self.site_id].reshape(3, 3) @ self.local_normal

    def ik(self, model, data, target, iterations=6):
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        for _ in range(iterations):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jp, jr, self.site_id)
            J = np.vstack((jp[:, self.dadr], ROT_W * jr[:, self.dadr]))
            error = np.r_[target - self.pos(data),
                          ROT_W * np.cross(self.normal(data), [0, 0, 1.0])]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), error)
            data.qpos[self.qadr] += np.clip(dq, -0.2, 0.2)
            for index, jid in enumerate(self.jids):
                if model.jnt_limited[jid]:
                    data.qpos[self.qadr[index]] = np.clip(
                        data.qpos[self.qadr[index]], *model.jnt_range[jid])


def control(model, data, qtarget, kp, kd):
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        if jid < 0:
            continue
        qa = model.jnt_qposadr[jid]
        da = model.jnt_dofadr[jid]
        if model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE:
            u = qtarget[qa]
        else:
            u = kp * (qtarget[qa] - data.qpos[qa]) - kd * data.qvel[da]
        if model.actuator_ctrllimited[i]:
            u = np.clip(u, *model.actuator_ctrlrange[i])
        data.ctrl[i] = u


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--xml', required=True)
    parser.add_argument('--left-body')
    parser.add_argument('--right-body')
    parser.add_argument('--palm-offset', type=float, nargs=3, default=[0.08, 0, 0])
    parser.add_argument('--normal-l', type=float, nargs=3, default=[0, 0, 1])
    parser.add_argument('--normal-r', type=float, nargs=3, default=[0, 0, 1])
    parser.add_argument('--release', type=float, nargs=3, default=[0.25, 0.13, 0.22],
                        help='left hand catch/release pose relative to robot base')
    parser.add_argument('--catch', type=float, nargs=3, default=[0.25, -0.13, 0.22],
                        help='right hand catch/release pose relative to robot base')
    parser.add_argument('--flight', type=float, default=0.65,
                        help='time in seconds for each throw (default 0.65)')
    parser.add_argument('--carry-time', type=float, default=0.65,
                        help='prepare and lift the throwing hand (default 0.65)')
    parser.add_argument('--settle-time', type=float, default=0.18,
                        help='smooth hand catch settling time (default 0.18)')
    parser.add_argument('--lift', type=float, default=0.10,
                        help='distance the throw arm rises (m)')
    parser.add_argument('--kp', type=float, default=500.0)
    parser.add_argument('--kd', type=float, default=25.0)
    parser.add_argument('--slowmo', type=float, default=1.0)
    parser.add_argument('--no-plate', action='store_true')
    parser.add_argument('--hold', action='store_true', help='hold the ball on left palm')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cycles', type=int, default=4,
                        help='number of left-right flights in --headless / --video')
    parser.add_argument('--video', help='render to mp4 (requires mediapy and OpenGL)')
    args = parser.parse_args()
    if min(args.flight, args.carry_time, args.settle_time, args.slowmo) <= 0:
        parser.error('flight, carry-time, settle-time and slowmo must be positive')
    if args.cycles < 1:
        parser.error('--cycles must be >= 1')

    model = build_model(args)
    data = mujoco.MjData(model)
    ik = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = data.xpos[1].copy()
    left = Hand(model, 'left', 'l_palm', args.normal_l)
    right = Hand(model, 'right', 'r_palm', args.normal_r)
    hands = [left, right]
    targets = [base + np.asarray(args.release), base + np.asarray(args.catch)]
    starts = [t - np.array([0, 0, args.lift]) for t in targets]
    apple_joint = model.body_jntadr[model.body('apple').id]
    ad = model.jnt_qposadr[apple_joint]
    vd = model.jnt_dofadr[apple_joint]
    dt = model.opt.timestep
    separation = BALL_RADIUS + 0.006

    # Start at the 'ready' positions, and solve both wrist poses.
    ik.qpos[:] = model.qpos0
    left.ik(model, ik, starts[0], iterations=300)
    right.ik(model, ik, targets[1], iterations=300)
    mujoco.mj_kinematics(model, ik)
    data.qpos[:] = ik.qpos
    data.qpos[ad:ad+7] = [*(left.pos(ik) + left.normal(ik)*separation), 1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # One state machine alternates L->R and R->L without restarting the sim.
    state = dict(phase='carry', thrower=0, start_time=0.0, throws=0,
                 carry_start=left.pos(data).copy(),
                 p0=None, destination=None, velocity=None,
                 settle_start=None, last_ball=data.qpos[ad:ad+3].copy(),
                 last_log=-1)

    def palm_ball(hand):
        return hand.pos(data) + hand.normal(data)*separation

    def set_ball(position):
        position = np.asarray(position, dtype=float)
        # Updates run each integration step: positions remain continuous.
        data.qpos[ad:ad+3] = position
        data.qpos[ad+3:ad+7] = [1., 0., 0., 0.]
        data.qvel[vd:vd+3] = (position - state['last_ball']) / dt
        data.qvel[vd+3:vd+6] = 0
        state['last_ball'] = position.copy()

    def step_command(t):
        owner = state['thrower']
        recv = 1-owner
        elapsed = t - state['start_time']

        if args.hold:
            l_goal, r_goal = starts[0], targets[1]
        elif state['phase'] == 'carry':
            progress = smooth(elapsed/args.carry_time)
            path = starts[owner] + (targets[owner] - starts[owner])*progress
            goals = [targets[0].copy(), targets[1].copy()]
            goals[owner] = path
            l_goal, r_goal = goals
        else:
            l_goal, r_goal = targets[0], targets[1]

        # Solve joint targets and apply motors. Actual palms may lag targets.
        left.ik(model, ik, l_goal)
        right.ik(model, ik, r_goal)
        control(model, data, ik.qpos, args.kp, args.kd)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        if args.hold:
            set_ball(palm_ball(left))
            return

        if state['phase'] == 'carry':
            set_ball(palm_ball(hands[owner]))
            if elapsed >= args.carry_time:
                # The exact ball position at takeoff is reused as the FIRST
                # trajectory point. This prevents release teleportation.
                p0 = state['last_ball'].copy()
                destination = palm_ball(hands[recv])
                T = args.flight
                velocity = (destination-p0)/T + np.array([0, 0, 0.5*G*T])
                state.update(phase='flight', start_time=t, p0=p0,
                             destination=destination, velocity=velocity)
                print(f"THROW {state['throws']+1}: "
                      f"{'LEFT -> RIGHT' if owner == 0 else 'RIGHT -> LEFT'} "
                      f"v={np.round(velocity, 2)} m/s", flush=True)
        elif state['phase'] == 'flight':
            tau = min(elapsed, args.flight)
            position = state['p0'] + state['velocity']*tau - np.array([0, 0, 0.5*G*tau*tau])
            set_ball(position)
            if elapsed >= args.flight:
                # Ball ends at destination with NO sudden position reset.
                state.update(phase='settle', start_time=t,
                             settle_start=position.copy())
        elif state['phase'] == 'settle':
            # Blend the actual landing position into the moving hand smoothly.
            weight = smooth(elapsed/args.settle_time)
            position = (1-weight)*state['settle_start'] + weight*palm_ball(hands[recv])
            set_ball(position)
            if elapsed >= args.settle_time:
                state['throws'] += 1
                state.update(phase='carry', thrower=recv, start_time=t,
                             carry_start=palm_ball(hands[recv]))
                print(f"CATCH {state['throws']}: "
                      f"{'RIGHT' if recv == 1 else 'LEFT'} hand", flush=True)

    t_end = 6.0 if args.hold else args.cycles*(args.carry_time + args.flight + args.settle_time)+0.01

    def log_report():
        p = data.qpos[ad:ad+3]
        print(f"t={data.time:.2f}s | phase={state['phase']} | "
              f"apple={np.round(p, 3)} | successful guided transfers={state['throws']}")

    if args.video:
        import mediapy as media
        renderer = mujoco.Renderer(model, width=640, height=480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = base + [0.3, 0, 0.2]
        cam.distance, cam.azimuth, cam.elevation = 1.8, 150, -14
        frames, fps = [], 60
        next_frame = 0
        while data.time < t_end:
            step_command(data.time)
            mujoco.mj_step(model, data)
            if data.time >= next_frame / fps:
                renderer.update_scene(data, cam)
                frames.append(renderer.render())
                next_frame += 1
        media.write_video(args.video, frames, fps=fps / args.slowmo)
        renderer.close()
        log_report()
        print('Wrote video:', args.video)
        return

    if args.headless:
        next_log = 1.0
        while data.time < t_end:
            step_command(data.time)
            mujoco.mj_step(model, data)
            if data.time >= next_log:
                log_report()
                next_log += 1.0
        log_report()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = base + [0.3, 0, 0.2]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 150, -14
        sync_every = max(1, int(1 / (60*dt)))
        steps = 0
        while viewer.is_running():
            started = time.perf_counter()
            step_command(data.time)
            mujoco.mj_step(model, data)
            steps += 1
            if steps % sync_every == 0:
                viewer.sync()
            if args.hold and data.time >= 6.0:
                log_report()
                break
            time.sleep(max(0, dt*args.slowmo - (time.perf_counter()-started)))


if __name__ == '__main__':
    main()
