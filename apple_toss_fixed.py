#!/usr/bin/env python3
"""
Unitree humanoid tosses an apple from the left hand to the right hand (MuJoCo).

  python apple_toss.py --xml path/to/g1.xml            # with viewer
  python apple_toss.py --xml path/to/g1.xml --headless  # prints catch error, good for tuning

Requires mujoco >= 3.2 (uses MjSpec to edit the model in Python).

How it works
  1. Loads your robot XML, pins the base (removes the free joint), adds an apple
     (free body) and a "palm" site + flat palm plate on each hand body.
  2. Plans a ballistic toss: given release point, catch point and flight time,
     it computes the release velocity v = (p_catch - p_rel)/T + 0.5*g*T.
  3. A temporary scripted grip keeps the apple centered on the left palm
     until release (avoids the rolling-off problem shown by --hold).
  4. At release, the apple becomes a freely simulated MuJoCo sphere with a
     calculated ballistic velocity. The right hand attempts to catch it.
     Optional --assist-catch provides an explicit scripted catch aid.
  5. Palm targets -> joint targets via damped least-squares IK (position + keep
     palm normal pointing up). Non-arm joints hold qpos0.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

G = 9.81
APPLE_R = 0.04
ROT_W = 0.4  # weight of orientation error in IK


# --------------------------------------------------------------------------- model
def quat_from_z(n):
    """Quaternion (w,x,y,z) rotating +z onto unit vector n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    v = np.cross([0, 0, 1.0], n)
    w = 1.0 + n[2]
    if w < 1e-8:
        return np.array([0, 1.0, 0, 0])
    q = np.array([w, *v])
    return q / np.linalg.norm(q)


def pick_body(spec, side, prefs):
    names = [b.name for b in spec.bodies if b.name]
    for p in prefs:
        hits = [n for n in names if side in n.lower() and p in n.lower()]
        if hits:
            return hits[-1]
    raise SystemExit(f"Could not find a {side} hand body; pass --{side}-body. Bodies: {names}")


def build_model(a):
    spec = mujoco.MjSpec.from_file(a.xml)

    # Pin the base: drop free joint(s) and keyframes (their qpos would no longer match)
    for k in list(spec.keys):
        spec.delete(k)
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)

    prefs = ["hand", "wrist_yaw", "wrist", "elbow"]
    lb = a.left_body or pick_body(spec, "left", prefs)
    rb = a.right_body or pick_body(spec, "right", prefs)
    print(f"Hand bodies: left={lb}  right={rb}")

    for body_name, site, normal in ((lb, "l_palm", a.normal_l), (rb, "r_palm", a.normal_r)):
        b = spec.body(body_name)
        off = np.array(a.palm_offset, float)
        b.add_site(name=site, pos=off, size=[0.012, 0, 0], rgba=[0, 1, 0, 1])
        if not a.no_plate:  # flat disc so the apple has something to sit on
            n = np.asarray(normal, float) / np.linalg.norm(normal)
            b.add_geom(
                name=site + "_plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.09, 0.004, 0.001], pos=off - n * 0.004, quat=quat_from_z(n),
                rgba=[0.3, 0.3, 0.3, 0.6], friction=[3.0, 0.05, 0.01],
            )

    apple = spec.worldbody.add_body(name="apple", pos=[0, 0, 2])
    apple.add_freejoint()
    apple.add_geom(
        name="apple_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[APPLE_R, 0, 0],
        mass=0.15, rgba=[0.8, 0.1, 0.1, 1], friction=[1.2, 0.02, 0.002], condim=4,
    )
    return spec.compile()


# --------------------------------------------------------------------------- hand / IK
class Hand:
    def __init__(self, model, side, site, normal):
        self.site = model.site(site).id
        self.normal = np.asarray(normal, float) / np.linalg.norm(normal)
        self.jids = []
        for j in range(model.njnt):
            n = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
            if side in n and any(k in n for k in ("shoulder", "elbow", "wrist")):
                self.jids.append(j)
        if not self.jids:
            raise SystemExit(f"No {side} arm joints found (looking for shoulder/elbow/wrist in names)")
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])

    def pos(self, d):
        return d.site_xpos[self.site].copy()

    def normal_w(self, d):
        return d.site_xmat[self.site].reshape(3, 3) @ self.normal

    def ik(self, model, d, target, iters=4):
        up = np.array([0, 0, 1.0])
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            mujoco.mj_jacSite(model, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], ROT_W * jr[:, self.dadr]])
            err = np.concatenate([target - d.site_xpos[self.site],
                                  ROT_W * np.cross(self.normal_w(d), up)])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[self.qadr] += np.clip(dq, -0.2, 0.2)
            for k, j in enumerate(self.jids):
                if model.jnt_limited[j]:
                    lo, hi = model.jnt_range[j]
                    d.qpos[self.qadr[k]] = np.clip(d.qpos[self.qadr[k]], lo, hi)


def apply_control(model, data, qtarget, kp, kd):
    """Position actuators get q_target directly; torque motors get a PD torque."""
    for i in range(model.nu):
        j = model.actuator_trnid[i, 0]
        qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
        if model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE:
            u = qtarget[qa]
        else:
            u = kp * (qtarget[qa] - data.qpos[qa]) - kd * data.qvel[da]
        if model.actuator_ctrllimited[i]:
            u = np.clip(u, *model.actuator_ctrlrange[i])
        data.ctrl[i] = u


def plan_toss(p_rel, p_catch, T):
    return (p_catch - p_rel) / T + 0.5 * np.array([0, 0, G]) * T


smooth = lambda s: s * s * (3 - 2 * s)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--left-body"); ap.add_argument("--right-body")
    ap.add_argument("--palm-offset", type=float, nargs=3, default=[0.08, 0, 0],
                    help="palm centre in the hand body's local frame")
    ap.add_argument("--normal-l", type=float, nargs=3, default=[0, 0, 1],
                    help="palm normal (local axis of the left hand body)")
    ap.add_argument("--normal-r", type=float, nargs=3, default=[0, 0, 1])
    ap.add_argument("--release", type=float, nargs=3, default=[0.25, 0.13, 0.22],
                    help="left palm release point, relative to base body (m)")
    ap.add_argument("--catch", type=float, nargs=3, default=[0.25, -0.13, 0.22],
                    help="right palm catch point, relative to base body (m)")
    ap.add_argument("--flight", type=float, default=0.35, help="flight time (s)")
    ap.add_argument("--sweep", type=float, default=0.22, help="left-hand acceleration distance (m)")
    ap.add_argument("--kp", type=float, default=100.0)  # only for torque-motor models
    ap.add_argument("--kd", type=float, default=5.0)
    ap.add_argument("--slowmo", type=float, default=1.0)
    ap.add_argument("--no-plate", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--hold", action="store_true",
                    help="diagnostic: keep apple attached to the left palm for 6 seconds")
    ap.add_argument("--assist-catch", action="store_true",
                    help="scripted catch when apple passes within --catch-radius of right palm")
    ap.add_argument("--catch-radius", type=float, default=0.12,
                    help="capture distance (m), only with --assist-catch")
    ap.add_argument("--video", help="render offscreen to this .mp4 (for Colab)")
    a = ap.parse_args()

    model = build_model(a)
    data, ik = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = data.xpos[1].copy()  # first body under world = pelvis/base

    L = Hand(model, "left", "l_palm", a.normal_l)
    R = Hand(model, "right", "r_palm", a.normal_r)

    p_rel, p_catch = base + a.release, base + a.catch
    v = plan_toss(p_rel, p_catch, a.flight)
    speed = np.linalg.norm(v)
    vhat = v / speed
    p_start = p_rel - a.sweep * vhat
    acc = speed ** 2 / (2 * a.sweep)
    t_sweep = speed / acc
    print(f"release velocity {np.round(v, 2)} m/s | hand accel {acc:.1f} m/s^2 | sweep {t_sweep:.2f}s")

    # settle IK to the start pose, copy into sim, put apple on the left palm
    ik.qpos[:] = model.qpos0
    L.ik(model, ik, p_start, 300)
    R.ik(model, ik, p_catch + [0, 0, 0.03], 300)
    mujoco.mj_kinematics(model, ik)
    data.qpos[:] = ik.qpos
    ad = model.jnt_qposadr[model.body_jntadr[model.body("apple").id]]
    vd = model.jnt_dofadr[model.body_jntadr[model.body("apple").id]]
    data.qpos[ad:ad + 7] = [*(L.pos(ik) + L.normal_w(ik) * (APPLE_R + 0.003)), 1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    t0, hold, t_ret = (1e6 if a.hold else 1.0), 0.15, 0.7
    t_rel = t0 + t_sweep
    t_end = 6.0 if a.hold else t_rel + 2.0
    st = {"pR": p_catch + [0, 0, 0.03], "land": None,
          "released": False, "caught": False, "closest": float("inf")}
    apple_geom_id = model.geom("apple_geom").id
    apple_contype = int(model.geom_contype[apple_geom_id])
    apple_conaffinity = int(model.geom_conaffinity[apple_geom_id])
    # Turn off contacts while the apple is gripped. Restore on release.
    model.geom_contype[apple_geom_id] = 0
    model.geom_conaffinity[apple_geom_id] = 0
    dt = model.opt.timestep

    def tilt_deg(hand, d):
        n = hand.normal_w(d)
        return np.degrees(np.arccos(np.clip(n[2], -1, 1)))

    print(f"IK-solved palm tilt from horizontal: left {tilt_deg(L, ik):.1f} deg, "
          f"right {tilt_deg(R, ik):.1f} deg (want ~0)")
    print(f"IK position error: left {np.linalg.norm(L.pos(ik) - p_start) * 100:.1f} cm, "
          f"right {np.linalg.norm(R.pos(ik) - (p_catch + [0, 0, 0.03])) * 100:.1f} cm (want ~0)")
    st["printed"] = False

    def attach_to(hand):
        # Scripted grip: place the apple just above the palm every timestep.
        # This is intentionally not a physical finger grasp.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        data.qpos[ad:ad + 3] = hand.pos(data) + hand.normal_w(data) * (APPLE_R + 0.004)
        data.qpos[ad + 3:ad + 7] = [1, 0, 0, 0]
        data.qvel[vd:vd + 6] = 0
        mujoco.mj_forward(model, data)

    def command(t):
        if a.hold or t < t_rel:
            attach_to(L)
        elif not st["released"]:
            # Enable normal physics at release: launch toward right palm.
            st["released"] = True
            model.geom_contype[apple_geom_id] = apple_contype
            model.geom_conaffinity[apple_geom_id] = apple_conaffinity
            mujoco.mj_forward(model, data)
            launch = data.qpos[ad:ad + 3].copy()
            destination = R.pos(data) + R.normal_w(data) * (APPLE_R + 0.004)
            launch_v = plan_toss(launch, destination, a.flight)
            data.qvel[vd:vd + 3] = launch_v
            data.qvel[vd + 3:vd + 6] = 0
            print(f"RELEASE t={t:.3f}s  velocity={np.round(launch_v, 3)} m/s")
            mujoco.mj_forward(model, data)
        elif st["caught"]:
            attach_to(R)
        elif a.assist_catch and t >= t_rel + 0.15:
            mujoco.mj_forward(model, data)
            center = R.pos(data) + R.normal_w(data) * (APPLE_R + 0.004)
            distance = np.linalg.norm(data.qpos[ad:ad + 3] - center)
            st["closest"] = min(st["closest"], distance)
            if distance <= a.catch_radius:
                st["caught"] = True
                model.geom_contype[apple_geom_id] = 0
                model.geom_conaffinity[apple_geom_id] = 0
                print(f"ASSISTED CATCH at t={t:.3f}s (distance {distance*100:.1f} cm)")
                attach_to(R)
        if t > 0.9 and not st["printed"]:
            st["printed"] = True
            mujoco.mj_forward(model, data)
            gap = data.qpos[ad + 2] - L.pos(data)[2]
            print(f"t=0.9s: actual left palm tilt {tilt_deg(L, data):.1f} deg, "
                  f"apple height above palm {gap * 100:.1f} cm (resting = {APPLE_R * 100:.1f})")
        # ---- left hand: hold, accelerate along v-hat, stop, return
        if t < t0:
            pl = p_start
        elif t < t_rel:
            pl = p_start + 0.5 * acc * (t - t0) ** 2 * vhat
        elif t < t_rel + hold:
            pl = p_rel
        else:
            pl = p_rel + smooth(min((t - t_rel - hold) / t_ret, 1.0)) * (p_start - p_rel)

        # ---- right hand: wait, track predicted landing point, cushion
        tgt = p_catch + [0, 0, 0.03]
        if not a.hold and st["released"] and not st["caught"] and t > t_rel + 0.05:
            pos, vel = data.qpos[ad:ad + 3], data.qvel[vd:vd + 3]
            zc = p_catch[2] + APPLE_R
            disc = vel[2] ** 2 + 2 * G * (pos[2] - zc)
            tl = (vel[2] + np.sqrt(disc)) / G if disc >= 0 else -1
            if tl > 0.04:
                st["land"] = (t + tl, pos[:2] + vel[:2] * tl)
            if st["land"]:
                tland, xy = st["land"]
                tgt[:2] = xy
                if t > tland - 0.03:
                    tgt[2] = p_catch[2] - 0.05 * (1 - np.exp(-(t - tland + 0.03) / 0.08))
        st["pR"] += (tgt - st["pR"]) * (1 - np.exp(-dt / 0.04))

        L.ik(model, ik, pl)
        R.ik(model, ik, st["pR"])
        apply_control(model, data, ik.qpos, a.kp, a.kd)

    def report():
        mujoco.mj_forward(model, data)
        if a.hold:
            target = L.pos(data) + L.normal_w(data) * (APPLE_R + 0.004)
            err = np.linalg.norm(data.qpos[ad:ad + 3] - target)
            print(f"HOLD TEST at t={data.time:.2f}s: "
                  f"apple-to-left-palm-center error {err * 100:.2f} cm")
            return
        cup = R.pos(data) + R.normal_w(data) * (APPLE_R + 0.004)
        err = np.linalg.norm(data.qpos[ad:ad + 3] - cup)
        label = "ASSISTED CATCH" if st["caught"] else "not settled on right palm"
        print(f"t={data.time:.2f}s  apple-to-right-palm distance: {err * 100:.1f} cm  ({label})")
        if a.assist_catch and not st["caught"]:
            print(f"Closest approach after 0.15s: {st['closest'] * 100:.1f} cm; "
                  f"try a larger --catch-radius or tune the arm trajectory")

    if a.video:  # offscreen render -> mp4 (works in Colab; set MUJOCO_GL=egl first)
        import mediapy as media
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = base + [0.3, 0, 0.2]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 150, -15
        opt = mujoco.MjvOption()
        opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        frames, fps = [], 60
        while data.time < t_end:
            command(data.time)
            mujoco.mj_step(model, data)
            if len(frames) < data.time * fps:
                renderer.update_scene(data, cam, opt)
                frames.append(renderer.render())
        media.write_video(a.video, frames, fps=fps / a.slowmo)
        print(f"wrote {a.video} ({len(frames)} frames)")
        report()
        return

    if a.headless:
        while data.time < t_end:
            command(data.time)
            mujoco.mj_step(model, data)
        report()
        return

    sync_every = max(1, int(1 / 60 / dt))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # shows palm axes: blue = z
        viewer.cam.lookat[:] = base + [0.3, 0, 0.2]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 150, -15
        n, reported = 0, False
        while viewer.is_running():
            tic = time.time()
            command(data.time)
            mujoco.mj_step(model, data)
            n += 1
            if n % sync_every == 0:
                viewer.sync()
            if data.time > t_end and not reported:
                report(); reported = True
            time.sleep(max(0.0, dt * a.slowmo - (time.time() - tic)))


if __name__ == "__main__":
    main()
