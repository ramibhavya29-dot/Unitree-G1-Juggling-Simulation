# Unitree G1 — Continuous One-Ball Juggling in MuJoCo

**ME206 Assignment 4 — supplementary source-code submission**

This project shows a Unitree G1 humanoid model performing a **continuous left-to-right and right-to-left one-ball juggling demonstration** in the MuJoCo simulator.

> **Scope / scientific accuracy:** The main demonstration uses arm inverse kinematics, motor commands, and a **scripted, gravity-shaped ball path** with smooth transitions between hands. Ball contact is disabled in this guided demo; catching and gripping are **not autonomous physical grasping**. The earlier tossing scripts are included for reference and experiments with simulated ball contact.

## Repository files

| File | Purpose |
| --- | --- |
| `apple_juggle.py` | **Main submission:** repeating, guided one-ball juggling simulation |
| `apple_toss.py` | Initial attempt at a physics-based single toss and catch |
| `apple_toss_fixed.py` | Intermediate toss experiment with assisted gripping/catching |
| `requirements.txt` | Minimum Python dependencies |
| `.gitignore` | Excludes temporary and recorded files |

## Model files (required external dependency)

The robot scene is from [Google DeepMind's MuJoCo Menagerie — Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1). It requires **`scene.xml`, `g1.xml`, and the referenced `assets/` mesh files**. The repository is set up to use the Menagerie folder alongside the Python scripts, rather than redistributing the model.

Expected directory structure **after setup**:

```text
Unitree-G1-Juggling-Submission/
├── apple_juggle.py
├── apple_toss.py
├── apple_toss_fixed.py
├── requirements.txt
├── README.md
└── mujoco_menagerie/                  # Download separately (see below)
    └── unitree_g1/
        ├── scene.xml
        ├── g1.xml
        └── assets/
```

## Requirements

- Windows 10/11, Linux, or macOS with a graphical desktop (Windows PowerShell commands below)
- Python 3.10 or newer; the project was developed with Python 3.14 on Windows
- MuJoCo Python package version 3.2 or later, NumPy
- Git (for retrieving the G1 model)
- A computer with working graphics drivers for the interactive MuJoCo viewer

## Setup — Windows PowerShell

1. Download or clone **this project repository**; open PowerShell inside its folder.

2. Optional but recommended: set up a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is restricted by PowerShell policy, skip the activation and use your existing `python` installation.

3. Install required Python packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Download the G1 simulation model (sparse clone downloads only the robot directory):

   ```powershell
   git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
   git -C mujoco_menagerie sparse-checkout set unitree_g1
   ```

   Alternatively, if you already have a `mujoco_menagerie` folder from an earlier installation, copy it into this project folder. Ensure `mujoco_menagerie/unitree_g1/scene.xml` exists.

5. Launch the **continuous juggling demo**:

   ```powershell
   python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --kp 500 --kd 25
   ```

   A MuJoCo viewer should open. Close the viewer or press Ctrl+C in PowerShell to stop.

## Useful run commands

**Slow motion (2× slower):**

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --kp 500 --kd 25 --slowmo 2
```

**Four guided transfers without GUI (diagnostics):**

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --headless --cycles 4
```

**Hold test:**

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --hold
```

**Earlier physics-based toss experiment:**

```powershell
python apple_toss.py --xml mujoco_menagerie/unitree_g1/scene.xml --hold --headless
```

## Video recording for the assignment

For a reproducible video, run the GUI simulation and capture the MuJoCo viewer using Windows Snipping Tool screen recording, Xbox Game Bar, or OBS Studio. Upload the resulting MP4 to YouTube and include the video link in the submission form.

There is an **optional** `--video` argument in `apple_juggle.py` that requires `mediapy` and a working FFmpeg installation; screen recording avoids common FFmpeg configuration problems on Windows.

## Implementation overview

1. Loads the Unitree G1 model with `mujoco.MjSpec` and fixes its floating base.
2. Adds sites representing the left and right palms and a red ball model.
3. Computes arm joint targets using damped least-squares inverse kinematics.
4. Uses a state machine with `carry → flight → settle` phases.
5. During `flight`, computes a gravity-shaped arc for a predetermined flight time; during `carry` and `settle`, the scripted ball is kept/blended near the palm.
6. Alternates throwing hands for repeated transfers.

Ball movement is scripted and is **not evidence of physically achieved robotic catching**. Timing, appearance, and pose accuracy depend on the selected G1 MJCF model and simulation environment.

## Troubleshooting

- **`No such file` / can't open `scene.xml`:** ensure the `mujoco_menagerie/unitree_g1` folder is downloaded and the PowerShell working directory is the project folder.
- **`ModuleNotFoundError: mujoco` / `numpy`:** install `requirements.txt` with the same Python interpreter used to launch the script.
- **Cannot type into PowerShell during viewer run:** close the viewer or press **Ctrl+C** before running another command.
- **Offscreen framebuffer resolution error:** the optional renderer is configured at 640 × 480; for higher resolution increase `model.vis.global_.offwidth/offheight` before constructing the renderer.
- **`ffmpeg` not found:** use screen recording, or install FFmpeg system-wide and verify `ffmpeg -version` works.

## Credits

- Unitree G1 robot model: [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) (upstream model has its own license).
- Physics/visualization engine: [MuJoCo](https://mujoco.org/).


## Group Member Information

- **Member 1:** Bhavya Rami
- **Member 1 ID:** 25110265
- **Member 2:** Tandel Smit
- **Member 2 ID:** 25110317
- **Member 3:** Rathod Prabhas
- **Member 3 ID:** 25110266
