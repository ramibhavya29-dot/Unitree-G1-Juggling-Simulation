# Unitree G1 — Continuous One-Ball Juggling (MuJoCo)

**ME206 • Assignment 4 • Supporting simulation source and reproduction instructions**

This repository contains the Python source for a Unitree G1 humanoid robot performing alternating **left → right → left** one-ball juggling in the MuJoCo simulator. The primary entry point is **`apple_juggle.py`**. The earlier toss programs are retained as experimental/diagnostic versions.

> **What this demonstrates (important):** This is a **guided/simulated juggling visualization**, not an autonomous contact-based juggling controller. Arm poses are produced with damped least-squares inverse kinematics (IK) and motor commands. During the main demo, ball positions are commanded along a gravity-shaped trajectory and blended to the palm; ball–hand collision is disabled. Console `CATCH` messages indicate scripted phase transitions, **not verified physical catches**.

## Group members

Fill in the names and roll numbers below before submitting the repository link.

| No. | Name | Roll number |
| --- | --- | --- |
| 1 | _Member 1 name_ | _Roll number_ |
| 2 | _Member 2 name_ | _Roll number_ |
| 3 | _Member 3 name_ | _Roll number_ |

> Add or remove rows to match your actual group. Also enter **all** members in the course submission form, as required by the TA.

## 1. Files supplied

| File | Description |
| --- | --- |
| `apple_juggle.py` | **Main program**: repeating alternating single-ball transfers, interactive viewer, diagnostics, optional video export |
| `apple_toss.py` | Original experimental physics-driven toss/catch code |
| `apple_toss_fixed.py` | Intermediate guided-grip/assisted-catch experiment |
| `requirements.txt` | Required Python packages for the main simulation; optional video dependency described below |
| `.gitignore` | Excludes local environments, videos, and temporary files |
| `README.md` | This guide |

**External robot assets:** The full Unitree G1 MuJoCo MJCF/mesh model belongs to [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1), and is **not bundled** here. You must obtain its `unitree_g1` directory including `scene.xml`, `g1.xml`, and `assets/` (plus any additional files referenced by the XML). Keep the upstream model files together so relative asset paths work. The model's upstream license applies.

## 2. System requirements

- Windows 10/11 with **PowerShell** (instructions below), or Linux/macOS with an equivalent terminal
- Python **3.10+**, with a Python/MuJoCo combination supported by your platform (original demonstration used Python 3.14 on Windows)
- Working graphics drivers/display for interactive viewer; headless mode does not show a window
- Internet access to download Python packages and the upstream G1 model
- `git` **only if** using the terminal-clone option; not required for the browser ZIP method

## 3. Download the project and model (Windows, no Git knowledge required)

1. On this GitHub repository, click **Code → Download ZIP**, extract it, then open the extracted folder containing `apple_juggle.py`.
2. Open [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and select **Code → Download ZIP**. Extract it.
3. From the extracted Menagerie folder, copy its **entire `unitree_g1` directory** into a new folder named **`mujoco_menagerie`** alongside `apple_juggle.py`. Do **not** copy `scene.xml` alone: the XML references other model and mesh assets.
4. Final folder layout (the G1 directory may contain additional files):

```text
Unitree-G1-Juggling-Simulation/
├── apple_juggle.py
├── apple_toss.py
├── apple_toss_fixed.py
├── README.md
├── requirements.txt
├── .gitignore
└── mujoco_menagerie/                 # downloaded separately
    └── unitree_g1/
        ├── scene.xml
        ├── g1.xml
        └── assets/
```

**Alternative if Git is installed** (run from the project folder; skip browser model download):

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set unitree_g1
```

If you already installed Menagerie earlier, you can instead pass the path to your existing `scene.xml` using `--xml`.

## 4. Install Python packages (PowerShell)

Open PowerShell in the project folder (File Explorer: open that folder, click address bar, type `powershell`, press Enter).

```powershell
python --version
python -m pip install -r requirements.txt
```

A virtual environment is optional. If desired:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then replace `python` in all subsequent commands with `.\.venv\Scripts\python.exe` when using the virtual environment.

If `python` is not recognized, install a supported Python version and reopen PowerShell. If installation fails, check that you use the same Python interpreter for installation and execution.

## 5. Run the main juggling demo

From the project folder, check the model exists:

```powershell
Test-Path .\mujoco_menagerie\unitree_g1\scene.xml
```

It should print `True`. Then run:

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --kp 500 --kd 25 --cycles 10
```

A MuJoCo window opens. The robot repeatedly transfers the ball from one hand to the other. `--cycles` is the number of **transfers** (not necessarily pairs). The GUI stays open until you close the viewer or interrupt PowerShell with **Ctrl+C**. The terminal logs `THROW` and `CATCH` events.

**Slow-motion viewer (2× slower):**

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --kp 500 --kd 25 --cycles 10 --slowmo 2
```

**Automated no-window diagnostic run (four transfers):**

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --kp 500 --kd 25 --headless --cycles 4
```

**Guided hold test (six simulated seconds):**

```powershell
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --hold --headless
```

Use `python apple_juggle.py --help` for every available flag (`--flight`, `--carry-time`, `--settle-time`, `--lift`, `--release`, `--catch`, hand body/site settings, etc.). The defaults were tuned for the referenced G1 model. Differences between model revisions may require retuning.

## 6. Run the earlier toss experiments (optional)

These are supplied for the TA to inspect the development history; they are **not the validated main juggling demonstration**.

```powershell
python apple_toss.py --xml mujoco_menagerie/unitree_g1/scene.xml --hold --headless
python apple_toss_fixed.py --xml mujoco_menagerie/unitree_g1/scene.xml --hold --headless
```

The original single-toss program was observed to let the ball drift/roll off its palm. Its inclusion is for transparency, not as a successful catch result.

## 7. Capture and submit a video

**Recommended for Windows:** run the GUI as in section 5 and record the MuJoCo viewer using OBS Studio or Windows screen recording. Save the recording as MP4, upload to YouTube, and submit the video URL in the course form.

**Optional script-based MP4 export:** install MediaPy separately and make sure FFmpeg is on your system `PATH`:

```powershell
python -m pip install mediapy
ffmpeg -version
python apple_juggle.py --xml mujoco_menagerie/unitree_g1/scene.xml --kp 500 --kd 25 --cycles 10 --video unitree_juggling.mp4
```

The included offscreen renderer uses **640 × 480** pixels to fit the default MuJoCo framebuffer. FFmpeg is an external executable, not provided by `requirements.txt`. If video encoding fails, use screen recording; the simulation itself can still run.

**Demonstration video:** add the uploaded YouTube URL here before handing in: `VIDEO_URL_TO_BE_ADDED`.

## 8. How the main simulation works

1. Uses `mujoco.MjSpec` to load the Unitree G1 MJCF model and remove the free base joint, fixing the robot in place.
2. Adds palm sites and a red sphere to the model.
3. Solves left/right arm joint targets with damped least-squares IK and sends actuator commands.
4. Uses a `carry → flight → settle` state machine, alternating the throwing hand after each cycle.
5. Computes a trajectory shaped by a constant-gravity ballistic equation; updates the ball position at each time step and blends it to the destination palm. Contact dynamics are deliberately disabled in the guided demo.
6. Displays the simulation interactively or reports transfer events in headless mode.

The code is the implementation record; the printed `CATCH` count alone is **not a physics-based catch-success metric**.

## 9. Troubleshooting

| Problem | Action |
| --- | --- |
| Missing `scene.xml` or other mesh assets | Confirm the **entire** upstream `unitree_g1` directory is present; use `Test-Path` above |
| `ModuleNotFoundError` | Run `python -m pip install -r requirements.txt` in the same Python environment used to launch the script |
| Error about `MjSpec` or compilation | Check the installed MuJoCo version and model path; try `python -m pip show mujoco` |
| PowerShell will not accept another command while viewer runs | Close the viewer or press **Ctrl+C** |
| Robot/ball motion differs from demo | Check model version, palm body names, and command-line geometry/timing parameters |
| `Image width > framebuffer width` | Use provided 640 × 480 offscreen renderer or adjust `model.vis.global_` framebuffer values before creating `Renderer` |
| `ffmpeg` not found when exporting video | Record the screen, or install FFmpeg and ensure `ffmpeg -version` works from a new PowerShell window |

## Credits and reproducibility notes

- [MuJoCo Menagerie — Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1): externally supplied robot model and associated assets; check its upstream licensing and attribution.
- [MuJoCo](https://mujoco.org/): simulator and Python API.

This repository provides the **project-specific Python code**, not a copy of the external G1 model. The installation instructions describe how to reproduce its required dependencies.
