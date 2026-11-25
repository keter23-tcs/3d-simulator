# 3D Simulator

**English** · [Deutsch](README.de.md)

Calculates the 3D frequency response of a box (bass horn, hybrid, bandpass, closed box …) from the SketchUp model, with finite elements on a
voxel grid of the air. Part of the **Raum Simulation / Stack Lab²³**: the app creates the job, the simulator calculates, the result goes back into the app.

## Installation (Windows) – one double click

1. Download this repository as a zip (**Code → Download ZIP**) and unzip it.
2. Double-click **`Start.bat`**.

On the first start the script sets up **everything in one go** (about 2–5 minutes, needs internet):

* Python is installed automatically if it is missing (through winget, for your user only),
* a private Python environment `.venv` with all packages (numpy, scipy, scikit-fem, pyamg),
* the folders **`Input`** and **`Output`**,
* with an NVIDIA graphics card, the question whether GPU acceleration should be installed too (1–2 GB, can also be done later in the menu).

After that the menu starts immediately every time. To only set things up without starting: `Installieren.bat`.

> If winget is missing, the Python download page opens: install it (tick **"Add python.exe to PATH"** at the bottom) and start `Start.bat` again.

## How to calculate a box

1. In the app: **Boxes → Measurement → Export 3D job** (the markers for the pressure and rear chamber must be set). The file
   `<boxname>.auftrag.json` lands in your Downloads folder; the model (.dae) is already inside it.
2. `Start.bat` → menu **`3`** (fetch jobs from the Downloads folder) → menu **`1`** (calculate all jobs).
3. The result is in **`Output`** as `<boxname>.box.json`.
4. In the app, in the horn simulation, click **Load 3D result** and choose this file.

Instead of menu `3` you can also put the job into the `Input` folder by hand.

## Menu

| Key | Function |
|---|---|
| `1` | Calculate all jobs in `Input` |
| `2` | Pick one job and calculate it |
| `3` | Fetch jobs from the Downloads folder into `Input` |
| `4` | Show the files in `Input` and `Output` |
| `5` | Open the `Input` and `Output` folders in the Explorer |
| `6` | Use the CPU only (switch the GPU off) |
| `7` | Move finished jobs to `Input\done`: ON/OFF (default ON) |
| `8` | Set up GPU acceleration (NVIDIA) |
| `0` | Quit |

## Duration and memory

Grid 2 cm: test run, about 30 s. Grid 1 cm: the real result, about 4 minutes and up to 5 GB of RAM
(times depend on the computer; faster with an NVIDIA GPU). You choose the grid in the app when you export.

## Without the menu (advanced)

```bat
.venv\Scripts\python rechner\rs3d.py Input\my-box.auftrag.json
```
`--cpu` forces the CPU. The job format is described at the top of `rechner/rs3d.py` (the JSON keys are German on purpose: that is the format the app writes).

## For developers

The calculation core (`rechner/*.py` except `menue_3d.py`) is taken over from the main project (`werkzeug/`):
run `py werkzeug/sync_3d.py` there, then copy this folder into the repository.
The user-facing texts of the menu and messages are English in this copy; comments and docstrings inside the calculation core are still German.
