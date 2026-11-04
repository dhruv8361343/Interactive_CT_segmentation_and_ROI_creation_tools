# CT Segmentation Toolkit

The CT Segmentation Toolkit (ctseg) investigates and delivers reusable software tools for manual and semi-automatic segmentation of 2D slices and 3D computed tomography (CT) volumes. Rather than locking algorithms inside a monolithic demo window, the project is architected as a modular Python package (src/ctseg/) so downstream CT analysis applications can import the data model, volumetric operations, and individual visualization widgets directly.




## Install

Python 3.10–3.13 is supported by the package metadata. Create an isolated environment first.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all,dev]"
```



- Core TIFF editing: NumPy and `tifffile`.
- Faster N-D morphology: SciPy (a pure NumPy fallback is included).
- Medical image formats: SimpleITK.
- Desktop interface and 3D rendering: PyQt6 and VTK.
- GPU: 3D rendering uses VTK/OpenGL, and the status strip under the 3D view names the device actually in use (green = NVIDIA, amber = another GPU). On a hybrid-graphics laptop Windows may pick the integrated Intel GPU; run `ctseg --prefer-nvidia-gpu` once (a per-user Windows Graphics setting, no admin rights) and restart to render on the RTX GPU. Undo it with `ctseg --reset-gpu-preference`. Optionally install CuPy with `.\.venv\Scripts\python.exe -m pip install -e ".[gpu]"` to run whole-volume 3D morphology on CUDA (verified bit-identical to the CPU path and about 4–8× faster on an RTX 4060). Thresholding, label masks, and boolean operations stay on the CPU because they are memory-bound and measured faster there; connected-component tracing also stays on the memory-bounded CPU path.
- Tests/development: pytest and Ruff.

## Pre print article
https://drive.google.com/file/d/1a5mlUe76LFBY9Ge_JB8-f3C80cBobLUb/view?usp=sharing

## Run

Open the supplied TIFF slice directory:

```powershell
.\.venv\Scripts\ctseg.exe folder name
```

Open one TIFF, NIfTI, or NRRD file:

```powershell
.\.venv\Scripts\ctseg.exe file path
```

Inspect a TIFF file/directory without loading pixels:

```powershell
.\.venv\Scripts\ctseg.exe --inspect folder name
```

Use `File → Open TIFF slice directory` or `File → Open TIFF/image file` in the GUI. Loading runs in the background; the current image remains usable until the new stack is validated. Navigate with the wheel over any slice to update the synchronized crosshair. The toolbox provides round/square brush and erase, rectangle/ellipse/freehand/polygon ROI tools, assisted operations, undo/redo, labels, ROI definitions, and slice propagation.

Use **File → Import labelmap into current image** to load a TIFF, NRRD, or NIfTI mask as a segmentation overlay. The import checks dimensions and, when available, physical geometry before replacing the current labelmap. **File → Export ROI definitions** and **Import ROI definitions** preserve editable ROI contours as JSON. The ROI manager lets you select an ROI to center the views, rename it, or remove its vector definition without deleting its rasterized label voxels.

For a whole-volume assisted segmentation, select **Whole volume** in the Assisted tools scope. Choose **Threshold preview** and set calibrated lower/upper limits, or choose **Seeded region grow** / **Flood fill**, set limits and connectivity, then click a seed on any orthogonal slice. The candidate is computed across the 3D stack, previewed in all slice views, and committed with one undo step when you choose **Accept preview**. Whole-volume thresholding and morphology use disk-backed/chunked processing for the supplied large stack. Island removal is currently limited to the selected slice.

The 3D pane displays committed segmentation surfaces for the complete label volume. **Segmentation** is enabled by default; use **Refresh** to rebuild surfaces from the current full-volume labelmap, and adjust their opacity independently from the CT volume. Because a labelled structure usually sits *inside* the sample, the **CT** selector defaults to **Auto**: while segmentation surfaces exist the CT is drawn as a transparent "ghost" so every segment stays visible through the whole stack. Choose **Full** for an opaque CT, **Ghost** to force the transparent CT, or **Hidden** to see only the segmentation. Surface generation runs asynchronously and the status strip reports how many segment surfaces are displayed.

To edit directly in 3D, select the active label, enable **3D brush**, choose **Paint** or **Erase**, select a round or square brush, then click the rendered volume or a visible label surface. The surface preview preserves labels smaller than its display downsampling stride. After each click the status bar reports the picked voxel or explains why the click did not produce an edit. Turn the toggle off to return to camera interaction.

The window fits the screen: axial is the large pane with coronal and sagittal below it, all dividers are draggable, and the tools dock scrolls instead of growing the window.

After painting or applying a mask on any orthogonal slice, the crosshair moves to an edited voxel so the same segmentation intersection is visible in the other two slice views. The 3D label surfaces rebuild in a debounced background task; newly painted segments may take a short moment to appear. The histogram in the tools dock shows sampled calibrated intensities; drag across it to set the assisted threshold interval, or click a segment swatch to select that label and double-click it to change its color.

## Project layout

```text
src/ctseg/
  core/         Geometry, image/label data, command history, algorithms
  io/           TIFF first, plus SimpleITK image readers and writers
  ui/           PyQt6 slice/label/tool/ROI/histogram controls and VTK 3D view
  app.py        CLI and standalone developer shell entry point
tests/          Geometry, algorithm, command, TIFF, and optional UI tests
examples/       Headless import, environment check, host embedding, profiling
docs/           Objective traceability, phase reports, public interface, phase plan
```

## Phase implementation map

| Phase | Implemented subsystem | Verification focus |
|---|---|---|
| 0 | Package, runtime dependencies, TIFF inspection, off-screen/UI feasibility hooks | TIFF metadata inventory, imports, synthetic image |
| 1 | `CTVolume`, `VolumeGeometry`, disk-backed TIFF stack import, orthogonal viewers | Slice ordering, axes, anisotropic geometry |
| 2 | Round/square brush/erase, polygon rasterization, rectangle/ellipse/freehand ROI, bounded commands | Exact masks, boundaries, one undo per stroke |
| 3 | Threshold previews, connected growth, flood fill, morphology, 2D/3D components, label merge/erase | Known-answer arrays; cancel/accept preview |
| 4 | Label metadata and bounded command history | Bit-exact undo/redo and lock enforcement |
| 5 | Nearest-slice and signed-distance interpolation, preview controls | Synthetic keyframes, one-command propagation |
| 6 | 3D volume renderer, asynchronous label surfaces, 3D paint/erase, cross-view recentering | Coordinate footprint and 2D/3D synchronization |
| 7 | Operation profiling example and optional optimization boundary | Reproducible timing and unchanged results |
| 8 | TIFF/JSON/NRRD/NIfTI/raw I/O, package API, host example | Geometry and data round trips |

