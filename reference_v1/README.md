# Pinned frame-V1 reference snapshot

Copied without modifying the user's original CARLsim tree on 2026-10-28.

- `src/motion_energy.cu`: SHA256 `d869f55639618c39c41d9e7c7b980fb0701b247975ecad48484d2beef0eec488`.
- `inc/`: matching class header and CUDA error helpers.
- `LICENSE`: upstream MotionEnergy MIT license.

Use `examples/build_cpp_v1_reference.py` to compile this snapshot into a new
output directory. No CARLsim network is built. CUDA Toolkit and its sample
`helper_cuda.h` are required only for this independent reference executable,
not for the NumPy/PyTorch frontend.

This source intentionally preserves disabled scale smoothing (`scale > 4`),
disabled complex spatial blur, enabled edge compensation, and RGB red weight
0.1299. The event/OMS functions are present but the new Python frontend ports
only `calcV1complex()` for image frames.
