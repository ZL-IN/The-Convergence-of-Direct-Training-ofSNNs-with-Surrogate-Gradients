# Third-party notices

This release contains independently authored implementation files. It does not vendor the
anonymous supplementary framework, PyTorch internals, spiking model repositories, or dataset
archives. Dependencies are installed separately and retain their own licenses.

- PyTorch: https://github.com/pytorch/pytorch
- NumPy: https://github.com/numpy/numpy
- torchvision: https://github.com/pytorch/vision (optional dataset adapters)
- h5py: https://github.com/h5py/h5py (optional SHD reader)
- SpikingJelly: https://github.com/fangwei123456/spikingjelly (optional event dataset APIs)
- Matplotlib: https://github.com/matplotlib/matplotlib (optional plotting)

Research provenance:

- The direct-SNN convergence paper: https://openreview.net/forum?id=QNkfQ8wsii
- Mukhoty et al., Direct Training of SNN using Local Zeroth Order Method (NeurIPS 2023):
  https://proceedings.neurips.cc/paper_files/paper/2023
- Fang et al., Deep Residual Learning in Spiking Neural Networks (NeurIPS 2021):
  https://proceedings.neurips.cc/paper/2021
- The anonymous experiments used the memory-optimization framework associated with Huang
  et al., Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient
  Checkpointing and Spike Compression (ICLR 2026):
  https://github.com/AllenYolk/snn-gradient-checkpointing

The last framework is acknowledged for provenance, not bundled or relicensed. Architectural
and procedural differences are listed in docs/PAPER_ALIGNMENT.md. No compatibility or
endorsement by the upstream authors is implied.
