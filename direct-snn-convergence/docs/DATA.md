# Data and split conventions

Datasets are not redistributed. Obtain them from their original sources and follow their
access conditions. The MIT license of this implementation does not relicense datasets.

| Task | `--data-root` contents | Adapter and evaluation split |
|---|---|---|
| scifar10/scifar100 | torchvision CIFAR storage | Explicit `--download` allowed; official test set retained |
| dvsgesture | prepared SpikingJelly DVS128Gesture root | Official train/test split; frames integrated by event count |
| cifar10dvs | prepared SpikingJelly CIFAR10DVS root | Seeded stratified 80/20 pool/test split; frames integrated by event count |
| shd | shd_train.h5 and shd_test.h5 | Original HDF5 events; train/test retained |
| tinyimagenet | train/ and val/ in tiny-imagenet-200 | Official validation set is the final evaluation set |

For all tasks, 10% of each training-pool class is held out as validation, using the run's
seed and at least one example per class. Only this validation set selects best.pt. This
differs from legacy recipes that monitor the official test set during training.

Sequential CIFAR uses image width as time; height stays 32. Training uses horizontal flip
and normalization, without legacy mixup/cutmix. DVSGesture frames remain 128x128; CIFAR10-DVS
frames resize to 48x48 using nearest-neighbor interpolation. SHD bins events from the first
second into `steps` half-open bins and keeps channel zero; repeated events in a bin are binary.
TinyImageNet uses its official val_annotations.txt directly, without moving validation files
or changing the class ordering from the training directories.

Optional imports give the dataset provider's normal errors if data or dependencies are
missing. A failed dataset read never falls back to synthetic data. Test fixtures and the
`--synthetic` command are mechanics checks and are labeled as such.

Original sources:

- CIFAR: https://www.cs.toronto.edu/~kriz/cifar.html
- SHD: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
- Event dataset API: https://spikingjelly.readthedocs.io/en/0.0.0.0.14/spikingjelly.datasets.html
- TinyImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip
