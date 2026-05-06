# Industrial Anomaly Detection Research Code

This repository contains code for an ongoing research project on industrial anomaly detection and segmentation.

Since the paper has not yet been published, some core implementation details are intentionally omitted from this public repository.

---

## Datasets

### Dataset Structure

Please organize the datasets as follows:

```text
datasets/
├── mvtec/
│   ├── bottle/
│   ├── cable/
│   ├── ..
│   └── foreground_mask/
├── visa/
│   ├── candle/
│   ├── capsules/
│   ├── ..
│   └── foreground_mask/
├── btad/
│   ├── 01/
│   ├── ..
│   └── foreground_mask/
├── dtd/
└── mad_man/
```

### Benchmark Datasets
- MVTec-AD: [LINK](https://drive.google.com/file/d/1e4A4cGJkCYD4KCD0GHutSleaJqHM5fNb/view?usp=drive_link)
- BTAD: [LINK](https://drive.google.com/file/d/1MXRqcY0yfbsOY59ZJ4p4rlmmDo4qS6dK/view?usp=drive_link)
- VisA: [LINK](https://drive.google.com/file/d/10r1moi4LW1DrlujY-1-aVYjVRcFJUSO_/view?usp=drive_link)

### Anomaly Source Dataset

- DTD: [LINK](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

### Additional Resources

The following resources are also available from the related repository: [LINK](https://github.com/cqylunlun/glass)

- Foreground Mask: [LINK](https://drive.google.com/file/d/1Fn84QCfMtgBGEDcmY44v97Ci8wwpABK8/view?usp=sharing/)
- MAD-man: [LINK](https://drive.google.com/file/d/1HJmw7hSmrS0NMxfAjDltF4cXlN5S96Iz/view?usp=sharing/)

---
## Usage

### Environment
The code was developed and tested in the following environment.

- OS: Ubuntu 18.04
- Python: 3.9.12
- PyTorch: 2.0.0+cu117
- CUDA: 11.7
- cuDNN: 8.5
- GPU: NVIDIA RTX A6000

### Docker Hub
Containers for implemetation: [LINK](https://hub.docker.com/repository/docker/hongjiho/iad/general)

### Implementation
The main code for each method is organized in the corresponding method folder. Additional details will be updated soon..
---

## Note

The full implementation may be released after publication, subject to approval.
