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
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   ├── zipper/
│   └── foreground_mask/
├── btad/
│   ├── 01/
│   ├── 02/
│   ├── 03/
│   └── foreground_mask/
├── visa/
│   ├── candle/
│   ├── capsules/
│   ├── ...
│   └── foreground_mask/
├── dtd/
└── mad_man/
```

### Benchmark Datasets
- MVTec-AD: [LINK](https://drive.google.com/file/d/1e4A4cGJkCYD4KCD0GHutSleaJqHM5fNb/view?usp=drive_link)
- BTAD: [LINK]([https://drive.google.com/file/d/1MXRqcY0yfbsOY59ZJ4p4rlmmDo4qS6dK/view?usp=drive_link](http://avires.dimi.uniud.it/papers/btad/btad.zip))
- VisA: [LINK]([https://drive.google.com/file/d/10r1moi4LW1DrlujY-1-aVYjVRcFJUSO_/view?usp=drive_link](https://github.com/amazon-science/spot-diff))

### Anomaly Source Dataset

- DTD: [LINK]([https://www.robots.ox.ac.uk/~vgg/data/dtd/](https://www.robots.ox.ac.uk/~vgg/data/dtd/))

### Additional Resources

The following resources are also available from the related repository: [LINK](https://github.com/cqylunlun/glass)

- Foreground Mask: [LINK](https://drive.google.com/file/d/1Fn84QCfMtgBGEDcmY44v97Ci8wwpABK8/view?usp=sharing/)
- MAD-man: [LINK](https://drive.google.com/file/d/1HJmw7hSmrS0NMxfAjDltF4cXlN5S96Iz/view?usp=sharing/)

---

## Note

The full implementation may be released after publication, subject to approval.
