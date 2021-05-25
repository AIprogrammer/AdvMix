# AdvMix
Official code for our CVPR 2021 paper: ["When Human Pose Estimation Meets Robustness: Adversarial Algorithms and Benchmarks"](https://arxiv.org/abs/2105.06152).

## Getting started
* Installation
```
git clone https://github.com/AIprogrammer/AdvMix
cd AdvMix
pip install -r requirements
```

* Download the datasets [COCO](https://cocodataset.org/), [MPII](http://human-pose.mpi-inf.mpg.de/), and [OCHuman](https://github.com/liruilong940607/OCHumanApi). Put them under "./data". 

## Benchmarking
### Contruct benchmarking datasets
```
python make_datasets.py --data_name which_dataset_to_processs --data_dir where_is_the_dataset --save_dir directory_to_save_corrupted data
```
### Visualization examples
![benchmark_dataset](./figures/image_corruption.png)
### Benchmark results


## AdvMix
![AdvMix](./figures/AdvMix.jpg)
### Quantitative results


### Visualization results
![AdvMix](./figures/Qualitative.png)
Qualitative comparisons between HRNet without and with AdvMix. For each image triplet, the images from left to right are ground truth, predicted results of Standard HRNet-W32, and predicted results of HRNet-W32 with AdvMix.

# Citations
If you find our work useful in your research, please consider citing:
```
@article{wang2021human,
  title={When Human Pose Estimation Meets Robustness: Adversarial Algorithms and Benchmarks},
  author={Wang, Jiahang and Jin, Sheng and Liu, Wentao and Liu, Weizhong and Qian, Chen and Luo, Ping},
  journal={arXiv preprint arXiv:2105.06152},
  year={2021}
}
```

# License
Our research code is released under the MIT license. See LICENSE for details.

# Acknowledgments


