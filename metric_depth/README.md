# Depth Anything for Metric Depth Estimation

Our Depth Anything models primarily focus on robust *relative* depth estimation. To achieve *metric* depth estimation, we follow ZoeDepth to fine-tune from our Depth Anything pre-trained encoder with metric depth information from NYUv2 or KITTI.


## Performance

### *In-domain* metric depth estimation

#### NYUv2

| Method | $\delta_1 \uparrow$ | $\delta_2 \uparrow$ | $\delta_3 \uparrow$ | AbsRel $\downarrow$ | RMSE $\downarrow$ | log10 $\downarrow$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ZoeDepth | 0.951 | 0.994 | 0.999 | 0.077 | 0.282 | 0.033 |
| Depth Anything | **0.984** | **0.998** | **1.000** | **0.056** | **0.206** | **0.024** |


#### KITTI

| Method | $\delta_1 \uparrow$ | $\delta_2 \uparrow$ | $\delta_3 \uparrow$ | AbsRel $\downarrow$ | RMSE $\downarrow$ | log10 $\downarrow$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ZoeDepth | 0.971 | 0.996 | 0.999 | 0.054 | 2.281 | 0.082 |
| Depth Anything | **0.982** | **0.998** | **1.000** | **0.046** | **1.896** | **0.069** |


### *Zero-shot* metric depth estimation

Indoor: NYUv2 $\rightarrow$ SUN RGB-D, iBims-1, and HyperSim<br>
Outdoor: KITTI $\rightarrow$ Virtual KITTI 2 and DIODE Outdoor


| Method | SUN || iBims || HyperSim || vKITTI || DIODE Outdoor ||
|-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ | AbsRel | $\delta_1$ |
| ZoeDepth | 0.520 | 0.545 | 0.169 | 0.656 | 0.407 | 0.302 | 0.106 | 0.844 | 0.814 | 0.237 |
| Depth Anything | **0.500** | **0.660** | **0.150** | **0.714** | **0.363** | **0.361** | **0.085** | **0.913** | **0.794** | **0.288** |




## Pre-trained metric depth estimation models

We provide [two pre-trained models](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth), one for *indoor* metric depth estimation trained on NYUv2, and the other for *outdoor* metric depth estimation trained on KITTI. 

## Installation

```bash
conda env create -n depth_anything_metric --file environment.yml
conda activate depth_anything_metric
```

Please follow [ZoeDepth](https://github.com/isl-org/ZoeDepth) to prepare the training and test datasets.

## Evaluation

Make sure you have downloaded our pre-trained metric-depth models [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth) (for evaluation) and pre-trained relative-depth model [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) (for initializing the encoder) and put them under the ``checkpoints`` directory.

Indoor:
```bash
python evaluate.py -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d <nyu | sunrgbd | ibims | hypersim_test>
```

Outdoor:
```bash
python evaluate.py -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d <kitti | vkitti2 | diode_outdoor>
```

## Training

Please first download our Depth Anything pre-trained model [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth), and put it under the ``checkpoints`` directory.

```bash
python train_mono.py -m zoedepth -d <nyu | kitti> --pretrained_resource=""
```

Ours: 
```bash
CUDA_VISIBLE_DEVICES=4,6 python train_mono.py -m zoedepth -d sn --pretrained_resource=""
```



## bxy version
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train_mono.py -m zoedepth 
-d sn --pretrained_resource="" > logs/base.log 2>&1 &
```
此版本每次保存最小rmse模型，并以此作为后缀命名。预计每个epoch排除eval时间只有8min（4卡）。rmse要降到2.5e-3以下才算好。

疑点: zoedepth（比depth any thing要差的方法）在SNdataset zero-shot上的rmse2.6e-3，但是anything 现在finetune刚刚好于这个数值（目前最好结果是2.47e-3）？之前跑过video-depth-anything zero-shot，可视化要比zoe-depth好，但是数值结果还是差。很奇怪。

设置相关：batch size(bs)和pred image size(img_size)的设置在/Depth-Anything-bxy/metric_depth/zoedepth/models/zoedepth/config_zoedepth.json，需要注意1. bs似乎是卡数*每张卡batch数。 2. 虽然在SNDataset里已经设置了image的输入尺寸，但是depth的输出尺寸在这个文件设置，代码会自适应把img_size设置为满足网络尺寸，这里我还不清楚是通过插值还是什么方式实现的。


TODO 
0. 在https://github.com/SoccerNet/sn-depth/tree/main下载NBA2K22的数据一起用于训练（比赛允许任何公开数据集）
1. 增加数据增广（群里发的关于颜色的增广）。 
2. 调整超参数，例如观察收敛情况是否需要增加epoch数。 
3. 目前我只做了ddp下的validate，需要再写一版离线的load模型、不用ddp的validate, 实现参照trainerl383 eval_batch 函数，dataset使用data_mono里的l148行mode=='test'（我对data_mono.py做了较大的修改。 
4. 实现原来kitti数据集的裁剪增广方法，这个我怀疑对depth估计有问题？所以把这点放后面做。注意图像和depth要同步裁剪。
5. 试试换成kitti finetune过的 
6. 有时间再顺手做个可视化吧，depth的可视化参照SNdataset里的colorize函数。
7. 可以适当调高num_workers，现在每张卡num_workers=4，但可能没提升空间了。

This will automatically use our Depth Anything pre-trained ViT-L encoder.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      journal={arXiv:2401.10891},
      year={2024},
}
```
