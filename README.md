# Dual super-resolution learning for semantic segmentation

# 2020-10-31 Good News! I achieved an mIoU of 0.6787 in the newest experiment(the experiment is still running and the final mIoU may be even higher)!
- So the FA module should be places after each path's final output.
- The FTM should be 19 channel -> 3 channel
- Hyper-Parameter fine-tuning

It's amazing that the final model converges at a extremely fast speed. Now the codes are all ready and you don't have to worry about anything. Just clone this repo and run train.py!

And thanks for the reminder of @XinruiYuan, currently this repo also differs from the original paper in the architecture of SISR path. I will be working on it after finishing my homework.



---

# 2020-10-22 First commit

**I implemented the framework proposed in this paper since the authors' code is still under legal scan and i just can't wait to see the results. This repo is based on Deeplab v3+ and Cityscapes, and i still have problems about the FA module.**

- so the code is runnable? yes. just run train.py directly and you can see DSRL starts training.(of course change the dataset path. See insturctions in the Deeplab v3+ part below.)
- any difference from the paper's proposed method? Actually yes. It's mainly about the FA module. I tried several mothods such as:
    - 19 channel SSSR output -> feature transform module -> 3 channel output -> calculate FAloss with 3 channel SISR output. Result is like a disaster
    - 19 channel SSSR last_conv(see the code and you'll know what it is) feature -> feature transform module -> calculate FAloss with 19 channel SISR last_conv feature. still disaster.
    - 19 channel SSSR last_conv(see the code and you'll know what it is) feature -> feature transform module -> calculate FAloss with 19 channel SISR last_conv feature, but no more normalization in the FA module. Seems not bad, but still cannot surpass simple original Deeplab v3+
    - Besides, this project use a square input(default 512\*512) which is cropped from the original image.

- so my results? mIoU about 0.6669 when use the original Deeplab v3+. 0.6638 when i add the SISR path but no FA module. and about 0.62 after i added the FA module.

The result doesn't look good, but this may because of the differences of the FA module.(but why the mIoU decreased after i added the SISR path)

**Currently the code doesn't use normalization in FA module. If you want to try using them, please cancel the comment of line 16,18,23,25 in 'utils/fa_loss.py'**

Please imform me if you have any questions about the code.


below are discriptions about Deeplab v3+(from the original repo).

# pytorch-deeplab-xception

**Update on 2018/12/06. Provide model trained on VOC and SBD datasets.**  

**Update on 2018/11/24. Release newest version code, which fix some previous issues and also add support for new backbones and multi-gpu training. For previous code, please see in `previous` branch**  

### TODO
- [x] Support different backbones
- [x] Support VOC, SBD, Cityscapes and COCO datasets
- [x] Multi-GPU training



| Backbone  | train/eval os  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet    | 16/16          | 78.43%     | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |
| MobileNet | 16/16          | 70.81%     | [google drive](https://drive.google.com/open?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt) |
| DRN       | 16/16          | 78.87%     | [google drive](https://drive.google.com/open?id=131gZN_dKEXO79NknIQazPJ-4UmRrZAfI) |



### Introduction
This is a PyTorch(0.4.1) implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD and Cityscapes datasets.

![Results](doc/results.png)


### Installation
The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```
### Training
Follow steps below to train your model:

0. Configure your dataset path in [mypath.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/mypath.py).

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```

2. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone:
    ```Shell
    bash train_voc.sh
    ```
3. To train deeplabv3+ using COCO dataset and ResNet as backbone:
    ```Shell
    bash train_coco.sh
    ```    

### Acknowledgement
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
