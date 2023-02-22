# SIRN
SIRN is a network for Transmission Line dense-tiny object detection. 
This repo is the implementation of the paper ("SIRN: An Iterative Reasoning Network for
Transmission Lines Based on Scene Prior Knowledge").

## Requirements
+ Python3.8
+ Python packages
  + PyTorch >= 1.0
  + Torchvision >= 0.9.0
  + opencv-python-headless
  + fvcore
  + cloudpickle
  + omegaaconf
  + pycocotools
  + tidecv
  + fairscale
  + timm
  + scikit-learn


## Get Started
After successfully completing requirements, you can be ready to run the demo.

+ **Download** the test.pth which finally use in the paper(SIRN) from [Weights](链接：https://pan.baidu.com/s/1NSEzvPzPby5FNv6IwZKiJw (extract code:8qb9)

+ **Download**  the datasets from [Datasets](https://pan.baidu.com/s/1atcDoJ2pDaXF8uSZh_3kvg (extractcode:8lsq)

+ **Put**  datasets into the: 
```sh
{repo_root}/
```
+ **Put**  test.pth into the: 
```sh
{repo_root}/
```
+ **Using** this code to see the detection results in the Transmission line datasets:
```sh
python train_net.py --config-file ./model_configs/faster_r50_s600.yaml --eval-only MODEL.WEIGHTS test.pth
```

