# 3D-HourGlass-Network
3D CNN Based Hourglass Network for Human Pose Estimation (3D Human Pose) from videos. This was my summer'18 research project.

## Discussion
In this work I try to extend the idea in [Carriera et. al. CVPR'17](https://arxiv.org/pdf/1705.07750.pdf) of 3D CNN inflation for action recognition from videos to human pose estimation from videos. We use a pretrained hourglass network with a fully connected depth regressor, inflate the 2D convolutions to 3D convolutions and perform temporal 3D human pose estimation. This inflation helps the network learn features from nearby frames and refine its predictions. Similar idea was used in [Girdhar et. al. CVPR'18](https://arxiv.org/pdf/1712.09184.pdf)  (at about the same time!) where they perform multiperson human pose estimartion from videos using an `inflated` Mask RCNN

## Requirements

* python 3.6
* pytorch 0.4
* torchvision
* progress

## Datasets
We used [Human 3.6](http://vision.imar.ro/human3.6m/) dataset for this project.

## Instructions to run
`python main.py -expID [EXP-NAME] -nFramesReg [NUM-FRAMES]`

## Results
We improved the baseline performance of hourglass network from MPJPE of 64 to MPJPE 62.8 and thus show significance of temporal features in real world problems. This idea could be easily extended for other tasks also like semantic segmentation and object detection.
