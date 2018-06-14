# 3D-HourGlass
3D HourGlass Networks for multitask training of Human Joint Locations and Human Activity Recognition



## Instructions

python version : 3.6
pytorch 0.4

`python main.py -expID "NAME TO PUT FOR EXPERIMENT" -scheme INIT_SCHEME -nRegFrames REG_FRAMES -freezefac FACTOR -valIntervals HOW_MANY_EPOCHS_BEFORE_VAL -nFramesLoad HOW_MANY_TO_LOAD -regWeight DEPTH_WEIGHT -mult MULT_FACTOR


In the top REPO directory run `mkdir exp`

1.) expID forms a folder for current experiment with logs and trained models 
2.) scheme refers to intialization scheme, keep it either to 1 or 3 (hyperparameter)
3.) REG_FRAMES keep between 1-6
4.) Freeze the old 2D CNN network by FACTOR, (reduce LR for old network). Keep 0 or 0.01 or 0.05 or 0.1 or 0.5 or 1 
5.) valIntervals : validates after these many iterations and also saves models...
6.) nFramesLoad  Frames per video to consider furing training, IMPORTANT, keep integer multiple of REG_FRAMES
7.) regWeight weight of the second loss function, 0.05, 0.1 0.2 0.3 0.35
8.) mult factor for initialization 1, is SCHEME is 1 then only. keep 0.05 0.1 0.2 0.3 0.5 (may try more)

for learning rate use flags -LRhg -LRdr, defualt to 2e-5, feel free to change
