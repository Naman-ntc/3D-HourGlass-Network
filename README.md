# 3D-HourGlass
3D HourGlass Networks for multitask training of Human Joint Locations and Human Activity Recognition

testScript Instructions `python testScript.py -imageFolder ...`


=================================================================================
nStacks | nChannels | numReductions | Time (per frame)  | Memory (per 65 frames)|
--------|-----------|---------------|-------------------|-----------------------|
2		|		256 |		4		|		41 ms 		|		~12 GB			|
2		|		128 |		4		|		24 ms 		|		~8 GB			|
2		|		064 |		4		|		17 ms 		|		~6.3 GB			|
1		|		128 |		4		|		19 ms 		|		~6.7 GB			|
1		|		256 |		4		|		2.8 ms 		|		~8.7 GB			|
=================================================================================