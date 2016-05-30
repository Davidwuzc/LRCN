# LRCN
 Long-term Recurrent Convolutional Networks (LRCN)
 [Paper](http://arxiv.org/pdf/1411.4389.pdf)


## Dataset
1. Please download Movie Dataset from http://crcv.ucf.edu/data/UCF101.php
2. Unzip the file in LRCN directory
3. If you cannot open the video data in your PC, please try ```./converter.sh``` to convert file.
4. Make 'images_seq' directory and try ```python movie2image_seq.py``` to make dataset, which convert each movie to sequencial 16 frame images and save them in folders.

## Preparation of Model
1. Please download AlexNet Caffe model from https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
2. Please try ```python caffe2chainer_single.py``` and ```python caffe2chainer_hybrid.py``` to generate the hybrid AlexNet chainer models (We use tuned alexnet model before LSTM.).

## Run
```python lrcn_single.py``` -> Only CNN
```python lrcn_hybrid.py``` -> LRCN(CNN + LSTM) 
