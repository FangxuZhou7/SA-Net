# SA-Net
Inspired by finding correspondences manually, which uses the topological relationship of image contents, we developed an attention-based neural network method for serial EM image registration to improve the registration accuracy.  More detail can be found in the following paper:

Human Attention-inspired Volume Reconstruction Method on Serial Section Electron Microscopy Images

## Table of Contents

- [Dependencies](#Dependencies)
- [Instructions for Use](#Instructions-for-Use)
- [Examples and Comparison Results](#Examples-and-Comparison-Results)
- [Contributing](#Contributing)

## Dependencies

Our method was trained on the Pytorch deep learning framework. 
The required libraries are as follows. 

python3.7, numpy, torch, torchvision, opencv

If you don't have some of these libraries, you can install them using pip or another package manager.

## Instructions for Use

If you just want to test our method, you can use "./main.py".


## Examples and Comparison Results

Here are some examples of aligning serial EM images using different image registration algorithms. And intuitively, our method has also achieved the best results

![Denoising results](https://github.com/VictorCSheng/VSID-Net/raw/main/paper_image/results.png)

## Contributing
Please refer to the paper "Human Attention-inspired Volume Reconstruction Method on Serial Section Electron Microscopy Images".
