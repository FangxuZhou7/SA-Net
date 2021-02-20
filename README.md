# SA-Net
Analyzing 2D EM image sequences of biological tissues in a 3D context is necessary to restore the original 3D integrity destroyed by physical sectioned by aligning the serial 2D images. However, similar texture intra-section and complex variations of serial EM images intersections make it challenging to find the correct correspondences. Inspired by finding correspondences manually, which uses the topological relationship of image contents, we developed an attention-based neural network method for serial EM image registration to improve the registration accuracy.  More detail can be found in the following paper:

Human Attention-inspired Volume Reconstruction Method on Serial Section Electron Microscopy Images

![3D Aligning](https://github.com/FangxuZhou7/SA-Net/blob/main/show-img/3D%20volume%20reconstruction%20and%20inspired.png)

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
The detailed usage and code annotations will be added later.


## Examples and Comparison Results

Here are some examples of aligning serial EM images using different image registration algorithms. And intuitively, our method has also achieved the best results

![Aligning results](https://github.com/FangxuZhou7/SA-Net/blob/main/show-img/res1.png)
![Aligning results](https://github.com/FangxuZhou7/SA-Net/blob/main/show-img/res.png)

## Contributing
Please refer to the paper "Human Attention-inspired Volume Reconstruction Method on Serial Section Electron Microscopy Images".
