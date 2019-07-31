# Semi-parametric Object Synthesis

<p align="center">
  <img src="gifs/teaser_car_h.gif"/ alt="Rotating cars">
</p>

## Abstract

We present a new semi-parametric approach to synthesize novel views of an object from a single monocular image. First, we exploit man-made object symmetry and piece-wise planarity to integrate rich *a-priori* visual information into the novel viewpoint synthesis process. An *Image Completion Network* (ICN) then leverages 2.5D sketches rendered from a 3D CAD as guidance to generate a realistic image. In contrast to concurrent works, we do not rely solely on synthetic data but leverage instead existing datasets for 3D object detection to operate in a real-world scenario. Differently from competitors, our semi-parametric framework allows the handling of a wide range of 3D transformations. Thorough experimental analysis against state-of-the-art baselines shows the efficacy of our method both from a quantitative and a perceptive point of view.

## Paper

<p align="center">
 <table>
  <tr>
  <td align="center"><a href="https://arxiv.org/abs/1907.10634" target="_blank"><img src="imgs/paper_thumb.png" width="200px"/></a></td>
  </tr>
  <tr>
  <td><pre>  
@article{palazzi2019semi,
  title={Semi-parametric Object Synthesis},
  author={Palazzi, Andrea and Bergamini, Luca and Calderara, Simone and Cucchiara, Rita},
  journal={arXiv preprint arXiv:1907.10634},
  year={2019}
}
</pre></td>
  </tr>
</table> 
</p>



## Code

### Install

Run the following in a fresh Python 3.6 environment to install all dependencies:

```bash
pip install -r requirements.txt
```

Code was tested on Ubuntu linux only (16.04, 17.04).

### How to run

To run our demo code, you need to download the following:
- Pascal3D+ vehicles dataset (`.zip` file [here](https://drive.google.com/open?id=1tP0MNK-505d8OWoyIp267JhkIfJt7jh1))
- 3D CADs (`.zip` file [here](https://drive.google.com/open?id=1V5sysWzg-jVfY50cYZzjg6cgoIgzu8u0))
- Pre-trained weights (`.pth` file [here](https://drive.google.com/open?id=1rF5sz_kXMmcu7wK9e6_PLqTh59GGIbuq))

Extract both archives in a location of your choice `<data_root>`;  move there the pre-trained weights file too.

The entry point is [`run_rotate.py`](https://github.com/iccv19sub265/semiparametric/blob/master/run_rotate.py). The script expects as mandatory arguments the car dataset, pre-trained weights and CAD dir.

Therefore, it can be run as follows:
```bash
python run_rotate.py <data_root>/pascal_car <data_root>/weights.pth <data_root>/cad --device cpu
```

### Description and usage

If everything went well,, you should see a GUI like the following:

<p align="center">
  <img src="imgs/viewport.png"/ alt="Viewport" width="80%">
</p>

The GUI is composed of two windows: the *viewport* and the *output* one.

While the focus is on the *viewport*, keyboard can be used to move around the object in spherical coordinates. [Here](https://github.com/iccv19sub265/semiparametric/blob/master/help.txt) the full list of commands is provided. While you move, the *output* shows both Image Completion Network (ICN) inputs (2.5D sketches, appearance prior) and network prediction. Please refer to Sec.3 of the paper for details.

*Notice*: it may happen that when starting the program, open3D does not render anything. This is an initialization issue. In case this happens, just focus on the *viewport* and press spacebar a couple of times until you see both windows rendered properly.

## Supplementary Material

### Other classes

<p align="center">
  <img src="gifs/teaser_chair.gif"/ alt="Rotating cars">
</p>

### Extreme viewpoint transformations (see Sec. 4)

Due to its *semi-parametric* nature, our method is much more robust than competitors to extreme viewpoint changes.

Here they are some examples:

<p align="center">
  <img src="gifs/gif_zoom.gif"/ alt="Zoom gif" width="40%">
  </br> Manipulation of radial distance.
</p>

<p align="center">
  <img src="gifs/gif_elevation.gif"/ alt="Elevation gif" width="40%">
  </br> Manipulation of elevation.
</p>

<p align="center">
  <img src="gifs/gif_pickup.gif"/ alt="Rototranslation gif" width="40%">
  </br> Arbitrary rototranslation.
</p>

### Data augmentation (see Sec. 4.4)

Additional examples generated synthetically using our model are shown below.

Each row is generated as follows. Given an image from [Pascal3D+](http://cvgl.stanford.edu/projects/pascal3d.html), other examples in the same pose are randomly sampled from the dataset. Then, our method is used to transfer the appearance of the latter to the pose of the first. Eventually, generated vehicles are stiched upon the original image. For a seamless collaging, we perform a small Gaussian blur at the mask border.

<p align="center">
  <img src="imgs/aug_data_supp.jpg"/ alt="Generated data" width="80%">
</p>

Percentage of Correct Keypoints (PCK) logged in TensorBoard during training (see Sec. 4.4)
<p align="center">
  <img src="imgs/pck_graph.png"/ alt="PCK graph" width="80%">
</p>


