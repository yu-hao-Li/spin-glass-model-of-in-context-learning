# Spin glass model of in-context learning

Code for the paper *[Spin glass model of in-context learning](https://arxiv.org/abs/2408.02288)* (arxiv: 2408.02288)

We study a simple yet expressive transformer with linear attention and map this structure to a spin glass model with real-valued spins, where the couplings and fields explain the intrinsic disorder in data. Our theory reveals that for instance learning, increasing the task diversity leads to the emergence of in-context learning, by allowing the Boltzmann distribution to converge to a unique correct solution of weight parameters. Therefore the pre-trained transformer displays a prediction power in a novel prompt setting. The proposed analytically tractable model thus offers a promising avenue for thinking about how to interpret many intriguing but puzzling properties of large language models.

**Instructions**

The `src` folder contains the basic code that all figures share, including models, data sets, common functions, and so on

The code used to generate the data and plot for each figure in the paper, is placed in the folder corresponding to the figure number.

Some data files named `fig-1/data/J1.npy` and `fig-1/data/J2.npy` are not included in the repository, because they're too big. Therefore, it is necessary to run `data.py` to regenerate the data before running `plot.py` in `fig-1`

**Requirements**

- python 3.12.9
- numpy 2.2.3
- pytorch 2.6.0 with cuda 12.4

Other versions may also work, but have not been checked by the author.

**Contact**

If you have any question, please contact me via Yu-Hao.Li@outlook.com.

**Citation**

This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.