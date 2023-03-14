# Multi PILOT: Learned Feasible Multiple Acquisition Trajectories for Dynamic MRI

This repository contains a PyTorch implementation of the paper:

[Multi PILOT: Learned Feasible Multiple Acquisition Trajectories for Dynamic MRI](https://arxiv.org/abs/2303.07150).

Tamir Shor (<tamir.shor@campus.technion.ac.il>), Tomer Weiss (<tomer196@gmail.com>), Dor Noti, Alex Bronstein

## Abstract

Dynamic Magnetic Resonance Imaging (MRI) is known to be a powerful and reliable technique for the dynamic imaging of internal organs and tissues, making it a leading diagnostic tool. A major difficulty in using MRI in this setting is the relatively long acquisition
time (and, hence, increased cost) required for imaging in high spatio-temporal resolution,
leading to the appearance of related motion artifacts and decrease in resolution. Compressed Sensing (CS) techniques have become a common tool to reduce MRI acquisition
time by subsampling images in the k-space according to some acquisition trajectory. Several studies have particularly focused on applying deep learning techniques to learn these
acquisition trajectories in order to attain better image reconstruction, rather than using
some predefined set of trajectories. To the best of our knowledge, learning acquisition
trajectories has been only explored in the context of static MRI. In this study, we consider
acquisition trajectory learning in the dynamic imaging setting. We design an end-to-end
pipeline for the joint optimization of multiple per-frame acquisition trajectories along with
a reconstruction neural network, and demonstrate improved image reconstruction quality
in shorter acquisition times.
This repo contains the codes to replicate our experiments.

## Dependencies

To install other requirements through `$ pip install -r requirements.txt`.

## Dataset
First you should download the OCMR dataset from [OCMR](https://ocmr.s3.amazonaws.com/data/ocmr_cine.tar.gz).

## Running Our Experiments

### Data Paths
All of our results can be recreated using the train.py script.
To run it, pass the path to your downloaded OCRM data using the '--data-path' argument. The OCMR dataset also contains a 'ocmr_data_attributes.csv' file - the path to this file must be passed using the '--ocmr-path' argument.\\
We recommend changing the default paths to your data paths to avoid passing these arguments within every run. To do that, change the defaults of these arguments in the 'create_arg_parser' function of the 'train.py' script.\\

### Augmentations
While good results can be achieved by training directly over the original dataset, our experiments from the paper have all been conducted using the augmented dataset. To augment your dataset:
1. Run the augment.py script. Make sure to pass the correct data paths as parameters. You can also control the inflation rate. The rate used in our experiments is the default. \\
2. After running the augmentation script, to use the train.py script with the augmented output, change the --data-path argument (in 'train.py') to the folder containing your augmented data.
3. With every run of the train.py script, pass in the '--augment' parameter (you can set it to be true by default in the 'create_arg_parser' function of train.py to avoid passing it in every run).

### Reconstruction Resets and Trajectory Freezing
By default, training is done without reconstruction resets or trajectory freezing. To use any of them, pass in the relevant flags - 'recons_resets' and 'traj_freeze' respectively.

### Using PILOT Baseline
The '--multi_traj' flag determines whether to learn independent per-frame trajectories (if true, this would be the MultiPILOT case) or a single trajectory shared across all frames (if false, this would be the PILOT baseline case). This flag is true by default. Change its default value to False to train the PILOT baseline.

### Using GAR Baseline
All of our experiments (beside GAR) had been conducted using radial initialization. This is the default initialization. To run the GAR baseline pass in 'golden' as the '--initialization' parameter.
Uniform, cartesian, gaussian and sprial initializations are also supported and can be used using this parameter.

### Shot Experiments
To recreate our experiments with the number of shots used, use the '--n-shots' parameter. 

### Quality Measures
Our research used PSNR, VIF and FSIM as quality ,measures. By default, VIF and FSIM aren't calculated because their computations times are relatively costly. We advise training the model without computing the FSIM and VIF values, and only computing them for evaluations of a trained model.
To compute VIF pass in the '--vif' flag. To compute FSIM pass in the '--fsim' flag.


## Citing this Work
Please cite our work if you find this approach useful in your research:
```latex
@ARTICLE{shor2023multipilot,
       author = {{Shor}, Tamir and {Weiss}, Tomer and {Noti}, Dor and {Bronstein}, Alex},
       title = "{Multi PILOT: Learned Feasible Multiple Acquisition Trajectories for Dynamic MRI}",
       journal = {arXiv e-prints},
       year = "2023",
       archivePrefix = {arXiv},
       eprint = {2303.07150},
}
```
