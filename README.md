# A Neural Builder for Spatial Subdivision Hierarchies
This repository contains the source code of the paper: *[A Neural Builder for Spatial Subdivision Hierarchies](https://doi.org/10.1007/s00371-023-02975-y)*, published in **The Visual Computer** and presented at the **[Computer Graphics International 2023](https://www.cgs-network.org/cgi23)** conference.

The code, written in Python, showcases how to train and test the network with both a default and your own model, in order to infer a fixed-level tree structure. The resulting fixed-depth tree can then be further expanded, in parallel, into either a full *k*-d tree or transformed into a bounding volume hierarchy, with any known conventional tree builder. For more details on the last part, please see the paper.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Downloading the Dataset](#downloading-the-dataset)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Building your Own Dataset](#building-your-own-dataset)
- [How to Cite](#how-to-cite)
- [Acknowledgments](#acknowledgments)
- [Contact Us](#contact-us)

## Prerequisites

The code has been tested on **Python 3.9** and requires the following packages to be present in your environment:
```python
trimesh
numpy==1.24.3
pandas==1.4.3
tensorflow==2.10.1
matplotlib==3.5.3
seaborn==0.11.2
```

## Downloading the Dataset
The dataset used both in the paper and this code is available *[here](https://cloud.aueb.gr/index.php/s/wH5Nt9wsHfn3r8c)* (1GB - 2.6GB extracted). Download and extract it at the root folder of the project.

Details:
- The dataset directory is at ``datasets/custom_scenes``
- The ``scene`` folders contains every individual scene in a separate folder, in polygon and point cloud format
- The index to the training or test data is located in the <code>train_fragments_2048</code> and <code>test_fragments_2048</code> *csv* files, respectively, where 2048 is the point cloud size (see next section)

## Training the Model
The code is configured to train and test an example model. To initiate training, simply run:
```bash
python model_train.py
```

The file [`global_config.py`](global_config.py) contains a set of hyperparameters that are used to tune the network. The most prevalent are:
- **pc_size**: the point cloud size (assumes point clouds in dataset to be of the same size, e.g. 2048)
- **lvls**: the maximum training depth
- **epochs**: the number of epochs to train
- **capacity**: the number of filters for the encoders
- **batch_size**: the batch size to train with

Some extra information:
- The set of hyperparameters dictates the model name, e.g. ``sah_kdtree_3lvl_2048pc_128capacity_model_name``. Here, ``model_name`` is a user defined variable set in the class constructor (see [`model_train.py`](model_train.py))
- You may resume training using the model's `.continue_training()` functionality. Please note that this function loads the last recorded model parameters and optimizer's state from ``metadata`` directory and appends the statistics cached in the ``plots/your_model_name`` existing directory

Relevant output during the training process is cached in the ``plots/your_model_name`` directory. This folder contains (for each trained model):
- Training-specific diagrams
- A model checkpoint containing the parameters and optimizer's state, of the best in terms of tree cost model, cached in ``plots/your_model_name/your_model_name_c/`` directory
- The ``metadata`` folder contains the model from the last recorded epoch

## Testing the Model
To test the trained network, simply run:
```bash
python model_test.py
```
As discussed in the paper, the steps performed in this operation are:
- A scene is loaded from a predefined folder
- The primitive population is sampled
- An instance of ``treeNet_model.neural_kdtree`` is created, using the **same name** and **global_config parameters** that it was trained with
- The model parameters are loaded from the ``metadata`` folder for the associate model. To load the best recorded model one has to pull the file parameters located in ``plots/your_model_name/your_model_name_c/`` directory
- The tree structure (planes/offsets) is computed with greedy and recursive inference functions
- The output is stored in binary files in the ``plots/predict hierarchy`` folder with the scene's corresponding name. The internal node layout is cached in a level-order fashion 

This data can be used as a fixed-depth tree and then expanded into a *k*-d tree or a BVH using known conventional tree builders. See details in the paper.


## Building your Own Dataset
An example model is used by default, which loads all scenes at the dataset folder. To use your own dataset, you need to add a new folder with your own scenes at the ``datasets`` folder, where each scene is also stored at its own folder, similar to ``datasets/custom_scenes``.

Then, you need to follow the helper functions in the [`dataset_builder.py`](dataset_builder.py) script. For example:
- The ``build_pc`` function contains the code used to construct the point clouds from the polygon soup
- The ``build_vh_dataset`` and ``build_sah_dataset`` functions contains the code for training, testing and validating SAH and VH, respectively. This essentially creates:
    - A train folder containing the pointclouds along with the associated ``.csv`` to index them
    - A test folder containing folders for each scene with point clouds sampled from the whole population, along with an associated ``.csv`` to index these point clouds during training for validation

Extra notes:
- Further details on the construction and extraction process of these point clouds are in the paper
- Manually adding/removing/excluding point clouds from these folders during training requires to modify the associate ``.csv`` files

## How to Cite
The license is [MIT](LICENSE). If you use the contents of this repository for your work, please cite it as described below:

<blockquote>
<pre style="white-space:pre-wrap;">
In our work, we have used the source code~\cite{Evangelou_2023_CGI_2023}, available at <em>' https://github.com/cgaueb/nss'</em>.
</pre>

<pre style="white-space:pre-wrap;">
@article{Evangelou_2023_CGI_2023,
	title    = "A neural builder for spatial subdivision hierarchies",
    author   = "Evangelou, Iordanis and Papaioannou, Georgios and Vardis, Konstantinos and Gkaravelis, Anastasios",
	journal  = "The Visual Computer",
	volume   =  39,
	number   =  8,
	pages    = "3797--3809",
	month    =  aug,
	year     =  2023,
    DOI      = {10.1007/s00371-023-02975-y}
}
</pre>
</blockquote>


## Acknowledgments
This research was funded by the Hellenic Foundation for Research and Innovation (HFRI) under the “3rd Call for H.F.R.I. Research Projects to support Post-Doctoral Researchers” (Project No: 7310).

Credits for the included scenes (per-scene credits are in **Figure 2** of the paper):
- [Rendering resources](https://benedikt-bitterli.me/resources/) by Benedikt Bitterli 
- [Scenes for pbrt-v3](https://pbrt.org/scenes-v3) by Matt Pharr, Jakob Wenzel and Greh Humphreys
- [Amazon Lumberyard Bistro, Open Research Content Archive (ORCA)](http://developer.nvidia.com/orca/amazon-lumberyard-bistro) by Amazon Lumberyard
- [Computer Graphics Archive](https://casual-effects.com/data) by Morgan McGuire

## Contact Us
For any assistance, please contact at iordanise@aueb.gr (1st author) or kvardis@aueb.gr. Thank you!
