# Parking Dwell Time Estimation Challenge

This folder contains all the **Parking Dwell Time Estimation MetaPKLot Challenge** related code, enabling anyone to reproduce the experiments by themselves.

## Challenge folder structure

The following are the main folders/experiments held in this challenge:

- **classification**: Training, validation and testing of a **single** CNN classification model.
- **ensemble**: Training, validation and testing of an **ensemble** CNN classification model.
- **siamese**: Training, validation and testing of a **siamese** CNN comparison model.
- **pipeline**: Parking Dwell Time main pipeline, being this the main experiment.
- **pipeline_time**: Alternative pipeline focusing on computing the processing time of the proposed approach.

Some extra code is also present, they were used to validate minor details of each experiment or to support the scripts’ execution:

- **tools**: Utils/general tools used to help the main experiments.
- **validations**: Minor experiments and validations.
- **requirements.txt**: Requirements file, containing all necessary libraries to run all scripts.

As mentioned in the paper, all results were an average of 5 runs. Therefore, all the available scripts (main experiments) are built to work in batches of 5 runs.

## MetaPKLot Dataset usage

To be able to run any script, the MetaPKLot Dataset needs to be cropped and stored in a specific way.

To help whoever wants to reproduce the experiments, we make available 2 cropping scripts that process the raw images, cropping, naming and saving them in the right format expected by the experiments scripts.

- `tools/generate_dataset.py`.
- `tools/generate_dataset_siamese.py`.

The first crop the dataset to work with the experiments **classification** and **ensemble**, and the second crop the dataset to work with the experiment **siamese**. This was done to easy some implementation details, but the resultant crop from ´tools/generate_dataset.py´ can be used in the **siamese** experiment with minor adjustments in the data_loader class.

Also, note that the raw images with the related annotation JSONs needs to be fed to the **pipeline** and **pipeline_time** experiments, as they aim to test and execute the complete process (cropping, classifying and comparing).

Those scripts are of great importance to the right functioning of all Parking Dwell Time experiments due to the naming convention used in each cropped image. Any other way of naming the resultant images will not work in those scripts.

### Executing the cropping scripts

To be able to run the cropping scripts, the MetaPKLot dataset needs to be re-organized in the following way:

```
<dataset_folder/any_selected_folder>
├── CNRParkEXT
│   ├── camera1 (Folder with all subset images. Follow the same structure as in the Dataset repository.)
│   ├── camera2
│   ├── camera3
│   ├── camera4
│   ├── camera5
│   ├── camera6
│   ├── camera7
│   ├── camera8
│   ├── camera9
│   ├── cnr-camera-1_spots.json
│   ├── cnr-camera-2_spots.json
│   ├── cnr-camera-3_spots.json
│   ├── cnr-camera-4_spots.json
│   ├── cnr-camera-5_spots.json
│   ├── cnr-camera-6_spots.json
│   ├── cnr-camera-7_spots.json
│   ├── cnr-camera-8_spots.json
│   └── cnr-camera-9_spots.json
├── PKLot
│   ├── PUCPR (Folder with all subset images. Follow the same structure as in the Dataset repository.)
│   ├── UFPR04
│   ├── UFPR05
│   ├── pucpr_spots.json
│   ├── ufpr04_spots.json
│   └── ufpr05_spots.json
└── PLds
    ├── isshk (Folder with all subset images. Follow the same structure as in the Dataset repository.)
    ├── qridr
    ├── vmlix
    ├── vxusd
    ├── isshk_spots.json
    ├── qridr_spots.json
    ├── vmlix_spots.json
    └── vxusd_spots.json
```

This was done to easy implementation details. Also, this is the expected folder structure for the scripts **pipeline** and **pipeline_time** (i.e. they only work with this folder structure).

An example run of each script can be found within the files itself.

## Main experiments execution

All scripts were formatted as similar as possible to easy reproduction and management. All experiments folders contains the following main scripts:

- `main.py`: Main script, holds all the logic to train, validate and test models, or execute the pipelines.
- `run.py`: Executable script, executes the peripheral logic (names the execution, create result folders, etc.) and also executes the `main.py` the necessary 5 times.
- `summary.py`: Summary script that concatenates all 5 runs results in a single summary file.

To be able to configure each experiment run, the `run.py` (and consequently `main.py`) receives a config JSON file as input. This JSON defines all configurable parameters, such as:

- Train/val and test datasets (according to the leave-one-out strategy).
- Loss function, optimizer and related configurations.
- Model used.
- Experiment details (Epochs, batch size, etc.).

All other folders/scripts are peripheral to the `main.py`, holding code related to data load, model wrappers, etc.

It’s important to note that this structure needs to be followed to guarantee proper experiment functioning. For example, to test a new model, it’s necessary to create a new model class wrapper (following the “interface” exemplified in the `mobilenet_v3_large.py` wrapper class) and pass it in the config json. The following structure is used to run any `run.py` script:

```
python3 run.py -c <path_to_config_file>
```

If you want to run only `main.py` or `summary.py`, you can find the parameter list inside the file.

### Scripts Nuances

The scripts **ensemble**, **pipeline** and **pipeline_time** have a few more details to be able to run correctly.

#### Ensemble

To run the **ensemble** experiment, it’s necessary to indicate the origin models (i.e. `c0` models, according to the paper). This models comes from the **classification** experiment. To do that, it’s necessary to create a folder (you can choose the name, but we choose `origin`), which will contain the origin models, following the structure below:

```
origin
├── cnrparkext_plds (trained at CNRPark-EXT/PLds and tested at PKlot)
│   ├── model_cnrparkext_plds_6911e.pt
│   ├── model_cnrparkext_plds_74e73.pt
│   ├── …
├── pklot_cnrparkext (trained atCNRPark-EXT/PKLot and tested a PLds)
│   ├── model_pklot_cnrparkext_365fa.pt
│   ├── model_pklot_cnrparkext_73d1c.pt
│   ├── …
└── pklot_plds (trained at PKLot/PLds and tested at CNRPark-Ext)
    ├── model_pklot_plds_2621d.pt
    ├── model_pklot_plds_35b6d.pt
    ├── …
```

This folder is already created with our origin models.

#### Pipeline & Pipeline_time

Similar to the `origin` folder at the **ensemble** experiment, for both pipelines experiments, it’s necessary to indicate the models that will be used in each execution. To do that, it’s necessary to create a folder (you can choose the name, but we choose `resources`), which will contain the models, following the structure below:

```
resources
├── classification
│   ├── CNRParkEXT
│   │   ├── ensemble_pklot_plds_205ef
│   │   │   ├── model_0.pt
│   │   │   ├── …
│   │   ├── ensemble_pklot_plds_4499a
│   │   │   ├── …
│   │   ├── ensemble_pklot_plds_7e87d
│   │   │   ├── …
│   │   ├── ensemble_pklot_plds_9ef3a
│   │   │   ├── …
│   │   └── ensemble_pklot_plds_a4d31
│   │       ├── ….
│   ├── PKLot
│   │   ├── ensemble_cnrparkext_plds_2bff1
│   │   │   ├── model_0.pt
│   │   │   ├── …
│   │   ├── ensemble_cnrparkext_plds_a9e8a
│   │   │   ├── …
│   │   ├── ensemble_cnrparkext_plds_adeeb
│   │   │   ├── …
│   │   ├── ensemble_cnrparkext_plds_ce445
│   │   │   ├── ….
│   │   └── ensemble_cnrparkext_plds_ec10a
│   │       ├── …
│   └── PLds
│       ├── ensemble_pklot_cnrparkext_3f836
│       │   ├── model_0.pt
│       │   ├── …
│       ├── ensemble_pklot_cnrparkext_41076
│       │   ├── …
│       ├── ensemble_pklot_cnrparkext_450c7
│       │   ├── …
│       ├── ensemble_pklot_cnrparkext_71ea6
│       │   ├── …
│       └── ensemble_pklot_cnrparkext_7d9e6
│           ├── …
└── siamese
    ├── CNRParkEXT
    │   ├── model_pklot_plds_01e83.json
    │   ├── model_pklot_plds_01e83.pt
    │   ├── …
    ├── PKLot
    │   ├── model_cnrparkext_plds_4366c.json
    │   ├── model_cnrparkext_plds_4366c.pt
    │   ├── …
    └── PLds
        ├── model_pklot_cnrparkext_131de.json
        ├── model_pklot_cnrparkext_131de.pt
        ├── …
```

In the `classification` sub-folder, it’s necessary to add each ensemble model folder (**ensemble** experiment model output). In the `siamese` sub-folder it’s necessary to add each siamese model and its related result JSON (**siamese** experiment outputs).

This folder is already created with our resource models.

### Running with already trained models

If you plan to execute the experiments **pipeline** and **pipeline_time** with our already trained models (our results) it’s not necessary to change anything in the code, just execute the `run.py` with the models available at the `resources` folder.

But, if you want to execute the other experiments (**classification**, **ensemble** and **siamese**) with our already trained models, it will be necessary to modify both `run.py` and `main.py` to loop over the experiment models specific folder (stored inside `_models`).

The alternative versions `_re_run.py` and `_re_main.py` are available for this purpose. To run it execute:

```
python3 _re_run.py -c <path_to_config_file> -m <path_to_models_folder>

or

python3 _re_run.py -c <path_to_config_file> -e <path_to_ensembles_folder> (for the **ensemble** experiment)
```

### Scripts outputs

All outputs are formatted in the same way to facilitate management and comparison. Therefore, all folders that start with `_` are automatically generated by the scripts and contain the related results. When running `run.py`, the outputs are:

- A new experiment summary stored in `_summaries`.
- A new experiment segmented results (one by each of the 5 runs) stored in the experiment folder (with the same name as the summary file) inside `_results`.
- A set of models stored in the experiment folder (with the same name as the summary file) inside of `_models` (when applicable).

## Validations folder

Some validation scripts (and minor experiments) that were used during the project run are stored in this folder. A naming convention was used to facilitate the search (to help us and to help you, if you want to look at them):

`<script_indicator>_<brief_description>`

The script indicators are:

- 1: Validations related to the experiments **classification** and **ensemble**.
- 2: Validations related to the experiment **siamese**.
- 3: Validations related to the experiments **pipeline** and **pipeline_time**.
- 4: General validations (mainly related to the car ID logic).
