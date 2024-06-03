# 競賽報告與程式碼 TEAM_5619 建構無人機於自然環境偵察時所需之導航資訊競賽 II － 導航資料生成 競賽

## File Hierarchy:

| Directory or file | Description |
| ---- | ---- |
| /images/ | Store images for `README.md` |
| /Dataset/Training_dataset/img/ | Store images for training. |
| /Dataset/Training_dataset/label_img/ | Store ground truth images for training. |
| /Dataset/Training_dataset/img/ | Store images for public testing. |
| /PrivateDataset/Testing_dataset/ | Store images for private testing. |
| /log/ | Store information from previous training. |
| /result/ | Store the result of the last inference. |
| /model/ | Store the models from the last training. During training, two models will be saved: `best_model.pth` and `newest_model.pth`. `best_model.pth` is the model with the lowest validation loss, while `newest_model.pth` is the most recent model. |
| /model/Strongest_model.pth | My best model in this competition, which scored `0.808814` in the public testing dataset and `0.808727` in the private testing dataset. |
| /utils.py | Python file which stores every helper function in this project. |
| /Dataset.py | Python file which stores the `NavigationDataset` for training and inferencing, and the `ValidationDataset` for checking the F-measure score. |
| /Loss.py | Python file which stores the `TverskyLoss` and `WeightedCrossEntropyLoss` source code for training. |
| /Model.py | Python file which stores my version of the `Unet` class for training. |
| /Train.ipynb | Jupyter notebook source code for training the model. |
| /Inference.ipynb | Jupyter notebook source code for inferencing the model after training. |
| /Check_Fmeasure.ipynb | Jupyter notebook source code to check the F-score in the `training_dataset` after training. |

## Before Running:

* Make sure your environment supports Python 3.9.19 or above, PyTorch 2.1.2 or above, and CUDA 12.1 or above for GPU computation.
* Download this project to your environment.
* Create empty directories: `/log/`, `/model/`, `/result/` and `/temp/`.
* Download `Dataset.zip` and `PrivateDataset.zip` from [Dataset](https://drive.google.com/file/d/1UoapNsosdGx4X2nO9FrdaqElFoc8BnC0/view?usp=sharing) and [PrivateDataset](https://drive.google.com/file/d/1lNh7ewL8dOc_2gOlL6azcWLePfHlmxME/view?usp=sharing).
* Unzip `Dataset.zip` and `PrivateDataset.zip` to the root of this project.
* (Optional) Download `Strongest_model.pth` from [Strongest_model.pth](https://drive.google.com/file/d/1kPrNtFWuDS1bq-hxK6VCbTn6Egh47R_F/view?usp=sharing) to the `/model/` directory.
* To open Jupyter notebook:
  1. Use the notebook package, type 
     ```
     pip install notebook
     ```
     In terminal.  
     After installation, in the terminal at the root directory of this project, type
     ```
     jupyter notebook
     ```
     ![Ipynb](./images/jupyter.jpg)
  2. Use VSCode  
     Install the Python and Jupyter extensions in your VSCode  
     ![Python extension](./images/Python.jpg)
     ![Jupyter extension](./images/Juypter_extension.jpg)  
     Then you can run .ipynb files in VSCode.

## Hyperparameters

| Name | Description |
| ---- | ---- |
| `batch_size` | `50` is the maximum size for 1080 Ti * 4. If your GPU has a larger CUDA memory, this parameter can be larger. |
| `learning_rate` | Initial learning rate for training. This value showed the best performance in training based on my trials. |
| `max_epoch` | In my observation, after `80` epochs, the loss in the training and validation datasets has already converged. |
| `valid_set_percent` | The percentage of the validation set in the original training dataset. I used `0.1` in this competition. |
| `weight_decay` | Parameter to implement L2 regularization. `0.02` showed the best performance in training based on my trials. |
| `pos_weight` | Positive sample weight in weighted binary cross entropy loss. `0.65/0.35` showed the best performance in training based on my trials. |
| `alpha` and `beta` in `TverskyLoss` | I used `alpha + beta = 1` and `beta = 0.3` in this competition. |

* `batch_size`, `learning_rate`, `max_epoch`, and `valid_set_percent` in `Train.ipynb`  
  ![hyper1](./images/main_hyper_parameter.png)

* `weight_decay` and `pos_weight` in `Train.ipynb`  
  ![hyper2](./images/weight_decay_and_pos_weight.png)

* `alpha` and `beta` in `Loss.py`  
  ![TverskyLoss](./images/alpha_and_beta_in_TverskyLoss.png)

## Usage

* Make sure there are empty directories: `/model/`, `/log/`, `/result/` and `/temp/`.
* (Optional) If you want to use my strongest model, download it from [Strongest_model.pth](https://drive.google.com/file/d/1kPrNtFWuDS1bq-hxK6VCbTn6Egh47R_F/view?usp=sharing) to `/model/`.
* Before running any file, ensure you have already downloaded the [Dataset](https://drive.google.com/file/d/1UoapNsosdGx4X2nO9FrdaqElFoc8BnC0/view?usp=sharing) and [PrivateDataset](https://drive.google.com/file/d/1lNh7ewL8dOc_2gOlL6azcWLePfHlmxME/view?usp=sharing) and unzipped them.
* Run `Train.ipynb` to train the model. You can change the hyperparameters.
* Run `Check_Fmeasure.ipynb` to check the F-score in the training dataset.
* Run `Inference.ipynb` to perform inference using the trained model. After running `Inference.ipynb`, it will create `result.zip` containing the inference results.  
* If you want to use a different model, change the file path from `"./model/best_model.pth"` to the desired model path in `Inference.ipynb` or `Check_Fmeasure.ipynb`.
  ![file path](./images/model_path.png)

## Note

The dataset in this project is downloaded from  
[以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 II － 導航資料生成競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/35)  
and renamed.

| Before rename | After rename |
| ---- | ---- |
| /35_Competition 2_Training dataset_V3.zip/Training_dataset/img | /Dataset/Training_dataset/img/ |
| /35_Competition 2_Training dataset_V3.zip/Training_dataset/label_img | /Dataset/Training_dataset/label_img/ |
| /35_Competition 2_public testing dataset.zip/img | /Dataset/Testing_dataset/img/ |
| /35_Competition 2_Private Test Dataset.zip/35_Competition 2_Private Test Dataset/img | /PrivateDataset/Testing_dataset/img/ |
