# Emotion Analysis from Speech - PyTorch Implementation

This repository contains work from https://github.com/samsudinng/speech_emo_recognition 

This is the topic of my final year project (Bachelor in CS, Nanyang Technological University, Singapore). This repository also includes resources for SER, including publications, datasets, and useful python packages. 

### Features extraction

The framework supports features extraction from the following database:

1. [__IEMOCAP__](https://sail.usc.edu/iemocap/), a database of approximately 12 hours of audiovisual data, including video, speech, motion capture of face, and text transcriptions. The database was organized into 5 sessions, where each session contains dialog between 2 unique actors (1 male, 1 female). Each session contains scripted and improvised dialog, manually segmented into utterances and annotated by at least 3 human annotators. The average duration of each utterance is 4-5 sec. The emotion classes are {*anger, happiness, excitement, sadness, frustration, fear, surprise, neutral, others*}.

2. [__emoDB__](http://www.emodb.bilderbar.info/start.html), a German database of emotional speech of approximately 500 utterances. The utterances were recorded with 10 actors (5 male, 5 female). The average duration of each utterance is 2-3 sec. The emotion classes are {*happy, angry, anxious, fearful, bored, disgust, neutral*}.

3. [__THAI_SER__](https://github.com/vistec-AI/dataset-releases/releases/tag/v1), a THAI database of of 41 hours, 36 mins of emotional speech of 27,854 utterances. It contains 100 recordings split into Studio and Zoom with over 200 actors. The emotion classes associated are {*happiness, anger, sadness, frustration, neutral*}.


### SER Models

Four models are available in the framework:  

1. __AlexNet Finetuning__ 
- Pre-trained AlexNet, finetuned to classify emotion from speech spectrogram images (IEMOCAP). 
- The model is based on *torchvision.models.alexnet* model in pyTorch.

2. __FCN+GAP__
- Pre-trained AlexNet with the fully connected layers replaced with global average pooling (GAP).
- Finetuned to classify emotion from speech spectrogram images (IEMOCAP)
- Fully-connected layers are prone to over-fitting and require large number of parameters. GAP was proposed by Lin et. al. (2014) in [*Network In Network*](https://arxiv.org/abs/1312.4400) to address these issues.
- This model perform as well as AlexNet Finetuning but requiring only 4.5% as many parameters.

3. __ResNet18__
- Pre-trained ResNet18, finetuned to classify emotion from speech spectrogram images (IEMOCAP).
- The model is based on *torchvision.models.resnet18* model in pyTorch. 
- This model perform better than AlexNet and FCN+GAP on the THAI dataset.

4. __VGG11__
- Pre-trained VGG11, finetuned to classify emotion from speech spectrogram images (IEMOCAP).
- The model is based on *torchvision.models.vgg11* model in pyTorch. 


The model to be trained can be selected via command line with the following labels. The summary of model parameters and accuracy (5-fold, speaker independent cross-validation) are also summarized below.

For 16khz IEMOCAP corpus:

|Model label|Model Name|# of Params.|Weighted Accuracy|Unweighted Accuracy| Model Setting |
|-----------|----------|----------|----------|----------| ----------|
|*'alexnet'*|AlexNet Finetuning| ~57m | 74.0% | 64.4%| baseline + stability training|
|*'alexnet_gap'*|FCN+GAP| ~2.5m | 73.2% | 62.6% | baseline |
|*'resnet18'*|ResNet 18| ~11m | 71.6% | 59.8% | baseline |

For 8khz IEMOCAP corpus:

|Model label|Model Name|# of Params.|Weighted Accuracy|Unweighted Accuracy| Model Setting |
|-----------|----------|----------|----------|----------| ----------|
|*'alexnet'*|AlexNet Finetuning| ~57m | 68.7% | 57.8%| baseline + stability training|
|*'alexnet_gap'*|FCN+GAP| ~2.5m | 70.5% | 58.3% | baseline |
|*'resnet18'*|ResNet 18| ~11m | 68.7% | 57.1% | baseline |


------------------------------------
## Requirements:

Audio files must be in .wav format.

-----------------------------
## Pretrained models

The models have been trained from the IEMOCAP dataset can be found in the **trained_models** folder.
For usage of pretrained model in application, see example in **flaskproject** folder.

## How to run the scripts:

#### 1. First extract features from the given database using the following command:
```python
python extract_features.py database_name path/to/database --save_dir path/to/where_the_extracted_features_are_saved --save_label name_of_file_to_be_saved
```
Example:
```python
python extract_features.py IEMOCAP /database/IEMOCAP_full_release --save_dir /home/desktop/ser --save_label logspec_features
 ```
 If using THAI SER then
 ```python
python extract_features.py THAI /database/THAI_SER --nfreq 100 --save_dir /home/desktop/ser --save_label THAI_logspec_features
 ```
#### 2. Train the model with the following command:
```python 
 python crossval_SER.py
``` 
#### 3. Get model aacuracies
```python 
 python get_crossval_result.py name_of_label_input_in_Step_1 num_of_kfold_runs num_of_completed_runs path_to_the_pkl_files_from_Step2 
``` 
 Example:
```python 
python get_crossval_result.py logspec_features 5 5 /home/desktop/ser
 ```
 
 To train with addition of noise,
 - change the directory of noise files in features_util.py
 - add in --mixnoise in Step 1
 - Use python crossval_alexnet_spec2gray_CLoss.py instead in Step 2

------------------------------------
## SER Publications
|Title| Link |
|-----------|----------|
| Deep Residual Learning for Image Recognition | https://arxiv.org/abs/1512.03385 |
| Very Deep Convolutional Networks for Large-Scale Image Recognition | https://arxiv.org/abs/1409.1556 |




------------------------------------
## SER Datasets

### IEMOCAP

### emoDB

### THAI SER




