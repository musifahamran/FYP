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
