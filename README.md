# UNet Model for City-Scapes Dataset

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->





<!-- /code_chunk_output -->

## Requirements
Install all requirements in `requirements.txt` file:
```nashorn js
pip3 install -r requirements.txt
```

## Train model
!! Before runing this command, you must change mode of this bash file by : 
```python
chmod +x run_train.sh
```

Training model by running this command
```nashorn js
./run_train.sh
```

## Evaluation 
!! Before runing this command, you must change mode of this bash file by : 
```python
chmod +x run_test.sh
```

Evaluate model by running this command 
```python
./run_test.sh
```

## Inference 
!! Before runing this command, you must change mode of this bash file by : 
```python
chmod +x run_infer.sh
```

```python
./run_infer.sh
```

## Dataset Architecture
```nashorn js

CityScapes
|----gtFine
|        |-- test
|        |-- train
|        |-- val
|    
|---- leftImg8bit
         |-- test
         |-- train
         |-- val
```
