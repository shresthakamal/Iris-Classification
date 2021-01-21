irisclassification
==============================

Classification of Iris 



<p align="center">
<img src=""  />
</p>


Project Organization
------------
```
├── api
│   ├── app.py
│   ├── config
│   │   └── config.py
│   ├── resources
│   ├── static
│   └── templates
│   
├── checkpoints
│   
├── data
│   ├── processed
│   └── raw
│   
├── docs
│   ├── Analysis.md
│   └── Requirements.md
│   
├── notebooks
│ 
├── irisclassification
│   ├── config
│   │   └── config.py
│   ├── data
│   │   └── make_dataset.py
│   ├── dispatcher
│   ├── features
│   │   ├── build_features.py
│   ├── models
│   │   ├── test_model.py
│   │   └── train_model.py
│   ├── utils
│   ├── visualisation
│   |   └── visualisation.py
│   └── main.py
│ 
├── Dockerfile
│ 
├── run.sh
├── logs
├── references
├── requirements.txt
├── README.md
├── LICENSE
└── tests
    └── test_environment.py
```
--------


## Getting Started

### Requirements

```
pip install -r requirements.txt
```

### Download the dataset

The following command will download the dataset from the URL given in `src/config/config.py` file .

```
python -m irisclassification.data.make_dataset
```

### Run

```
python -m irisclassification.main
```
OR

```
./run.sh
```

### Test

```
python -m tests.test_environment
```


### To-do List

- [ ] Download dataset
- [ ] Pre-process data
- [ ] Train model
- [ ] Test model
- [ ] Main Pipeline

-------------------------------
