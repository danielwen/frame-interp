# Video Frame Interpolation

10-707: Topics in Deep Learning

## Dataset

[DAVIS](https://davischallenge.org/davis2017/code.html)

## Naive Baseline

```
python naive.py <image_dir>
```

## Train model
```
python train.py config.json
```
Edit config.json to change data directory and model

To train over ssh on AWS. Use
```
nohup python train.py config.json &
```
This will make sure process is still running if ssh is disconnected. Program logs will be in nohup.out