# Training CNN Image Classification
Here, we provide guides for training a CNN image classification model.

### 1. Configuration Preparation
To train a CNN image classification model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: mps                             # examples: [0], [0,1], [1,2,3], cpu, mps... 

# project config
project: outputs/CNN
name: MNIST

# image setting config
height: 28
width: 28
color_channel: 1
convert2grayscale: False

# data config
workers: 0                              # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
MNIST_train: True                       # if True, MNIST will be loaded automatically.
class_num: 10                           # Number of image label classes.
MNIST:
    path: data/
    MNIST_valset_proportion: 0.2        # MNIST has only train and test data. Thus, part of the training data is used as a validation set.
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 128
epochs: 20
lr: 0.001

# prediction result visualzation config
result_num: 10                          # number of prediction results to be shown

# logging config
common: ['train_loss', 'train_acc', 'validation_loss', 'validation_acc']
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```

When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.