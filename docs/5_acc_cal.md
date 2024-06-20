# Accuracy Calculation
Here, we provide guides for calculating accuracy of your data.

### 1. Accuracy
#### 1.1 Arguments
There are several arguments for running `src/run/cal_acc.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to calculate accuracy. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to calculate accuracy.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `test`) Choose one of [`train`, `validation`, `test`].
* [`-n`, `--result_num`]: (default: `10`) The number of random data to visualize.

#### 1.2 Command
`src/run/cal_acc.py` file is used to calculate accuracy of the model with the following command:
```bash
python3 src/run/cal_acc.py --resume_model_dir ${project}/${name}
```