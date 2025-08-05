# SDHI_Tracker
Sequence-Based Object Tracker with Denoising and Historical Information

## Installation environment
```
conda create -n artrack_m
conda activate artrack_m
bash install.sh
```

* Add project path to python
```
export PYTHONPATH=<absolute_path_of_SeqTrack>:$PYTHONPATH
```

## Data preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

## Path Setting
%运行下列命令以设置项目路径
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
%运行上述命令后，可以前往以下两个文件中修改相关路径设置（例如预训练网络和数据集）
After running the above command, you can go to the following two files to modify the relevant path settings (such as the paths of the pre trained network and dataset)
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### The first stage of training %第一阶段训练
%进行训练前，可以将'experience/artrack/*.yaml'中的 *BACKBONE_PTH* 指定为提前准备好的预训练权重文件夹，并修改 *PRETRAIN_FILE*来指定预训练类型
Before training, we can specify the *BACKBONE_PTH* in 'experience/artrack/*.yaml' as the pre training weight folder prepared in advance, and modify *PRETRAIN_FILE* to specify the pre training type

%多卡训练，以两卡训练为例
Multi card training (taking two card training as an example)
```
torchrun --nproc_per_node 2 lib/train/run_training.py --script artrack --config artrack_256_full --save_dir ./output
```

Debugging in a single GPU card environment.%单卡调试

```
python lib/train/run_training.py --script artrack --config artrack_256_full --save_dir ./output
```

%使用`experiments/artrack`中的文件名替换`--config`就可以使用不同的实验设置
By replacing '-- config' with the file name in 'experiments/artrack', we can use different experimental settings.

### The second stage of training %第二阶段训练

%进行第二阶段训练前，需要将'experience/artrack_seq/*.yaml'中的 *PRETRAIN_PTH* 指定为第一阶段训练的结果

Before proceeding with the second stage of training, we need to specify *PRETRAIN_PTH* in 'experience/artrack_seq/*.yaml' as the result of the first stage of training.

```
torchrun --nproc_per_node 2 lib/train/run_training.py --script artrack_seq --config artrack_seq_256_full --save_dir ./output
```

## Test %测试

- LaSOT
```
python tracking/test.py artrack_seq artrack_seq_256_full --dataset lasot --threads 8 --num_gpus 1
python tracking/analysis_results.py _dataset_name lasot
```
- UAV123
```
python tracking/test.py artrack_seq artrack_seq_256_full --dataset uav --threads 8 --num_gpus 1
python tracking/analysis_results.py _dataset_name uav
```
- GOT10K-test
```
python tracking/test.py artrack_seq artrack_seq_256_full --dataset got10k_test --threads 8 --num_gpus 1
python lib/test/utils/transform_got10k.py --tracker_name artrack_seq --cfg_name artrack_seq_256_full
```
- TrackingNet
```
python tracking/test.py artrack_seq artrack_seq_256_full --dataset trackingnet --threads 8 --num_gpus 1
python lib/test/utils/transform_trackingnet.py --tracker_name artrack_seq --cfg_name artrack_seq_256_full
```


## Related references %相关事项
* Basic framework：[PyTracking](https://github.com/visionml/pytracking)
* Baseline model：[ARTrack](https://github.com/MIV-XJTU/ARTrack) 
* VOT test：[AlphaRefine](https://github.com/MasterBin-IIAU/AlphaRefine)
