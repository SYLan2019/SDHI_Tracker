# SDHI_Tracker
Sequence-Based Object Tracker with Denoising and Historical Information

# Please unzip a.rar first, all the code is in this compressed file.

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
Run the following command to set the project path
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running the above command, you can go to the following two files to modify the relevant path settings (such as the paths of the pre trained network and dataset)
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### The first stage of training 
Before training, we can specify the *BACKBONE_PTH* in 'experience/artrack/*.yaml' as the pre training weight folder prepared in advance, and modify *PRETRAIN_FILE* to specify the pre training type


Multi GPU training (taking two card training as an example)
```
torchrun --nproc_per_node 2 lib/train/run_training.py --script artrack --config artrack_256_full --save_dir ./output
```

Debugging in a single GPU environment.

```
python lib/train/run_training.py --script artrack --config artrack_256_full --save_dir ./output
```


By replacing '-- config' with the file name in 'experiments/artrack', we can use different experimental settings.

### The second stage of training 


Before proceeding with the second stage of training, we need to specify *PRETRAIN_PTH* in 'experience/artrack_seq/*.yaml' as the result of the first stage of training.

```
torchrun --nproc_per_node 2 lib/train/run_training.py --script artrack_seq --config artrack_seq_256_full --save_dir ./output
```

## Test 

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


## Related references 
* Basic framework：[PyTracking](https://github.com/visionml/pytracking)
* Baseline model：[ARTrack](https://github.com/MIV-XJTU/ARTrack) 
* VOT test：[AlphaRefine](https://github.com/MasterBin-IIAU/AlphaRefine)
