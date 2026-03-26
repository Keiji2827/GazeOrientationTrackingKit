# GazeWholeBodyTransformer
Gaze estimation from whole body kepoints


# Command Lines
## training
```
python3 -u -m models.tools.newfile --n_frames 7 --lr 1e-5 --logging_steps 1000 --is_GAFA --no_use_lstm --no_use_MF --train_batch_size 16
```
## evaluation
```
python3 -u -m models.tools.newfile --test --n_frames 7 --logging_steps 200  --is_GAFA --no_use_lstm --model_checkpoint output/checkpoint***/state_dict.bin 
```