gpuid=0
cd ../../../

DATA_ROOT=/root/Skinpbdcn/data/Skin_Cancer
MODEL_1SHOT_PATH=./checkpoints/Skin_Cancer/ResNet12_meta_Skinpbdcn_2way_1shot_metatrain/best_model.tar
MODEL_5SHOT_PATH=./checkpoints/Skin_Cancer/ResNet12_meta_Skinpbdcn_2way_5shot_metatrain/best_model.tar

N_SHOT=1
python test.py --dataset Skin_Cancer --data_path $DATA_ROOT --model ResNet12 --method meta_Skinpbdcn --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 10 --test_n_episode 2000

N_SHOT=5
python test.py --dataset Skin_Cancer --data_path $DATA_ROOT --model ResNet12 --method meta_Skinpbdcn --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 10 --test_n_episode 2000

