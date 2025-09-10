gpuid=0

DATA_ROOT=/root/Skinpbdcn/data/Skin_Cancer
cd ../../../

echo "============= pre-train ============="
python pretrain.py --dataset Skin_Cancer --data_path $DATA_ROOT --model ResNet12 --method meta_Skinpbdcn --image_size 84 --gpu ${gpuid} --lr 0.02 --t_lr 1e-3 --epoch 250 --milestones 100 150 --save_freq 100 --reduce_dim 640 --dropout_rate 0.8 --val meta --val_n_episode 600
