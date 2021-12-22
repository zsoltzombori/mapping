# parent: exp4
# epochs 50 -> 100
# d_model 512 -> 256

exp=5
schema=cmt_renamed
epochs=50
batch_size=8
neg_weight=0.0
num_layers=3
d_model=256
lr=0.001
GPU=5

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="outdata/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr}  --checkpoint_path ${checkpoint_dir}  > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
echo $CMD
eval $CMD

