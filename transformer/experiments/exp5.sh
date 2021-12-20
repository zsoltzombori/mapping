# parent: exp2
# epochs 200 -> 50
# batch_size 12 -> 20
# d_model 512 -> 1024

exp=5
schema=cmt_renamed
epochs=50
batch_size=20
neg_weight=0.0
num_layers=2
d_model=1024
lr=0.001
GPU=7

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="outdata/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr}  --checkpoint_path ${checkpoint_dir}  > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
echo $CMD
eval $CMD

