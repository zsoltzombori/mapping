# parent: exp1
# epochs 50 -> 100
# num_layers 2 -> 4
# batch_size 12 -> 20

exp=5
schema=cmt_renamed
epochs=100
batch_size=20
neg_weight=3.0
num_layers=4
d_model=512
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

