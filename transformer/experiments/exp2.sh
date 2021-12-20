# parent: exp1
# neg_weight 3.0 -> 0.0
# epochs 500 -> 200

exp=2
schema=cmt_renamed
epochs=200
batch_size=12
neg_weight=0.0
num_layers=2
d_model=512
lr=0.001
GPU=4

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="outdata/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr}  --checkpoint_path ${checkpoint_dir}  > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
echo $CMD
eval $CMD

