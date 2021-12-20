# parent exp1
# epochs 500 -> 200
# optimizer sgd -> adam

exp=7
schema=cmt_renamed
epochs=500
batch_size=12
neg_weight=3.0
num_layers=2
d_model=512
lr=0.001
optimizer=adam
GPU=5

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="outdata/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --optimizer $optimizer  --checkpoint_path ${checkpoint_dir}  > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
echo $CMD
eval $CMD

