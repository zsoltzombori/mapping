# parent: exp10
# epochs: 50 -> 100
# beta2: 0.999 -> 0.7

exp=14
schema=cmt_renamed
epochs=100
batch_size=12
neg_weight=3.0
num_layers=2
d_model=512
lr=0.0001
optimizer=adamax
beta1=0.5
beta2=0.7
GPU=4

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="outdata/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2  > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
echo $CMD
eval $CMD
