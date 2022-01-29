# parent: exp32
# d_model: 128 -> 256

exp=33
schema=npd_dummy
epochs=1000
batch_size=40
neg_weight=3.0
num_layers=4
d_model=256
lr=0.01
lr_decay_steps=10
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
GPU=5

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="outdata/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
echo $CMD
eval $CMD
