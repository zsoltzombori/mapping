# parent: exp45
# loss_type: lprp -> prp

exp=46
schema=syn1
basedir="synthetic"
epochs=200
batch_size=20
neg_weight=3.0
num_layers=1
d_model=32
lr=0.0003
lr_decay_steps=30
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
loss_type="prp"
split="0.5,0.5,0"
monitor_probs=1
GPU=6

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="${basedir}/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs" 
CMD="nohup $CMD > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo $CMD
eval $CMD
