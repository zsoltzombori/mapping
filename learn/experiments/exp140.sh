# parent: exp88
# loss_type: lprp -> democracy
# opt_steps=10 
# alpha=1.01



exp=140
schema="cmt_renamed"
basedir="outdata/cmt_renamed"
epochs=10
batch_size=40
neg_weight=3
num_layers=4
d_model=128
lr=0.0001
lr_decay_steps=30
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
loss_type="democracy"
split="0.7,0.15,0.15"
monitor_probs=0
opt_steps=10
alpha=1.01
GPU=6

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="${basedir}/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs --opt_steps $opt_steps --multiplier $alpha" 
CMD="nohup $CMD > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo $CMD
eval $CMD
