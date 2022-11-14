# Agapi's tests for sequencial updates

exp=0
schema=syn4
basedir="synthetic"
epochs=2000
batch_size=2 # 1
neg_weight=0.0
num_layers=1 # 2
d_model=1024
lr=0.001
lr_decay_steps=1000000 # not applied if steps>epochs, TODO: verify
# optimizer=adamax
# beta1=0.3
# beta2=0.9
optimizer=sgd # default: sgd
beta1=0.0
beta2=0.0
CHAR_TOKENIZER=0
loss_type="seq_prp" # "seq_prp" "lprp" 
split="1,0,0"
monitor_probs=1
GPU=0
opt_steps=20 # 100
alpha=1.05

echo "Schema $schema"
echo "GPU $GPU"

checkpoint_dir="checkpoints/exp${exp}"
datadir="${basedir}/${schema}"
outdir="out/exp${exp}"
mkdir -p $outdir

# --checkpoint_path ${checkpoint_dir}
CMD="python train.py --datadir $datadir --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs --opt_steps $opt_steps --multiplier $alpha" 
CMD="nohup $CMD > $outdir/${schema}.cout 2> $outdir/${schema}.cerr &"
CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo $CMD
eval $CMD
