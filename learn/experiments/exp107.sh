# parent: exp105
# neg_weight: 0.1 -> 0.0

exp=107
schema="MULTI"
basedir="outdata/npd"
epochs=200
batch_size=10
neg_weight=0.0
num_layers=1,3
d_model=32
lr=0.01
lr_decay_steps=30
optimizer=adamax
beta1=0.3
beta2=0.9
CHAR_TOKENIZER=0
loss_type="lprp"
split="0.7,0.15,0.15"
monitor_probs=0
seq_out_len=50
GPU=4
CORENUM=15

outdir="out/exp${exp}"
mkdir -p $outdir
checkpoint_dir="none"

CMD="python train.py --datadir ${basedir}/{} --epochs $epochs --batch_size ${batch_size} --neg_weight ${neg_weight} --num_layers ${num_layers} --d_model ${d_model} --lr ${lr} --lr_decay_steps ${lr_decay_steps}  --checkpoint_path ${checkpoint_dir} --optimizer $optimizer --beta1 $beta1 --beta2 $beta2 --char_tokenizer $CHAR_TOKENIZER --loss_type ${loss_type} --split ${split} --outdir ${outdir} --monitor_probs $monitor_probs --seq_out_len ${seq_out_len}" 
CMD="$CMD > $outdir/{}.cout 2> $outdir/{}.cerr"
CMD="CUDA_VISIBLE_DEVICES=$GPU $CMD"


echo "Schema $schema"
nohup find ${basedir} -maxdepth 1 -type d -not -name "npd" | xargs basename -a | parallel -j $CORENUM --no-notice $CMD &
