NOWDIR=`dirname "$0"`
NOWDIR=`cd $NOWDIR; pwd`
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=${HOME}/workshops/FinalProject/inception/

python $NOWDIR/inception/flowers_eval.py \
    --eval_dir=$NOWDIR/log_val \
    --data_dir=/home/zhangxuesen/workshops/data/tfrecord \
    --checkpoint_dir=$NOWDIR/log_0830 \
    --input_queue_memory_factor=1 \
    --batch_size=100 \
    --image_size=32 \
    --num_readers=1 \
    --num_preprocess_threads=4 \

