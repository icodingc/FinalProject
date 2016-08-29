NOWDIR=`dirname "$0"`
NOWDIR=`cd $NOWDIR; pwd`
export PYTHONPATH=/home/zhangxuesen/workshops/staff/inception/

/usr/bin/python $NOWDIR/inception/flowers_train.py \
    --train_dir=$NOWDIR/log \
    --data_dir=/home/zhangxuesen/workshops/data/tfrecord \
    --initial_learning_rate=0.05 \
    --input_queue_memory_factor=1 \
    --max_steps=110000 \
    --num_epochs_per_decay=30 \
    --learning_rate_decay_factor=0.50 \
    --batch_size=64 \
    --image_size=32 \
    --num_readers=1 \
    --num_preprocess_threads=4 \

###################3
#base_lr:0.1
#weight_decay:0.0001
#batch_size:128
#stepsize:100000
#~200 epoch
