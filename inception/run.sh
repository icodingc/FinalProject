NOWDIR=`dirname "$0"`
NOWDIR=`cd $NOWDIR; pwd`
export PYTHONPATH=${HOME}/workshops/FinalProject/inception/

/usr/bin/python $NOWDIR/inception/flowers_train.py \
    --train_dir=$NOWDIR/log_08302 \
    --pretrained_model_checkpoint_path=$NOWDIR/log_0830/model.ckpt-60000 \
    --data_dir=/home/zhangxuesen/workshops/data/tfrecord \
    --initial_learning_rate=0.01 \
    --input_queue_memory_factor=1 \
    --max_steps=40000 \
    --num_epochs_per_decay=30 \
    --learning_rate_decay_factor=0.16 \
    --batch_size=64 \
    --image_size=32 \
    --num_readers=1 \
    --num_preprocess_threads=4 \
