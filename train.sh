if [[ "$1" == "mp" ]] || [[ "$1" == "all" ]] ; then
    ./script/train_merge_mp_3.sh $2 $3
fi

if [[ "$1" == "r" ]] || [[ "$1" == "all" ]] ; then
    ./script/train_merge_reacher_3.sh $2 $3
fi
