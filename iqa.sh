for method in "random" "UEs" "RUE" "TUE" "GUE" 
do 
    data=`date +%F-%T`
    printf "${data} " >> logs/iqalog.txt
    printf "${data} ${method}\n" 
    dataset="WebFace10"
    python iqa.py $method $dataset "log"
    # python fid.py $method $dataset 
    # root="/data/zhangzhiling/"
    # root_="/fidtrain"
    # python -m pytorch_fid "${root}${dataset}${root_}clean" "${root}${dataset}${root_}${method}" >> logs/iqalog.txt
    printf "\n" >> logs/iqalog.txt
    printf "${root}${dataset}${root_}clean"
done
