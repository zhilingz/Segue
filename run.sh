log="log" # "log" "ter"
log_path="logarti.txt" # logarti loggue
# python noise_train.py "GUE" "WebFace10" "log" $log_path
for method in "GUE" # "GUE" "UEs" "RUE" "TUE" "_clean" 
do 
    # printf "%s %s\n" $method "WebFace10"
    # printf "noise_train "
    # python noise_train.py $method "WebFace10" "log" $log_path
    # printf "mknoisedata "
    # python mknoisedata.py $method "WebFace10" "log" $log_path
    # for dataset in "WebFace10" # "WebFace10" # "WebFace50" "WebFace10_" "VGGFace10" "CelebA10" # "WebFace10" 
    for model in "resnet50" # "resnet18" "resnet50" "mobilenet_v1" "mobilenet_v2" "inception_v3" # 
    # for quality in  "20" # "75" "80" "85"
    # for sigma in '1' '2' '3' '4' '5' 
    do
        # printf "%s %s\n" $method $quality
        # printf "%s %s\n" $method $sigma
        # printf "%s %s\n" $method $dataset
        printf "%s %s\n" $method $model
        printf "noise_train "
        python noise_train.py $method $model $log $log_path
        printf "mknoisedata "
        python mknoisedata.py $method $model $log $log_path
        printf "train_model "
        python train_model.py $method $model $log $log_path # "WebFace10"
        # for rho_train in "0" "1" "2" "3" "4" #"0" "1" "2"
        # for model_ in "mobilenet_v2" "inception_v3" # "resnet18" 
        # do
        #     printf "%s " $model_
        #     python train_model.py $method $model_ "log" $log_path
        # done
        printf "\n"
    done
done
#python noise_train.py "GUE" 1 "log" "logjpg.txt"
# for method in "GUE"  #   "TUE"  "_clean" 
# do 
#     # printf "%s %s\n" $method "WebFace10"
#     # printf "noise_train "
#     # python noise_train.py $method "WebFace10" "log" "logtranbin.txt"
#     # for rho in "2" "4" #"0" "1" "2" "3" "4"
#     #for dataset in "WebFace10" "WebFace50" "WebFace10_" "VGGFace10" "CelebA10" 
#     # for model in "resnet18" "mobilenet" "inception_resnetv1"
#     for quality in "30" "40" "50" # "60" "70" "80" # "75" "80" "85"
#     do
#         printf "%s %s\n" $method $quality
#         printf "mknoisedata "
#         python mknoisedata.py $method $quality "log" "logjpg.txt"
#         printf "train_model "
#         # for rho_train in "0" "1" "2" "3" "4" 
#         # do
#             #printf "%s " $rho_train
#         python train_model.py $method "WebFace10" "log" "logjpg.txt"
#         # done
#         printf "\n"
#     done
# done