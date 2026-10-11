gammas=(0.05 0.15 0.25)
lamdas=(0.25 0.5 1.0)

for gamma in "${gammas[@]}"; do
    for lamda in "${lamdas[@]}"; do
    echo "Running Proximal LMC algorithms for gamma = $gamma and lambda = $lamda"
        for ((i=5; i>=2; i--)); do
            echo "Running for number of mixtures n = $i"          
            python prox_lmc.py --gamma_pgld=$gamma --gamma_myula=$gamma --gamma_mymala=$gamma --gamma_ppula=$gamma --gamma_fbula=$gamma --gamma_lbmumla=$gamma --alpha=1.5e-1 --lamda=$lamda --K=50000 --n=$i
            echo "Proximal LMC algorithms with gamma = $gamma and lambda = $lamda finished for $i mixtures"
            echo "------------------------------------------------------------------"
        done
    done
done
