# gammas=(0.05 0.1 0.15)
# lamdas=(0.1 0.5 1.0)
gammas=(0.1)
lamdas=(0.1)

for gamma in "${gammas[@]}"; do
    for lamda in "${lamdas[@]}"; do
    echo "Running LMC algorithms for gamma = $gamma and lambda = $lamda"
        for ((i=5; i>=1; i--)); do
            echo "Running for number of mixtures n = $i"          
            python lmc_laplace.py --gamma_ula=$gamma --gamma_mala=$gamma --gamma_pula=$gamma --gamma_mla=$gamma --lamda=$lamda --alpha=5e-1 --n=$i --K=50000
            echo "LMC algorithms with gamma = $gamma and lambda = $lamda finished for $i mixtures"
            echo "------------------------------------------------------------------"
        done
    done
done
