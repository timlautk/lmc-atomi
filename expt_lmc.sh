gammas=(0.1 0.05 0.01)

for gamma in "${gammas[@]}"; do
echo "Running LMC algorithms for gamma = $gamma"
    for ((i=1; i<=5; i++)); do
        echo "Running for number of mixtures n = $i"          
        python lmc.py --gamma_ula=$gamma --gamma_mala=$gamma --gamma_pula=$gamma --gamma_ihpula=$gamma --gamma_mla=$gamma --K=10000 --n=$i
        echo "LMC algorithms with gamma = $gamma finished for $i mixtures"
        echo "------------------------------------------------------------------"
    done
done
