python lmc.py --gamma_ula=7.5e-2 --gamma_mala=7.5e-2 --gamma_pula=7.5e-2 --gamma_ihpula=2.5e-2 --gamma_mla=7.5e-2 --K=10000 --n=5

python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=1 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=2 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=3 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=4 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=8e-2 --gamma_mala=8e-2 --gamma_pula=8e-2 --gamma_mla=8e-2 --lamda=5e-1 --alpha=5e-1 --n=5 --K=80000 --seed=0

python prox_lmc.py --gamma_pgld=5e-3 --gamma_myula=5e-3 --gamma_mymala=5e-3 --gamma_ppula=5e-3 --gamma_fbula=5e-3 --gamma_lbmumla=5e-3 --gamma0_ulpda=5e-3 --gamma1_ulpda=5e-3 --alpha=1.5e-1 --lamda=2.5e-1 --K=50000 --n=1
python prox_lmc.py --gamma_pgld=8e-2 --gamma_myula=8e-2 --gamma_mymala=8e-2 --gamma_ppula=8e-2 --gamma_fbula=8e-2 --gamma_lbmumla=8e-2 --gamma0_ulpda=8e-2 --gamma1_ulpda=8e-2 --alpha=1.5e-1 --lamda=2.5e-1 --t=100 --seed=0 --K=50000 --n=2

python prox_lmc_deconv.py --N=100 --tau=0.3 --sigma=0.75 --gamma_ulpda=15. --image='camera' --alg='ULPDA'
python prox_lmc_deconv.py --N=500 --gamma_mc=100. --gamma_me=15. --tau=0.3 --sigma=.75 --image='camera' --alg='MYULA'


