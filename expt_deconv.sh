python prox_lmc_deconv.py --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --niter_MAP=1000 --image='camera' --compute_MAP=True
python prox_lmc_deconv.py --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --image='camera' --alg='ULPDA'
python prox_lmc_deconv.py --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --image='camera' --alg='MYULA'

python prox_lmc_deconv.py --image='einstein' --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --niter_MAP=1000 --compute_MAP=True
python prox_lmc_deconv.py --image='einstein' --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --alg='ULPDA'
python prox_lmc_deconv.py --image='einstein' --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --alg='MYULA'