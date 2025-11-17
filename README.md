# xec-ml-wl
Analysis using machine learning for MEG II liquid xenon detector 

First time in merlin7:
module load anaconda/2024.08
conda env create -f xec-ml-wl-gpu.yml
chmod +x start_jupyter_xec.sh start_jupyter_xec_gpu.sh

(wait for a while)
./start_jupyter_xec.sh 8888
or
./start_jupyter_xec_gpu.sh 8888

(when using gpu node, after lunching jupyter,)
1. check which node you are allocated (ex. gpu001)
2. ssh -N -L 8888:localhost:8888 -J <user_name>@login001 <user_name>@gpuXXX (for XXX put the allocated node)

Now you can click the link shown in the console to start JupyterLab.
When using gpu nodes, you need to enter the token also shown in the console.

You can use MLflow and TensorBoard to automatically log the parameters and results.
To see the tracks:
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000 &
tensorboard --logdir runs --host 0.0.0.0 --port 6006 &
-> then open http://localhost:5000 and http://localhost:6006
