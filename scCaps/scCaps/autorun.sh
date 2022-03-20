echo "bash:$ cd scCaps"
cd scCaps
echo "bash:$ pthon Model_Training_docker.py"
python Model_Training_docker.py
echo "bash:$ python Model_Classify_docker.py --inputdata='scClass_data/CordBlood_modelC.h5ad'"
python Model_Classify_docker.py --inputdata='scClass_data/CordBlood_modelC.h5ad'
echo "bash:$ python Model_Classify_docker.py --inputdata='scClass_data/PBMC68k_modelC.h5ad'"
python Model_Classify_docker.py --inputdata='scClass_data/PBMC68k_modelC.h5ad'
