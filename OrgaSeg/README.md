This program is based on Cellpose.
To avoid potential conflicts, it is better to create a new environnement to run it:

For example, in a conda prompt:

conda create --name cellpose python=3.8
conda activate cellpose


It requires the following packages:

python -m pip install cellpose[gui]
python -m pip install matplotlib
conda install -c conda-forge jupyterlab
conda install scikit-image
pip install seaborn
pip install ipywidgets
pip install openpyxl