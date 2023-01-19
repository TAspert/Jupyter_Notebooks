<p>This program is based on Cellpose (more info: This notebook requires at https://github.com/MouseLand/cellpose)
To avoid potential conflicts, it is better to create a new environnement to run it.</p>

For example, in a conda prompt:

<code>conda create --name cellpose python=3.8</code>
<code>conda activate cellpose</code>


It requires the following packages:

<code>python -m pip install cellpose[gui]</code>
<code>python -m pip install matplotlib</code>
<code>conda install -c conda-forge jupyterlab</code>
<code>conda install scikit-image</code>
<code>pip install seaborn</code>
<code>pip install ipywidgets</code>
<code>pip install openpyxl</code>