#### Create a conda environment with pythion3.8 and activate

```
conda create -n dig python=3.8
conda activate dig

```

#### Install pytorch 1.11 with with other matching libraries

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch -c conda-forge
if torchaudio PackageNotFound error occurs 
    - use "pip install torchaudio==0.11.0"
```

#### make sure to install compatible scatter ,sparser and finally torch geometric with compatible hardware

```
pip install torch-scatter==2.1.1 torch-sparse==0.6.15 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0
```

#### Install DIG 1.0.0 with steamlit and watchdog

```
pip install dive-into-graphs
pip install streamlit
pip install watchdog
```
#### Install other libraries

```
pip install networkx
pip install openai
```
After installation to run the app

```
streamlit run streamlit_app.py
```
