# Code and Data for Paper "Vision-Language Navigation with Multi-granularity Observation and Auxiliary Reasoning Tasks" 

## A updated version of AuxRN

## Environment Installation
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

Download image features for environments:
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

Python requirements: Need python3.6 (python 3.5 should be OK since I removed the allennlp dependencies)
```
pip install -r python_requirements.txt
```

Install Matterport3D simulators:
```
git submodule update --init --recursive 
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make -j8
```

## Code

### Train Speaker
```
bash run/speaker.bash 0
```
0 is the id of GPU. It will train the speaker and save the snapshot under snap/speaker/

### Train Baseline
```
bash run/baseline.bash 0
```

### Train MG-AuxRN 
```
bash run/mg-auxrn.bash 0
```

### Top-down Visualization
![bird_view](./visualizations/bird_view.png)
