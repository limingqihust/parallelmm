## 安装依赖

```shell
sudo apt update

sudo apt install -y git autoconf cmake zlib1g zlib1g-def openmpi-bin openmpi-common libopenmpi-dev gfortran python3-pip
```



## 安装anaconda3

```shell
cd [/path/to/conda]

bash Anaconda3-2024.06-1-Linux-x86_64.sh

source ~/.bashrc
```



## 创建python3.9环境

```shell
conda create -n py39 python=3.9
```



## 激活py39

```shell
conda activate py39
```



## 安装python依赖

```shell
pip3 install "numpy>=1.10.0" cython greenlet setuptools
```



## 安装charm4py

```shell
$ git clone https://github.com/UIUC-PPL/charm4py
$ cd charm4py
$ git clone https://github.com/UIUC-PPL/charm charm_src/charm
$ python3 setup.py install --mpi
```

