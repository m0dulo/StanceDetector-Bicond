# StanceDetector-Bicond
Stance Detector with Bidirectional Conditional Encoding implemented in [N3LDG](http://xbna.pku.edu.cn/EN/10.13209/j.0479-8023.2018.065).

N3LDG is a lightweight neural network library for natural language processing.
## Model
see [the compution graph](https://github.com/m0dulo/StanceDetector-Bicond/blob/master/StanceDetector/model/ComputionGraph.h)
## Requirements
* Intel® MKL [[link]](https://software.intel.com/en-us/mkl)
* boost [[link]](https://www.boost.org/)
* cmake [[link]](https://cmake.org/)

## Usage
```bash
mkdir build
cmake .. -DUSE_CUDA=0 -DTEST_CUDA=0 -DCMAKE_BUILD_TYPE=release -DMEMORY=custom -DMKL=1
make
../bin/StanceDetector -train ../data/stanceResponse.train.txt -dev ../data/stanceResponse.dev -test ../data/stanceResponse.test.txt -option ../data/option.debug
```
## Parameter tuning
For details, see [老中医——基于网格搜索的自动调参程序](https://github.com/m0dulo/laozhongyi)

## Reference
Stance Detection with Bidirectional Conditional Encoding [[paper link]](https://www.aclweb.org/anthology/D16-1084)
