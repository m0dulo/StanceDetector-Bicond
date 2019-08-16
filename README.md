# StanceDetector-Bicond
Stance Detector with Bidirectional Conditional Encoding implemented in [N3LDG](http://xbna.pku.edu.cn/EN/abstract/abstract3327.shtml).

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

## Reference
Stance Detection with Bidirectional Conditional Encoding [[paper link]](https://www.aclweb.org/anthology/D16-1084)
