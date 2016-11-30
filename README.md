#Intel Software Optimization for Theano*
---

This repo is dedicated to improving Theano performance when running on CPU, in particular Intel® Xeon® processors (HSW+).

Please refer to the document [Install_Guide.pdf](https://github.com/intel/theano/blob/master/Install_Guide.pdf) for the installation guide.


* This dl4mt-opti dev branch is used for optimization of dl4mt.
* Optimization includes Softmax, SoftmaxGrad etc., and other optimizations are still going on.

* Benchmark is based on dl4mt-tutorial session1(https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session1).
* Dataset is parallel corpus French-English from http://www.statmt.org/europarl/. <br />
  Need use the tools from dl4mt-tutorial to generate the files below:
     * datasets: europarl-v7.fr-en.en.tok, europarl-v7.fr-en.fr.tok
     * valid_datasets: newstest2011.en.tok, newstest2011.fr.tok
     * dictionaries: europarl-v7.fr-en.en.tok.pkl, europarl-v7.fr-en.fr.tok.pkl

* Statistics time is 1000 iters total time in session1.
* Software dependency:
     * intel-theano https://github.com/intel/Theano.git
     * intel-numpy  https://github.com/intel/numpy.git
     * MKL-(201609)
     * icc version 17.0.0
* Hardware dependency:
     * CPU: Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz

* theanorc configuration
<pre><code>[global]<br />
device = cpu<br />
floatX = float32<br />
cxx = icpc<br />
mode = FAST_RUN<br />
openmp = True<br />
openmp_elemwise_minsize = 10<br />
allow_gc = False<br />
<br />
[gcc]<br />
cxxflags = -qopenmp -march=native -O3 -qopt-report=3 -fno-alias -qopt-prefetch=2 -fp-trap=none<br />
[blas]<br />
ldflags=-lmkl_rt<br /></code></pre>

* Benchmark
![](https://raw.githubusercontent.com/intel/Theano/dl4mt-opti/doc/images/simple-encoder-decoder_benchmark.png)
