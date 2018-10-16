# CTC 模型解码器
## decoder-cpp:c++版本解码器
### 项目结构
1. 采用cmake部署，在不同层次的文件夹内分别新建cmake文件  
2. 其中build， cmake, cmake-build-debug是编译文件夹
3. data文件夹是样本文件夹， include是头文件文件夹， lm，src是语言模型源文件文件夹，src是Beam Search算法源文件夹， test是测试文件夹
4. 后续如果调整为其他语言模型，将其头文件放入include文件夹，源文件置于和 decoder-cpp的根目录（和src同层次）
### KenLM语言模型的安装，依赖，训练，使用
#### 平台和项目源码
1. 安装平台：Ubuntu 18.04.1 LTS， 编译器版本： gcc version 7.3.0 (Ubuntu 7.3.0-16ubuntu3) 
2. [实际开发主页](https://kheafield.com/code/kenlm/)--推荐访问,[KenLM项目地址](https://github.com/kpu/kenlm)
#### 安装和依赖
    * 依赖
        * 必须： g++,bash
        * 推荐： Boost, zlib, bzip2, cmake
        * 非必须： Eigen3, OpenMP,  xz
    * 安装
        * 依赖安装 参考 https://kheafield.com/code/kenlm/dependencies/)
        * g++和bash默认已安装
        * boost库的安装依赖于zlib和zip2,因此先安装zlib和bzip2
        * kenlM作者提供的zlib的安装有点小问题，正确的安装流程如代码1所示
        * 同样bzip2的链接下载后文件也不完整，正确的安装流程如代码2所示
        * Boost库的安装直接按照
代码1：zlib安装
``` bash
// wget http://zlib.net/zlib-1.2.8.tar.gz 1.2.8版本的地址已无法下载，因此换用最新版本的1.2.11
wget https://zlib.net/zlib-1.2.11.tar.gz
tar xzf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure --prefix=$PREFIX --libdir=$LIBDIR
make -j4
make install
make clean
./configure --prefix=$PREFIX --libdir=$LIBDIR --static
make -j4
make install 
        
```
代码2： bzip2安装
```bash
//wget http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz
//tar xzvf bzip2-1.0.6.tar.gz
git clone https://github.com/enthought/bzip2-1.0.6.git
cd bzip2-1.0.6/
#Compile and install libbz2.a (static library)
make
make install PREFIX=$PREFIX
mkdir -p $LIBDIR
#Note this may be the same file; you can ignore the error
mv $PREFIX/lib/libbz2.a $LIBDIR 2>/dev/null
#Compile and install libbz2.so (dynamic library)
make clean
make -f Makefile-libbz2_so
cp libbz2.so.* $LIBDIR
ln -sf libbz2.so.1.0 $LIBDIR/libbz2.so
```   
#### 模型的安装
```bash
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```
#### 模型的训练
* 语料来源： 这里使用的是搜狗，新浪微博，以及一个通用英文语料均位于offline11服务器上
* 语料预处理： 把你要训练的文本分好词-用空格隔开，如果是做基于字的模型，就把模型的每个字用空格隔开；把中文，英文的语料做一个拼接，合并到同一个文件之中
   ，拼接程序位于decoder-python/src/merge_corpus.py
* 训练： 打开bash，进入kenlm的安装文件夹
```bash
cd kenlm/build
bin/lmplz -o 2 --verbose_header --text /home/augustus/Documents/decoder-python/data/mixed.txt --arpa /home/augustus/Documents/decoder-python/data/mixed.arpa
// -o 后的参数代表要训练的N-gram模型，一般去2，3即可， --text后是语料存储位置， --arpa后是训练好的模型输出位置
bin/build_binary text.arpa text.binary
// 将训练好的模型从arpa格式转换为bin格式，这会大大减少其体积，降低后续调用时的内存占用
```
#### 模型的使用
* python接口的使用

```python
import kenlm
model = kenlm.Model('sougou.bin')
model.score('微 信', bos=False, eos=False)
//score函数输出的是对数概率，即log10(p('微 信'))，其中字符串可以是gbk，也可以是utf-8
//bos=False, eos=False意思是不自动添加句首和句末标记符
```

* C++接口的使用
```c++
#include "lm/model.hh"
#include <iostream>
#include <string>

int main() {
  using namespace lm::ngram;
  //Model model("file.arpa");
  Model model{"/home/augustus/Documents/decoder/decoder-python/data/mixed.bin"};
  //建议使用花括号初始化模型对象，尤其是在类内初始化模型对象的时候，如果使用普通的括号初始化，会导致一个很麻烦的的bug
  State state(model.BeginSentenceState()), out_state;
  const Vocabulary &vocab = model.GetVocabulary();
  std::string word;
  while (std::cin >> word) {
    std::cout << model.Score(state, vocab.Index(word), out_state) << '\n';
    state = out_state;
  }
}
```
### BeamSearch解码器的使用
1. 单线程测试用例见decoder-cpp/test/main_single.cpp
2. 多线程测试用例见decoder-cpp/test/main_multi_threads.cpp
3. 更换测试用例需要在decoder-cpp/CMakeLists.txt中更新
add_executable(main test/main_multi_threads.cpp)测试入口代码的路径
4. 编译命令
```bash
cd decoder/decoder-cpp/build
cmake ..
make
cd bin
./main
```

## decoder-python:算法原型开发
整体使用与C++版相同
