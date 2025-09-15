实现CUDA的热门算子


一维数组相较多维数组的好处:
1.存储开销少，一维数组只需要一个指针，多维数组除了最后一维外都需要存指针，需要D1*D 2*..*D(N-1)个指针
2.Malloc友好，多维数组需要多次malloc，并且释放后还会造成内存碎片
3.访问效率高-维数组算出index后只需要-次dereference就能拿到对应元素，多维数组需要N次4.缓存友好,维数组内存连续，缓存命中率高
5.方便做向量化，向量化指令(比如cpu上的avx和tensor core上的mma)对内存布局都有严格要求，不连续的内存基本告别这些指令优化
6.对优化更友好，cuda编程到最后基本都是在优化内存访问，内存不连续时访问效率是非常低的


_____________________________________________________
1.gemm

代码学习来源：https://zhuanlan.zhihu.com/p/657632577

下面是对
“ii) 在TM = TN = 8的情况下，无论矩阵A还是矩阵B，从Shared Memory中取数时需要取连续的8个数，即便用LDS.128指令一条指令取四个数，也需要两条指令，由于一个线程的两条load指令的地址是连续的，那么同一个Warp不同线程的同一条load指令的访存地址就是被间隔开的，便存在着 Bank Conflict。”
以及贴主解决方案的解释。

首先本人认为贴主没有表达的一个点是它这个做法其实并没有彻底的解决bankconflict。但是确实做到了很大程度缓解bankconflict。要说有没有比它做法更好的，那肯定有，但是性能还真不一定有贴主这版本性能高。贴主是在保证float4的前提下做bank conflict处理的。事实上float4是比处理bank conflict快的。但是float4有一个问题就是你所开辟的空间必须是4的倍数，否则就报错。所以常规处理bank conflict的方式padding其实作用不是很大。

而贴主在处理bank问题时是将一次读取8个连续地址给差开来，下面是基于贴主的解释再增加一层解释，方便更好理解：
如果一次读4个float就已经让某些线程落在相同的bank，那么紧随其后的下一条读(往往地址上连续+4)还会产生相似的碰撞，于是连续两条指令都可能发生bank conflict。
这在某些矩阵取列向量或者小tile加载的场景下尤为常见：要想并行取列，就被迫用行主序的二维数组里“非连续”的列，导致warp里各线程访问模式非常容易出现规律性冲突。

float4优先级是比bank conflict高的！！！！！要尽量确保float4的前提下实现bank conflict。

学习感悟（学不明白可以参考的思路）：一开始每行代码学习，自己手推cuda中的矩阵乘法计算，因为该算子需要算局部再累加，所以可能会给读者盘逻辑的时候绕懵，这里建议复习一下线性代数，尤其是矩阵分块乘法这块，把关于矩阵的乘法抽象化，会更有利于自己盘逻辑，复现代码。

_____________________________________________________
2.reduce
这里写了两个版本来做比较：

#################
m_reduce函数来源：https://zhuanlan.zhihu.com/p/426978026
#################
book_reduce函数来源：CUDA编程：基础与实践_樊哲勇
#################

reduce其实本身逻辑也不难，相比前面的gemm没啥好讲的

代码中做了十次比较
m_reduce elasped_time:0.2132
m_reduce compute sum:262144.000000,true compute sum:262144
book_reduce elasped_time:0.1088
book_reduce compute sum:262144.000000,true compute sum:262144
m_reduce elasped_time:0.1282
m_reduce compute sum:262144.000000,true compute sum:262144
book_reduce elasped_time:0.0736
book_reduce compute sum:262144.000000,true compute sum:262144
m_reduce elasped_time:0.1262
m_reduce compute sum:262144.000000,true compute sum:262144
book_reduce elasped_time:0.0688
book_reduce compute sum:262144.000000,true compute sum:262144
m_reduce elasped_time:0.1254
m_reduce compute sum:262144.000000,true compute sum:262144
book_reduce elasped_time:0.0575
book_reduce compute sum:262144.000000,true compute sum:262144
m_reduce elasped_time:0.1254
m_reduce compute sum:262144.000000,true compute sum:262144
book_reduce elasped_time:0.0701
book_reduce compute sum:262144.000000,true compute sum:262144

相比之下CUDA编程：基础与实践_樊哲勇里面的实现方法是完胜的。个人认为主要是m_reduce的做法考虑过于局部了，可以发现，用书里的做法只用执行两次核函数即可，而m_reduce里面执行核函数的次数是不确定的，其次也没有过好的考虑自己的硬件资源是否支持这样的线程数量开销，书里通过跨步长一次将整体的reduce浓缩到一个网格中的做法很好的考虑了性能，查看连接里的帖子可以发现m_reduce的实现并没有考虑最后的聚合，注意力只放在了大头。

不过二者在reduce上都提供了不错的思路。

_____________________________________________________
3.relu
这个没啥好说的，不过我没用上float4。加上这个应该会更快，也不难写，懒得写。


_____________________________________________________
4.rmsnorm
就固定32个线程，然后每次float4读取。这是因为之前刚发现线程数量也是有说法的，不要设置太多，好像block的极限是(1024,1024,64)

_____________________________________________________
5.softmax
实现了基础版的softmax和online softmax；


_____________________________________________________
5.mha
写的是GHA，根据flashattention来写，加入float4.
