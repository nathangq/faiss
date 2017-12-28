import os
import time
import numpy as np
import pdb

import faiss

index = faiss.read_index('saved_index')



# 创建测试集
d = 500
nb = 1000000
nq = 1000
np.random.seed(1234)
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


# 这里要设置下gpu index 的参数，以把cpu index 转换为gpu index
# we need only a StandardGpuResources per GPU
res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()

# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
co.useFloat16 = True

index = faiss.index_cpu_to_gpu(res, 0, index, co)




print "benchmark"

for lnprobe in range(10):
    nprobe = 1 << lnprobe
    index.setNumProbes(nprobe)
    t0 = time.time()
    D, I = index.search(xq, 100)
    t1 = time.time()
    print "nprobe=%4d %.3f s recalls=" % (nprobe, t1 - t0)
    I

