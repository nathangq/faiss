import os
import time
import numpy as np
import pdb

import faiss


# 创建数据集
d = 500
nb = 1000000
nq = 1000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
xt = np.random.random((nb, d)).astype('float32')
xt[:, 0] += np.arange(nb) / 1000.
k = 4



# we need only a StandardGpuResources per GPU
res = faiss.StandardGpuResources()


#################################################################
#  Exact search experiment 暴力搜索
#################################################################

print "============ Exact search"

flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0

index = faiss.GpuIndexFlatL2(res, d, flat_config)

print "add vectors to index"

index.add(xb)

print "warmup"


print "benchmark"

for lk in range(11):
    k = 1 << lk
    t0 = time.time()
    D, I = index.search(xq, k)
    D
    I
    t1 = time.time()
    # the recall should be 1 at all times
    print "k=%d %.3f s" % (
        k, t1 - t0)



#################################################################
#  Approximate search experiment 近似搜索，快
#################################################################

print "============ Approximate search"


index = faiss.index_factory(d, "IVF4096,Flat")

# faster, uses more memory
# index = faiss.index_factory(d, "IVF16384,Flat")

co = faiss.GpuClonerOptions()

# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
co.useFloat16 = True

#转换为 gpu index
index = faiss.index_cpu_to_gpu(res, 0, index, co)

print "train"

t0 = time.time()
index.train(xt)
t1 = time.time()
print "train time =%.3f s" % (t1 - t0)

print "add vectors to index"

t0 = time.time()
index.add(xb)
t1 = time.time()
print "train time =%.3f s" % (t1 - t0)


# 这里如果要保存index，必须要把gpu index 转换为 cpu index
t0 = time.time()
populated_index_path = 'saved_index'
print "save index"
index_cpu = faiss.index_gpu_to_cpu(index)
t1 = time.time()
print "train time =%.3f s" % (t1 - t0)
faiss.write_index(index_cpu, populated_index_path)


print "warmup"

print "benchmark"

for lnprobe in range(10):
    nprobe = 1 << lnprobe
    index.setNumProbes(nprobe)
    t0 = time.time()
    D, I = index.search(xq, 100)
    t1 = time.time()
    print "nprobe=%4d %.3f s recalls=" % (nprobe, t1 - t0)
    I

 

