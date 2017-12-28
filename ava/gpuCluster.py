import numpy as np
import time
import faiss
import sys


# Get command-line arguments

k = 4
ngpu = 1

d = 500
nb = 1000000
nq = 1000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.



t0 = time.time()


"Runs kmeans on one or several GPUs"
d = xb.shape[1]
clus = faiss.Clustering(d, k)
clus.verbose = True
clus.niter = 20

# otherwise the kmeans implementation sub-samples the training set
clus.max_points_per_centroid = 10000000

res = [faiss.StandardGpuResources() for i in range(ngpu)]

flat_config = []
for i in range(ngpu):
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = i
    flat_config.append(cfg)

if ngpu == 1:
    index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
else:
    indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
               for i in range(ngpu)]
    index = faiss.IndexProxy()
    for sub_index in indexes:
        index.addIndex(sub_index)

# perform the training
clus.train(xb, index)
centroids = faiss.vector_float_to_array(clus.centroids)

obj = faiss.vector_float_to_array(clus.obj)
print "final objective: %.4g" % obj[-1]

centroids.reshape(k, d)




t1 = time.time()

print "total runtime: %.3f s" % (t1 - t0)




