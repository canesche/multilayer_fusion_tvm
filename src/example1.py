import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm", host="llvm")

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

func = tvm.build(s, [A, B, C], tgt, name="myadd")

dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
mean_time = evaluator(a, b, c).mean
print("%f" % (mean_time))