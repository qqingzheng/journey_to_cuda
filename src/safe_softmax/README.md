# 朴素的Softmax实现

在朴素的Softmax中，计算是row-wise的，因此只能在行上做并行，并行度是比较低的。

朴素的Softmax在行数过多或者列数过多的时候都会遇到问题。因为一行会分配个给一个thread执行，那么行过多时，一个block中分配的thread数会过多，导致无法分配。