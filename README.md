# MathQuestion

Out of boredom, I decided to recall the following equation:

![](https://latex.codecogs.com/gif.latex?x^2=\underbrace{x&plus;x&plus;\cdots&plus;x}_{(x\text{&space;times})})

Then taking derivative on both side with respect to x:

![](https://latex.codecogs.com/gif.latex?\frac{d}{dx}x^2=\frac{d}{dx}[\underbrace{x&plus;x&plus;\cdots&plus;x}_{(x\text{&space;times})}])

![](https://latex.codecogs.com/gif.latex?2x=1&plus;1&plus;\cdots&plus;1=x)

2=1

I wrote a Tensorflow code to verify:

```python
import tensorflow as tf

x = tf.Variable(1)
r1 = x*x
g1 = tf.gradients(r1,x)
i, t = tf.constant(0), tf.constant(0)
c = lambda i, t: tf.less(i, x)
b = lambda i, t: [tf.add(i, 1), tf.add(t,x)]
_, r2 = tf.while_loop(c, b, [i,t])
g2 = tf.gradients(r2,x)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print sess.run([x,r1,r2,g1,g2])
```

This indeed evaluates to `g1=2` and `g2=1`. So why isn't 2=1? Where did I go wrong?
