import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Make tensor 
x = tf.constant(4,shape=(1,1),dtype= 'float32')
matrix = tf.constant([[1,2,3],[4,5,6]], shape =(1,6))

ones = tf.ones((3,3))
zeroes = tf.zeros((3,3))

identity_matrix = tf.eye(3)

norm_distrib = tf.random.normal((3,3), mean= 0, stddev = 1)
uniform_distrib = tf.random.uniform((1,3), minval= 0, maxval = 1)

range = tf.range(start =1,limit =10, delta=2 )
range = tf.cast(x, dtype = tf.float64)


# Mathematical operations 
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x,y)
z = tf.subtract(x,y)
division = tf.divide(x,y)
multiplication = tf.multiply(x,y)

dot_product = tf.tensordot(x,y, axes = 1)

exponent = x ** 5

mat_1 = tf.random.normal((2,3))
mat_2 = tf.random.normal((3,4))

mat_res = tf.matmul(mat_1,mat_2)

# Indexing 

x = tf.constant([0,0,1,1,2,2,3,4])

# print(x[1:])
# print(x[1:3])
# print(x[::2])
# print(x[::-1])

indeces = tf.constant([0,3])
x_ind = tf.gather(x,indeces)

x = tf.constant([[1,2],
                [3,4],
                [5,6]])

#print(x[0:2, :])


# Reshaping 
x = tf.range(9)
x = tf.reshape(x,(3,3))
x = tf.transpose(x, perm=[1,0])
print(x)


