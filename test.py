from jax import random
#import neural_tangents as nt
#from neural_tangents import stax
#
#key1, key2 = random.split(random.PRNGKey(1), 2)
#x_train = random.normal(key1, (20, 32, 32, 3))
#y_train = random.uniform(key1, (20, 10))
#x_test = random.normal(key2, (5, 32, 32, 3))
#
#init_fn, apply_fn, kernel_fn = stax.serial(
#         stax.Conv(128, (3, 3)),
#         stax.Relu(),
#         stax.Conv(256, (3, 3)),
#         stax.Relu(),
#         stax.Conv(512, (3, 3)),
#         stax.Flatten(),
#         stax.Dense(1)
#     )
#
#res = kernel_fn(x_train, x_test, 'ntk')
#
#
#from jax import random
#from neural_tangents import stax
#
#init_fn, apply_fn, kernel_fn = stax.serial(
#            stax.Dense(512), stax.Relu(),
#                stax.Dense(512), stax.Relu(),
#                    stax.Dense(1)
#                    )
#
#key1, key2 = random.split(random.PRNGKey(1))
#x1 = random.normal(key1, (10, 100))
#x2 = random.normal(key2, (20, 100))
#
#kernel = kernel_fn(x1, x2, 'nngp')
#print(kernel.shape)




from jax.config import config
# Enable float64 for JAX
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit
import functools
from jax import random

import neural_tangents as nt
from neural_tangents import stax

key = random.PRNGKey(0)

# Network architecture described in 
# Shankar et al., Neural Kernels Without Tangents, 2020.
# https://arxiv.org/abs/2003.02237

def MyrtleNetwork(depth, W_std=np.sqrt(2.0), b_std=0.):
    layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
    width = 1
    activation_fn = stax.Relu()
    layers = []
    conv = functools.partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')

    layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][0]
    # layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    # layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][1]
    # layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][2]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 3

    layers += [stax.Flatten(), stax.Dense(10, W_std, b_std)]

    return stax.serial(*layers)


_, _, ker_fn1 = MyrtleNetwork(5)
# ker_fn1 = jit(ker_fn, static_argnums=(2,))

key1, key2 = random.split(key)
input_x1 = random.normal(key1, shape=(10, 32, 32, 3))
input_x2 = random.normal(key2, shape=(10, 32, 32, 3))

#kdd = ker_fn1(input_x1, None, 'nngp').block_until_ready()
kdd = ker_fn1(input_x1, None, 'nngp')
# print(kdd.shape)
print(kdd)



from jax import jit
import neural_tangents as nt
from neural_tangents import stax

key1, key2 = random.split(random.PRNGKey(1), 2)
x_train = random.normal(key1, (10, 32, 32, 3))
y_train = random.uniform(key1, (20, 10))
x_test = random.normal(key2, (10, 32, 32, 3))

init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Conv(128, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.Conv(20, (3, 3)),
        stax.Relu(),
        stax.Conv(20, (3, 3), strides=(2,2), padding='SAME'),
        stax.AvgPool((1, 1), strides=(1, 1)),
        stax.Flatten(),
        stax.Dense(1)
    )

n_samples = 200
# kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key1, n_samples)
# kernel_fn = jit(kernel_fn, static_argnums=(2,))
###kernel = ker_fn1(x_train, x_test, 'ntk')
kernel = kernel_fn(x_train, x_test, 'ntk')
# `kernel` is a tuple of NNGP and NTK MC estimate using `n_samples`.
print(kernel)



