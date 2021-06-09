import numpy as onp
import jax
from jax import numpy as jnp, random, jit, lax
import flax
from flax import nn, optim
from tqdm import tqdm

import jax.experimental.stax as jstax
import neural_tangents.stax as nstax
from neural_tangents import monte_carlo_kernel_fn

# 1 make one-layer feature extractor with cos activation


PLACEHOLDER = 100

class CosFeatureMap(nn.Module):
    def apply(self, g0_x, width, bandwidth, filter_size, pooling):
        g_x = g0_x
        hw = g0_x.shape[1]
        prev_width = g_x.shape[3]

        # b = self.param(f'b_cos', (width,), jax.nn.initializers.uniform(scale=2*onp.pi) )
        # stdv = 1.0/(bandwidth * onp.sqrt(prev_width*hw*hw) )

        # strided convolution next
        s = int(pooling[2])
        f_x = flax.nn.Conv(g_x, features=width, kernel_size=(filter_size, filter_size),
                           strides=(s, s), padding='SAME')
        g_x = flax.nn.relu(f_x)
        return g_x

# 2 make exact ntk version 
def SCNTK(filter_sizes, poolings, widths, kernel_type):
    # 1 layer of cos feature mapping

    conv_layers = []
    for i, filter_size in enumerate(filter_sizes):
        if i == 0:
            continue
        s = int(poolings[i][2])
        conv_layers.append(nstax.Conv(widths[i], (filter_size, filter_size), strides=(s,s), padding='SAME'))
        conv_layers.append(nstax.Relu())
    if "monte_carlo" in kernel_type:
        model_init_fns = nstax.serial(
            *conv_layers,
            nstax.Flatten(),
            nstax.Dense(1, 1., 0.)
        )
    else:
        model_init_fns = nstax.serial(
            *conv_layers,
            nstax.AvgPool((1,1), strides=(1,1)),
            nstax.Flatten(),
            nstax.Dense(1, 1., 0.)
        )
    return model_init_fns


# 3 make monte-carol version of ntk


#activs not needed?
def compute_NTK(X_train, X_test, batch_size,
                widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings,
                use_components, kernel_type, model=None, vmap_Y_grad_fn=None): # kernel_type "exact_ntk", "monte_carlo100"
    if X_train.shape[1] == 3072:
        X_train = X_train.reshape([-1, 32, 32, 3])
        X_test = X_test.reshape([-1, 32, 32, 3])
    elif X_train.shape[1] == 1024:
        X_train = X_train.reshape([-1, 32, 32, 1])
        X_test = X_test.reshape([-1, 32, 32, 1])
    else:
        pass

    if model is None:
        module = CosFeatureMap.partial(width=widths[0], bandwidth=bandwidths[0], filter_size=filter_sizes[0], pooling=poolings[0])
        key = random.PRNGKey(1)
        d0 = list(X_train.shape[1:])
        x = jnp.ones([1] + d0)
        _, init_params = module.init(key, x)
        cos_feature_model = nn.Model(module, init_params)

        init_fn, apply_fn, kernel_fn = SCNTK(filter_sizes, poolings, widths, kernel_type)
        if "monte_carlo" in kernel_type:
            key2 = random.PRNGKey(2)
            kernel_fn = monte_carlo_kernel_fn(init_fn, apply_fn, key2, int(kernel_type[11:]))
            # import ipdb; ipdb.set_trace()
            print("using monte carlo")
        else:
            assert kernel_type == "exact_ntk"
    else:
        (cos_feature_model, kernel_fn) = model

    len_train, len_test = len(X_train), len(X_test)
    assert len_train == len_test
    if batch_size > len_train:
        batch_size = len_train

    K = onp.zeros([len_train, len_test], dtype=onp.float32)
    # if len_train != len_test:
    #     import ipdb; ipdb.set_trace()
    #     print()
    for i in range(0, len_train, batch_size):
        #[100, 32, 32, 1]
        batch_X_train = X_train[i:min(i+batch_size, len_train), ::]
        for j in range(0, len_test, batch_size):
            batch_X_test = X_test[j:min(j+batch_size, len_test), ::]
            # [100, 11, 11, 300]
            batch_Y_train = cos_feature_model(batch_X_train)
            # [100, 11, 11, 300]
            batch_Y_test = cos_feature_model(batch_X_test)
            # import ipdb; ipdb.set_trace()
            batch_K = kernel_fn(batch_Y_train, batch_Y_test, 'ntk')
            # batch_K = kernel_fn(batch_X_train, batch_X_test, 'ntk')
            # import ipdb; ipdb.set_trace() # check batch_K dimension
            # kernel quadratic kernel
            # batch_K = jnp.dot(batch_Y_train, jnp.transpose(batch_Y_test))
            K[i:min(i+batch_size, len_train), j:min(j+batch_size, len_test)] = onp.asarray(batch_K)
    return K, (cos_feature_model, kernel_fn), None

