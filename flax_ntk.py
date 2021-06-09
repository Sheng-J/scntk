import numpy as onp
import jax
from jax import numpy as jnp, random, jit, lax
import flax
from flax import nn, optim
from tqdm import tqdm


class Alice(nn.Module):
    def apply(self, g0_x, g0_mode, out_mode, parameterization, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings):
        assert g0_x.shape[1] == g0_x.shape[2]
        assert (len(widths) == len(g_activs)) and (len(g_activs)==len(bandwidths)) and (len(bandwidths)==len(init_stdvs)) and (len(init_stdvs)==len(filter_sizes))
        assert (len(poolings)==len(widths))
        hw = g0_x.shape[1]
        ch = g0_x.shape[3]
        if g0_mode is None:
            g_x = g0_x
        elif g0_mode[:2] == "ap":
            s = int(g0_mode[2])
            g_x = nn.avg_pool(g0_x, window_shape=(s,s), strides=(s,s))
        elif g0_mode[:2] == "mp":
            s = int(g0_mode[2])
            g_x = nn.max_pool(g0_x, window_shape=(s,s), strides=(s,s))
        elif "pretrain" in g0_mode:
            pass
        else:
            raise NotImplementedError()

        for i, width in enumerate(widths):
            if i == 0:
                prev_width = g_x.shape[3]
            else:
                prev_width = widths[i-1]

            if g_activs[i] == "cos":
                b = self.param(f'b{i}', (width,), jax.nn.initializers.uniform(scale=2*onp.pi))
                stdv = 1.0/(bandwidths[i] * onp.sqrt(prev_width*hw*hw)) #1/(bandwidths[i]*sqrt(32*32*3)) sqrt(3072)=55
            elif g_activs[i] == "relu":
                stdv = init_stdvs[i]/onp.sqrt(prev_width*hw*hw)

            if parameterization == "ntk":
                if (poolings[i][:2] == "st"):
                    s = int(poolings[i][2])
                    f_x = flax.nn.Conv(g_x, features=width, kernel_size=(filter_sizes[i], filter_sizes[i]),
                            kernel_init=jax.nn.initializers.normal(stddev=1.), 
                            strides=(s, s), padding="SAME", bias=False
                            )
                else:
                    f_x = flax.nn.Conv(g_x, features=width, kernel_size=(filter_sizes[i], filter_sizes[i]),
                            kernel_init=jax.nn.initializers.normal(stddev=1.), padding='SAME', bias=False)
                f_x =  stdv * f_x
            elif parameterization == "standard":
                f_x = flax.nn.Conv(g_x, features=width, kernel_size=(filter_sizes[i], filter_sizes[i]),
                        kernel_init=jax.nn.initializers.normal(stddev=stdv), bias=False)
            else:
                raise NotImplementedError()

            if g_activs[i] == "cos":
                f_x = f_x + b
                g_x = jnp.cos(f_x)
            elif g_activs[i] == "relu":
                g_x = flax.nn.relu(f_x)
            else:
                raise NotImplementedError()

            if (poolings[i] is None) or (poolings[i][:2] == "st"):
                pass
            elif poolings[i] == "gap":
                g_x = nn.avg_pool(g_x, window_shape=(hw, hw), strides=(hw, hw))
            elif poolings[i][:2] == "ap":
                s = int(poolings[i][2])
                g_x = nn.avg_pool(g_x, window_shape=(s, s), strides=(s, s))
            elif poolings[i][:2] == "mp":
                s = int(poolings[i][2])
                g_x = nn.max_pool(g_x, window_shape=(s, s), strides=(s, s))
            else:
                raise NotImplementedError()

        g_x = g_x.reshape((g_x.shape[0], -1))
        if out_mode == "features":
            return g_x
        elif out_mode == "prediction":
            f_L1_x = flax.nn.Dense(g_x, features=1, kernel_init=jax.nn.initializers.normal(stddev=1.), bias=False)
            f_L1_x = (1/onp.sqrt(g_x.shape[1])) * f_L1_x
            return f_L1_x.squeeze()


@jit
def vmap_Y_compute(model, X):
    X = jnp.expand_dims(X, 0)
    return model(X)


def compute_NTK(X_train, X_test, batch_size,
                widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components,
                kernel_type, model=None, vmap_Y_grad_fn=None):
    # # X_train X_test both [100, 1024]
    # if X_train.shape[1] == 3072:
    #     X_train = X_train.reshape([-1, 32, 32, 3])
    #     X_test = X_test.reshape([-1, 32, 32, 3])
    # elif X_train.shape[1] == 1024:
    #     X_train = X_train.reshape([-1, 32, 32, 1])
    #     X_test = X_test.reshape([-1, 32, 32, 1])
    # else:
    #     pass

    if kernel_type == "ntk":
        if model is None:
            module = Alice.partial(g0_mode=None, out_mode="prediction", parameterization="ntk",
                                widths=widths,
                                g_activs=g_activs, bandwidths=bandwidths,
                                init_stdvs=init_stdvs, filter_sizes=filter_sizes, poolings=poolings)
            # module = Alice.partial(g0_mode=None, out_mode="prediction", parameterization="ntk", widths=[300, 300, 300],
            #                        g_activs=["cos", "relu", "relu"], bandwidths=[bandwidth, None, None],
            #                        init_stdvs=[None, 1.0, 1.0], filter_sizes=[3,3,3], poolings=[None, None, "mp3"])
            vmap_Y_grad_fn = jax.vmap(jax.grad(vmap_Y_compute), in_axes=(None, 0))
            key = random.PRNGKey(1)
            d0 = list(X_train.shape[1:])
            x = jnp.ones([1] + d0)
            _, init_params = module.init(key, x)
            model = nn.Model(module, init_params)
        else:
            pass

        len_train, len_test = len(X_train), len(X_test)
        assert len_train == len_test
        if batch_size > len_train:
            batch_size = len_train

        K = onp.zeros([len_train, len_test], dtype=onp.float32)
        # if len_train != len_test:
        #     import ipdb; ipdb.set_trace()
        #     print()
        for i in range(0, len_train, batch_size):
            batch_X_train = X_train[i:min(i+batch_size, len_train), ::]
            for j in range(0, len_test, batch_size):
                batch_X_test = X_test[j:min(j+batch_size, len_test), ::]

                batch_X_train_grads = vmap_Y_grad_fn(model, batch_X_train)
                batch_X_test_grads = vmap_Y_grad_fn(model, batch_X_test)

                batch_K = None

                ##############
                # NOTE Debug #
                ##############
                # all_params_keys = batch_X_train_grads.params.keys()
                # import ipdb; ipdb.set_trace()

                for name in batch_X_train_grads.params.keys():
                    if "b" in name:
                        continue

                    ##############
                    # NOTE Debug removing the first layer NTK component #
                    ##############
                    layer_id = int(name[-1])
                    if layer_id == 0:
                        continue

                    # Loop one train grad w.r.t all test grads
                    # import ipdb; ipdb.set_trace()
                    train_grads_W = (batch_X_train_grads.params[name]["kernel"]).reshape([batch_size,-1])  # [m1, d]
                    test_grads_W = (batch_X_test_grads.params[name]["kernel"]).reshape([batch_size, -1]) #[m2, d]

                    # [m1, m2]
                    # if train_grads_W.shape[1] != test_grads_W.shape[1]:
                    #     import ipdb; ipdb.set_trace()
                    #     print()

                    K_add = jnp.dot(train_grads_W, jnp.transpose(test_grads_W))

                    # K_rows = []
                    # for m in range(batch_size):
                    #     K_row = jnp.dot(test_grads_W, train_grads_W[m,:])
                    #     K_rows.append(K_row)
                    # # [mb_train, mb_test]
                    # K_add = jnp.stack(K_rows)

                    # sum for different paramters gradients
                    if batch_K is None:
                        batch_K = K_add
                    else:
                        batch_K = batch_K + K_add
                # set for each batch size 
                K[i:min(i+batch_size, len_train), j:min(j+batch_size, len_test)] = onp.asarray(batch_K)
        return K, model, vmap_Y_grad_fn
    elif kernel_type == "rf":
        module = Alice.partial(g0_mode=None, out_mode="features", parameterization="ntk",
                            widths=widths,
                            g_activs=g_activs, bandwidths=bandwidths,
                            init_stdvs=init_stdvs, filter_sizes=filter_sizes, poolings=poolings)
        vmap_Y_grad_fn = jax.vmap(jax.grad(vmap_Y_compute), in_axes=(None, 0))
        key = random.PRNGKey(1)
        d0 = list(X_train.shape[1:])
        x = jnp.ones([1] + d0)
        _, init_params = module.init(key, x)
        model = nn.Model(module, init_params)

        len_train, len_test = len(X_train), len(X_test)
        assert len_train == len_test
        if batch_size > len_train:
            batch_size = len_train

        K = onp.zeros([len_train, len_test], dtype=onp.float32)
        # if len_train != len_test:
        #     import ipdb; ipdb.set_trace()
        #     print()
        for i in range(0, len_train, batch_size):
            batch_X_train = X_train[i:min(i+batch_size, len_train), ::]
            for j in range(0, len_test, batch_size):
                batch_X_test = X_test[j:min(j+batch_size, len_test), ::]
                batch_Y_train = model(batch_X_train)
                batch_Y_test = model(batch_X_test)
                # import ipdb; ipdb.set_trace()

                batch_K = jnp.dot(batch_Y_train, jnp.transpose(batch_Y_test))
                K[i:min(i+batch_size, len_train), j:min(j+batch_size, len_test)] = onp.asarray(batch_K)
        return K



def compute_NTK_MMD(X_train, X_test, batch_size,
                    widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components,
                    kernel_type, model=None, vmap_Y_grad_fn=None):
    # if X_train.shape[1] == 3072:
    #     X_train = X_train.reshape([-1, 32, 32, 3])
    #     X_test = X_test.reshape([-1, 32, 32, 3])
    # elif X_train.shape[1] == 1024:
    #     X_train = X_train.reshape([-1, 32, 32, 1])
    #     X_test = X_test.reshape([-1, 32, 32, 1])
    # else:
    #     pass

    if kernel_type == "ntk":
        if model is None:
            module = Alice.partial(g0_mode=None, out_mode="prediction", parameterization="ntk",
                                widths=widths,
                                g_activs=g_activs, bandwidths=bandwidths,
                                init_stdvs=init_stdvs, filter_sizes=filter_sizes, poolings=poolings)
            vmap_Y_grad_fn = jax.vmap(jax.grad(vmap_Y_compute), in_axes=(None, 0))
            key = random.PRNGKey(1)
            d0 = list(X_train.shape[1:])
            x = jnp.ones([1] + d0)
            _, init_params = module.init(key, x)
            model = nn.Model(module, init_params)
        else:
            pass

        len_train, len_test = len(X_train), len(X_test)
        assert len_train == len_test
        if batch_size > len_train:
            batch_size = len_train

        # S_P
        sum_phi_x_squared = 0
        X_grad_vec = []
        for i in range(0, len_train, batch_size):
            # batch_X_train_grads_dict= {}
            batch_X_train = X_train[i:min(i+batch_size, len_train), ::]
            batch_X_train_grads = vmap_Y_grad_fn(model, batch_X_train)
            batch_param_vec = []
            for name in batch_X_train_grads.params.keys():
                if "b" in name:
                    continue
                layer_id = int(name[-1])
                if layer_id == 0:
                    continue
                # [m1, d]
                train_grads_W = (batch_X_train_grads.params[name]["kernel"]).reshape([batch_size,-1])  
                sum_phi_x_squared += jnp.sum(train_grads_W * train_grads_W)
                batch_param_vec.append(train_grads_W)
            # [m_batch , d]
            batch_param_vec = jnp.concatenate(batch_param_vec, axis=1)
            X_grad_vec.append(batch_param_vec)
        # [n_samples, d]
        X_grad_vec = jnp.concatenate(X_grad_vec, axis=0)
        sum_X_grad_vec = jnp.sum(X_grad_vec, axis=0)
        squared_sum_phi_x = jnp.sum(sum_X_grad_vec*sum_X_grad_vec)
        mmd_x = (squared_sum_phi_x-sum_phi_x_squared)/(len_train*(len_train-1))

        # S_Q
        sum_phi_y_squared = 0
        Y_grad_vec = []
        for i in range(0, len_test, batch_size):
            batch_X_test = X_test[i:min(i+batch_size, len_test), ::]
            batch_X_test_grads = vmap_Y_grad_fn(model, batch_X_test)
            batch_param_vec = []
            for name in batch_X_test_grads.params.keys():
                if "b" in name:
                    continue
                layer_id = int(name[-1])
                if layer_id == 0:
                    continue
                # [m1, d]
                test_grads_W = (batch_X_test_grads.params[name]["kernel"]).reshape([batch_size,-1])  
                sum_phi_y_squared += jnp.sum(test_grads_W * test_grads_W)
                batch_param_vec.append(test_grads_W)
            # [m_batch , d]
            batch_param_vec = jnp.concatenate(batch_param_vec, axis=1)
            Y_grad_vec.append(batch_param_vec)
        # [n_samples, d]
        Y_grad_vec = jnp.concatenate(Y_grad_vec, axis=0)
        sum_Y_grad_vec = jnp.sum(Y_grad_vec, axis=0)
        squared_sum_phi_y = jnp.sum(sum_Y_grad_vec*sum_Y_grad_vec)
        mmd_y = (squared_sum_phi_y-sum_phi_y_squared)/(len_test*(len_test-1))

        mmd_xy_prod_sum = jnp.sum(sum_X_grad_vec*sum_Y_grad_vec)
        mmd_xy_sum_prod = jnp.sum(X_grad_vec * Y_grad_vec)
        mmd_xy = 2*(mmd_xy_prod_sum - mmd_xy_sum_prod)/(len_train*(len_test-1))
        mmd = mmd_x + mmd_y - mmd_xy
    return mmd, model, vmap_Y_grad_fn
