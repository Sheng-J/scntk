# numpy & jax only
# Experiment type: hardware speed time vs type 1 null hypo accep acc P=Q  vs type 2 rejection accuracy when P!=Q
# datasets type: CIFAR10 vs CIFAR10.1, MNIST vs CIFAR10,  ... etc
# method type: NTK vs SRF vs ... vs , also optimize test power

# MMD computation time 
import argparse
import os
import numpy as onp
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from flax_ntk import compute_NTK, compute_NTK_MMD
from mmd_d import compute_deep_K, compute_gaussian_K
import timeit


alpha = 0.05


def main(args, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type):
    # set up S_P
    if args.S_P_data == "MNIST":
        os.makedirs("./data/mnist", exist_ok=True)
        dataloader_FULL = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=60000,
            shuffle=True,
        )
        for imgs, _ in dataloader_FULL:
            P_data = imgs
    elif args.S_P_data == "CIFAR10":
        pass

    # # set up S_Q.
    # if args.S_Q_data == "GANMNIST":
    #     GANMNIST = pickle.load(open('./Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    #     # Collect fake MNIST images
    # elif args.S_Q_data == "CIFAR10_1":
    #     pass
    # elif args.S_Q_data == "MNIST":
    #     dataloader_FULL_te = torch.utils.data.DataLoader(
    #         datasets.MNIST(
    #             "./data/mnist",
    #             train=False,
    #             download=True,
    #             transform=transforms.Compose(
    #                 [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    #             ),
    #         ),
    #         batch_size=10000,
    #         shuffle=True,
    #     )
    #     for imgs, _ in dataloader_FULL_te:
    #         Q_data = imgs

    # if method == optimized
    model = None
    grad_fn = None
    Results = onp.zeros([args.n_seeds])
    for i_seed in range(args.n_seeds):
        # train data chosen (test data pool also chosen)
        torch.manual_seed(i_seed*19 + args.n_samples)
        onp.random.seed(seed=1102*(i_seed+10)+args.n_samples)

        P_train_data = []
        P_ind_all = onp.arange(4000)
        P_ind_tr = onp.random.choice(4000, args.n_samples, replace=False)
        P_ind_te_left = onp.delete(P_ind_all, P_ind_tr)
        # for i in P_ind_tr:
        #     P_train_data.append([P_data[i], None])
        
        # train_dataloader = torch.utils.data.DataLoader(P_train_data, batch_size=args.bach_size=, shuffle=True)
        Q_ind_all = onp.arange(4000)
        Q_ind_tr = onp.random.choice(4000, args.n_samples, replace=False)
        Q_ind_te_left = onp.delete(Q_ind_all, Q_ind_tr)
        if args.S_Q_data == "GANMNIST":
            GANMNIST = pickle.load(open('/h/sheng/NTKTST/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
            Q_te_left = GANMNIST[0][Q_ind_te_left]


        chosen_hypothesis_vector = onp.zeros(args.n_tests)
        for i_test in range(args.n_tests):
            onp.random.seed(seed=1102*(i_seed+1) + args.n_samples)
            # test data chosen from test data pool
            P_ind_te = onp.random.choice(len(P_ind_te_left), args.n_samples, replace=False)
            S_P = P_data[P_ind_te_left[P_ind_te]]
            S_P = onp.reshape(S_P.numpy(), [args.n_samples, -1])

            Q_ind_te = onp.random.choice(len(Q_ind_te_left), args.n_samples, replace=False)
            S_Q = Q_te_left[Q_ind_te]
            S_Q = onp.reshape(S_Q, [args.n_samples, -1])

            if args.exp_type == "complexity":
                start_time = timeit.default_timer()
                nx, ny =  len(S_P), len(S_Q)
                assert nx == ny
                if args.method == "fixed_ntk":
                    # [1] MMD kernel compute mode
                    Kx, model, grad_fn = compute_NTK(S_P, S_P, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                    Ky, _, __ = compute_NTK(S_Q, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                    Kxy, _, __ = compute_NTK(S_P, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                    sum_xx = (onp.sum(Kx) - onp.sum(onp.diag(Kx)) )/(nx*(nx-1))
                    sum_yy = (onp.sum(Ky) - onp.sum(onp.diag(Ky)) )/(ny*(ny-1))
                    sum_xy = (onp.sum(Kxy) - onp.sum(onp.diag(Kxy)) )/(nx*(ny-1))
                    mmd_slow = sum_xx - 2*sum_xy + sum_yy
                    slow_time = timeit.default_timer() - start_time

                    # [2] MMD feature compute mode
                    start_time = timeit.default_timer()
                    mmd_fast, model, grad_fn = compute_NTK_MMD(S_P, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                    fast_time = timeit.default_timer() - start_time
                    print("Slow mode time: ", slow_time)
                    print("Fast mode time: ", fast_time)
                    import ipdb; ipdb.set_trace()
                    print()
            elif args.exp_type == "accuracy":
                # h_chosen, mmd_value, mmd_var = 1,1,1
                count = 0
                mmd_vector = onp.zeros(args.n_permutes)
                mmd0, model, grad_fn = compute_NTK_MMD(S_P, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                for i_permute in range(args.n_permutes):
                    pass # for permuation test, always pick kernel MMD computation mode
                    mmd_fast, model, grad_fn = compute_NTK_MMD(S_P, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                    mmd_vector[i_permute] = mmd_fast
                    if mmd_vector[i_permute] > mmd0:
                        count = count + 1 
                    if count > onp.ceil(args.n_permutes * alpha):
                        # reject
                        chosen_hypothesis = 0 # accept null hypothesis P=Q  (failure rejection)
                        break
                    else:
                        chosen_hypothesis = 1 # reject null hypothesis P!=Q (successful rejection)
                chosen_hypothesis_vector[i_test] = chosen_hypothesis
            else:
                raise NotImplementedError("not valid exp_type")
        Results[i_seed] = chosen_hypothesis_vector.sum() / float(args.n_tests)

# TODO type 1 vs type 2 error
# TODO print, hparam tensorboard
# TODO shape handle


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)
def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_dir", type=str,
        default="/scratch/hdd001/home/sheng/scntk_icml")  

    ########### Experiment setup #############
    parser.add_argument("--exp_type", type=str, default="complexity")  # complexity, accuracy

    # "MNIST", "CIFAR10"
    parser.add_argument("--S_P_data", type=str, default="MNIST")  
    # type2 "GANMNIST", "CIFAR10_1", type1 "MNIST", "CIFAR10"
    parser.add_argument("--S_Q_data", type=str, default="GANMNIST")  

    parser.add_argument("--n_seeds", type=int, default=2)   # 2 previously but use 10 now
    parser.add_argument("--n_tests", type=int, default=20)  # number of tests data randomly sampled from the dataset, 10 prev, 100 now
    parser.add_argument("--n_permutes", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=400) #default 100, 400 MMDD 0.996
    parser.add_argument("--batch_size", type=int, default=100)

    ##############  Method  ###############
    parser.add_argument("--method", type=str, default="fixed_ntk") # fixed_ntk vs optimized_ntk vs mmd_o vs mmd_d
    parser.add_argument("--kernel_type", type=str, default="ntk") # fixed vs optimized 
    # parser.add_argument("--mmd_compute_mode", type=str) # feature vs kernel 

    parser.add_argument("--widths", type=int, nargs="*")
    parser.add_argument("--g_activs", type=str, nargs="*")
    parser.add_argument("--bandwidths", type=none_or_float, nargs="*")
    parser.add_argument("--init_stdvs", type=none_or_float, nargs="*")
    parser.add_argument("--filter_sizes", type=int, nargs="*")
    parser.add_argument("--poolings", type=none_or_str, nargs="*")
    parser.add_argument("--use_components", type=int, nargs="*")

    parser.add_argument("--run_id", type=int, default=0)

    args = parser.parse_args()
    # save_dir = os.path.join(args.top_dir, f"MNIST_jan_20th/n{args.n}/N_per{args.N_per}/K{args.K}/N{args.N}/batch_size{args.batch_size}")
    # save_dir = os.path.join(save_dir, f"widths{str(args.widths)}/g_activs{str(args.g_activs)}/bandwidths{str(args.bandwidths)}")
    # save_dir = os.path.join(save_dir, f"init_stdvs{str(args.init_stdvs)}/filter_sizes{str(args.filter_sizes)}/poolings{str(args.poolings)}/kernel_type_{args.kernel_type}/run_id{args.run_id}")

    # args.save_dir = save_dir
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    print("Arguments: ", args)
    main(args, args.batch_size, args.widths, args.g_activs, args.bandwidths, args.init_stdvs, args.filter_sizes, args.poolings,
         args.use_components, args.kernel_type)




