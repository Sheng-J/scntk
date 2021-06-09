# datasets type: MNIST vs MNIST, MNIST vs GANMNIST, CIFAR10 vs CIFAR10.1, CIFAR10 vs CIFAR10
# method type: NTK vs SRF vs ... vs , also optimize test power
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
# from flax_ntk import compute_NTK, compute_NTK_MMD
# from nt_ntk import compute_NTK
# from nt_ntk_nocos import compute_NTK
from mmd_d import compute_deep_K, compute_gaussian_K
import timeit
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


alpha = 0.05


def main(args, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type):
    onp.random.seed(819)
    torch.manual_seed(819)
    # torch.cuda.manual_seed(819)
    if args.writer_record:
        writer = SummaryWriter(args.save_dir)
    hparam_dict = {}
    res_dict = {}
    if args.S_P_data == "MNIST":
        os.makedirs("/h/sheng/scntk_icml/data/mnist", exist_ok=True)
        dataloader_FULL = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/h/sheng/scntk_icml/data/mnist",
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
            P_data = imgs.numpy()
        # P_data [m, 1, 32, 32]
        P_data = onp.transpose(P_data, [0,2,3,1])
        # import ipdb; ipdb.set_trace() # transpose if not [m, 32, 32, 1] and numpy TODO 1
        print()
    elif args.S_P_data == "CIFAR10":
        dataset_test = datasets.CIFAR10(root='/h/sheng/scntk_icml/data', download=False,train=False,
                            transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                                    shuffle=True, num_workers=1)
        # Obtain CIFAR10 images
        for imgs, _ in dataloader_test:
            P_data = imgs.numpy()
        P_data = onp.transpose(P_data, [0, 2, 3, 1])
        P_data_ind_all = onp.arange(len(P_data))

    if args.S_Q_data == "GANMNIST":
        GANMNIST = pickle.load(open('/h/sheng/scntk_icml/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
        Q_data = onp.transpose(GANMNIST[0], [0,2,3,1])  # TODO 3, [m, 32, 32, 3], already numpy
        # import ipdb; ipdb.set_trace()
        print()
    # TODO MNIST again
    elif args.S_Q_data == "MNIST":
        assert args.S_P_data == "MNIST"
        Q_data = P_data
    elif args.S_Q_data == "CIFAR10_1":
        S_Q = onp.load('/h/sheng/scntk_icml/cifar10.1_v4_data.npy')
        # import ipdb; ipdb.set_trace() # check shape to be [m, 32, 32, 3] instead of [m, 3, 32, 32] TODO 4
        S_Q = onp.transpose(S_Q, [0,3,1,2])
        ind_Q = onp.random.choice(len(S_Q), len(S_Q), replace=False)
        S_Q = S_Q[ind_Q]  # just randomize it?
        TT = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans = transforms.ToPILImage()
        data_trans = torch.zeros([len(S_Q),3,32,32])
        data_T_tensor = torch.from_numpy(S_Q)
        for i in range(len(S_Q)):
            d0 = trans(data_T_tensor[i])
            data_trans[i] = TT(d0)
        Q_data = onp.transpose(data_trans.numpy(), [0, 2, 3, 1])
        # Ind_v4_all = onp.arange(len(S_Q))
    elif args.S_Q_data == "CIFAR10":
        pass

    model = None
    grad_fn = None
    Results = onp.zeros([args.n_seeds])
    for i_seed in range(args.n_seeds):
        # train data chosen (test data pool also chosen)
        torch.manual_seed(i_seed*19 + args.n_samples)
        onp.random.seed(seed=1102*(i_seed+10)+args.n_samples)

        if args.S_P_data == "MNIST":
            P_ind_all = onp.arange(4000)
            P_ind_tr = onp.random.choice(4000, args.n_samples, replace=False)
            P_ind_te_left = onp.delete(P_ind_all, P_ind_tr)
            
            Q_ind_all = onp.arange(4000)
            Q_ind_tr = onp.random.choice(4000, args.n_samples, replace=False)
            Q_ind_te_left = onp.delete(Q_ind_all, Q_ind_tr)
        elif args.S_P_data == "CIFAR10":
            # P_ind_all = onp.arange(len(P_data))
            # P_ind_tr = onp.random.choice(len(P_data), args.n_samples, replace=False)
            # P_ind_te_left = onp.delete(P_ind_all, P_ind_tr)
            # 
            # Q_ind_all = onp.arange(len(Q_data))
            # Q_ind_tr = onp.random.choice(len(Q_data), args.n_samples, replace=False)
            # Q_ind_te_left = onp.delete(Q_ind_all, Q_ind_tr)

            P_ind_te_left = onp.arange(len(P_data))
            Q_ind_te_left = onp.arange(len(Q_data))


        # # TODO train phase for optimized test power version
        # if args.method == "optimized_ntk":
        #     pass

        chosen_hypothesis_vector = onp.zeros(args.n_tests)
        for i_test in tqdm(range(args.n_tests)):
            onp.random.seed(seed=1102*(i_test+1) + args.n_samples)
            # test data chosen from test data pool
            P_ind_te = onp.random.choice(len(P_ind_te_left), args.n_samples, replace=False)
            S_P = P_data[P_ind_te_left[P_ind_te]]  # [m, 32, 32, 3]
            # S_P = onp.reshape(S_P.numpy(), [args.n_samples, -1])

            Q_ind_te = onp.random.choice(len(Q_ind_te_left), args.n_samples, replace=False)
            S_Q = Q_data[Q_ind_te_left[Q_ind_te]]
            # S_Q = onp.reshape(S_Q, [args.n_samples, -1])

            nx, ny = len(S_Q), len(S_Q)
            nxy = nx + ny
            assert nx == ny

            if args.exp_type == "complexity":
                pass
            elif args.exp_type == "accuracy":
                # h_chosen, mmd_value, mmd_var = 1,1,1
                count = 0
                mmd_vector = onp.zeros(args.n_permutes)
                # Kx, model, grad_fn = compute_NTK(S_P, S_P, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                # Ky, _, __ = compute_NTK(S_Q, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                # Kxy, _, __ = compute_NTK(S_P, S_Q, batch_size, widths, g_activs, bandwidths, init_stdvs, filter_sizes, poolings, use_components, kernel_type, model, grad_fn)
                S_P_1 = onp.repeat(onp.expand_dims(S_P, 1), nx, axis=1) 
                S_P_2 = onp.repeat(onp.expand_dims(S_P, 0), nx, axis=0)
                S_Q_1 = onp.repeat(onp.expand_dims(S_Q, 1), nx, axis=1)
                S_Q_2 = onp.repeat(onp.expand_dims(S_Q, 0), nx, axis=0)
                Kx = onp.exp(-args.bandwidths[0]*onp.linalg.norm(S_P_1-S_P_2, ord=1, axis=2) )
                Ky = onp.exp(-args.bandwidths[0]*onp.linalg.norm(S_Q_1-S_Q_2, ord=1, axis=2) )
                Kxy = onp.exp(-args.bandwidths[0]*onp.linalg.norm(S_P_1-S_Q_2, ord=1, axis=2) )

                sum_xx = (onp.sum(Kx) - onp.sum(onp.diag(Kx)) )/(nx*(nx-1))
                sum_yy = (onp.sum(Ky) - onp.sum(onp.diag(Ky)) )/(ny*(ny-1))
                sum_xy = (onp.sum(Kxy) - onp.sum(onp.diag(Kxy)) )/(nx*(ny-1))
                mmd0 = sum_xx - 2*sum_xy + sum_yy
                # import ipdb; ipdb.set_trace()
                Kxyxy_left = onp.concatenate([Kx, Kxy.T], axis=0)
                Kxyxy_right = onp.concatenate([Kxy, Ky], axis=0)
                Kxyxy = onp.concatenate([Kxyxy_left, Kxyxy_right], axis=1)
                for i_permute in range(args.n_permutes):
                    permuted_ind = onp.random.choice(nxy, nxy, replace=False)
                    indx = permuted_ind[:nx]
                    indy = permuted_ind[nx:]
                    Kx = Kxyxy[onp.ix_(indx, indx)]
                    Ky = Kxyxy[onp.ix_(indy, indy)]
                    Kxy = Kxyxy[onp.ix_(indx, indy)]
                    # [1] MMD kernel compute mode
                    sum_xx = (onp.sum(Kx) - onp.sum(onp.diag(Kx)) )/(nx*(nx-1))
                    sum_yy = (onp.sum(Ky) - onp.sum(onp.diag(Ky)) )/(ny*(ny-1))
                    sum_xy = (onp.sum(Kxy) - onp.sum(onp.diag(Kxy)) )/(nx*(ny-1))
                    mmd_slow = sum_xx - 2*sum_xy + sum_yy
                    mmd_vector[i_permute] = mmd_slow

                    # mmd_vector[i_permute] = mmd_fast

                    if mmd_vector[i_permute] > mmd0:
                        count = count + 1 
                    if count > onp.ceil(args.n_permutes * alpha):
                        # reject
                        chosen_hypothesis = 0 # accept null hypothesis P=Q  (failure rejection)
                        break
                    else:
                        chosen_hypothesis = 1 # reject null hypothesis P!=Q (successful rejection)
                chosen_hypothesis_vector[i_test] = chosen_hypothesis
                print(f"test {i_test} chosen hypothesis {chosen_hypothesis}")
            else:
                raise NotImplementedError("not valid exp_type")
        avg_hypothesis_val = chosen_hypothesis_vector.sum() / float(args.n_tests)
        Results[i_seed] = avg_hypothesis_val
        print(Results) # TODO check print result, for each seed, i_seed
        if args.S_P_data == args.S_Q_data:
            print("ground truth: P==Q")
            print("successful accept accuracy (accept h0 rate)=", (1-avg_hypothesis_val))
        else:
            print("ground truth: P!=Q")
            print("successful reject accuracy (accept h1 rate)=", avg_hypothesis_val)
    if args.writer_record:
        writer.add_hparams(hparam_dict=hparam_dict, metric_dict=res_dict)


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
    parser.add_argument("--exp_type", type=str, default="accuracy")  # complexity, accuracy
    parser.add_argument("--writer_record", type=int, default=0)  # complexity, accuracy

    # "MNIST", "CIFAR10"
    parser.add_argument("--S_P_data", type=str, default="MNIST")  
    # type2 "GANMNIST", "CIFAR10_1", type1 "MNIST", "CIFAR10"
    parser.add_argument("--S_Q_data", type=str, default="GANMNIST")  

    parser.add_argument("--n_seeds", type=int, default=5)   # 2 previously but use 10 now
    parser.add_argument("--n_tests", type=int, default=50)  # number of tests data randomly sampled from the dataset, 10 prev, 100 now
    parser.add_argument("--n_permutes", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=400) #default 100, 400 MMDD 0.996
    parser.add_argument("--batch_size", type=int, default=100)

    ##############  Method  ###############
    parser.add_argument("--method", type=str, default="fixed_ntk") # fixed_ntk vs optimized_ntk vs mmd_o vs mmd_d
    parser.add_argument("--kernel_type", type=str, default="exact_ntk") # fixed vs optimized ntk
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
    save_dir = os.path.join(args.top_dir, f"MNIST_jan_30th/S_P_{args.S_P_data}/S_Q_{args.S_Q_data}/n{args.n_seeds}/n_tests{args.n_tests}/n_permutes{args.n_permutes}/n_samples{args.n_samples}")
    save_dir = os.path.join(save_dir, f"widths{str(args.widths)}/g_activs{str(args.g_activs)}/bandwidths{str(args.bandwidths)}")
    save_dir = os.path.join(save_dir, f"init_stdvs{str(args.init_stdvs)}/filter_sizes{str(args.filter_sizes)}/poolings{str(args.poolings)}/kernel_type_{args.kernel_type}/run_id{args.run_id}")

    args.save_dir = save_dir
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    print("Arguments: ", args)
    main(args, args.batch_size, args.widths, args.g_activs, args.bandwidths, args.init_stdvs, args.filter_sizes, args.poolings,
         args.use_components, args.kernel_type)




