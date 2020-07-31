# encoding=utf-8


def kernel_mu(n_kernels):
    """
    get mu for each guassian kernel, Mu is the middele of each bin
    :param n_kernels: number of kernels( including exact match). first one is the exact match
    :return: mus, a list of mu
    """
    mus = [1]  # exact match
    if n_kernels == 1:
        return mus
    bin_step = (1 - (-1)) / (n_kernels - 1)  # score from [-1, 1]
    mus.append(1 - bin_step / 2)  # the margain mu value
    for k in range(1, n_kernels - 1):
        mus.append(mus[k] - bin_step)
    return mus


def kernel_sigma(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels(including the exact match)
    :return: sigmas, a list of sigma
    """
    sigmas = [0.001]  # exact match small variance means exact match ?
    if n_kernels == 1:
        return sigmas
    return sigmas + [0.1] * (n_kernels - 1)


if __name__ == '__main__':
    print(kernel_mu(11))
    print(kernel_sigma(11))
