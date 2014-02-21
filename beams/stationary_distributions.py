def exponential(H, H0):

    Hmax = 1
    psi = np.exp(H / H0) - np.exp(Hmax / H0)

    return psi
