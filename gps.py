from casadi import *
from scipy.linalg import solve_triangular


# RBF Kernel for the Gaussian Process
def RBF(X, Y, model):
    sX = X.shape[0]
    sY = Y.shape[0]
    length_scale = model.kernel_.get_params()['k1__k2__length_scale'].reshape(1, -1)
    constant = model.kernel_.get_params()['k1__k1__constant_value']
    X = X / repmat(length_scale, sX, 1)
    Y = Y / repmat(length_scale, sY, 1)
    dist = repmat(sum1(X.T ** 2).T, 1, sY) + repmat(sum1(Y.T ** 2), sX, 1) - 2 * mtimes(X,
                                                                                        Y.T)  # sum1 is sum by row and mtimes is matrix multiplication in casadi
    K = constant * exp(-.5 * dist)
    return K


def Constant(X, Y, model):
    constant = model.kernel_.get_params()['k2__constant_value']
    sX = X.shape[0]
    sY = Y.shape[0]
    K = constant * SX.ones((sX, sY))
    return K


# define the Gaussian Process model used for the model correction
def GPModel(name, model, yscaler):
    X = model.X_train_
    x = SX.sym('x', 1, X.shape[1])  # create casadi matrix
    # parameters for the RBF-kernel GP
    K1 = RBF(x, X, model)
    K2 = Constant(x, X, model)
    K = K1 + K2
    # meean
    y_mu = mtimes(K, model.alpha_) + model._y_train_mean
    y_mu = y_mu * yscaler.scale_ + yscaler.mean_
    # variance
    L_inv = solve_triangular(model.L_.T, np.eye(model.L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    K1_ = RBF(x, x, model)
    K2_ = Constant(x, x, model)
    K_ = K1_ + K2_

    y_var = diag(K_) - sum2(mtimes(K, K_inv) * K)  # sum2 is sum by columns in casadi
    y_var = fmax(y_var, 0)  # fmax is element-wise maximum between 2 values
    y_std = sqrt(y_var)
    y_std *= yscaler.scale_

    gpmodel = Function(name, [x], [y_mu, y_std])
    return gpmodel
