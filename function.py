import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise
import numpy as np
from scipy.optimize import minimize
import time


np.random.seed(1773784)
tol = 10 ** -9
opt_tol = 10**-3
max_iter = 2000

def load_data(filename):
    # Load CSV using Pandas
    data = pd.read_csv(filename)
    Y = data.pop('y').reshape(-1, 1)
    Y[Y == 0] = -1
    X = scale(data.values)
    return X,Y


def split_data(X, Y):
    return train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=1)

def gram_matrix(X, Y, gamma, kernel='gaussian'):
    K = X*Y
    if (kernel == 'gaussian'):
        return pairwise.rbf_kernel(K, gamma=gamma)
    elif (kernel == 'polynomial'):
        return pairwise.polynomial_kernel(K,gamma=gamma)

def gaussian_kernel(X, x, gamma):
    return np.exp(-gamma*np.square(np.linalg.norm(X-x,axis=1)))


def objective_func(alpha, Q):
    alpha = alpha.reshape(-1, 1)
    e = np.ones_like(alpha)
    aQa = np.dot(np.dot(np.transpose(alpha), Q),alpha)
    objective_value = (1 / 2) * aQa - alpha.sum()

    grad = np.dot(Q, alpha) - e
    return objective_value, grad

def func_deriv(alpha,Q):
    e = np.ones((Q.shape[0],1))
    return np.dot(Q,alpha) - e

def find_alpha_star(Xtr, Ytr, C, gamma):
    Q = gram_matrix(Xtr, Ytr, gamma)
    alpha_0 = np.zeros((Q.shape[0],1))
    boxbnds = np.tile([0,C],Xtr.shape[0]).reshape(-1,2)

    cons = ({'type': 'eq', 'fun': lambda alpha: np.dot(np.transpose(Ytr), alpha)})

    res = minimize(objective_func, x0=alpha_0, args=(Q),
                   method='SLSQP', jac=True, bounds=boxbnds, constraints=cons, options={'disp': False})

    return res.x.reshape(-1,1), res


def find_b_star(alpha_star, X, Y, C, gamma):
    b = np.empty(0)
    for i, ai in enumerate(alpha_star):
        if tol < ai < C-tol:
            w_x = np.dot(alpha_star * Y.ravel(), gaussian_kernel(X, X[i], gamma))
            b = np.append(b, [1 / Y[i] - w_x])
    return np.average(b)

#############################CHECK##########################################
def predict(alpha_star, b, Xtr, Ytr, Xtst, gamma):
    pred = np.empty(0)
    for x in Xtst:
        w_x = np.dot((alpha_star * Ytr).ravel(), gaussian_kernel(Xtr, x, gamma))
        pred = np.append(pred, np.sign(w_x + b))
    return pred.reshape(-1,1)

def acc_score(ypred, ytst):
    from sklearn.metrics import accuracy_score
    return accuracy_score(ypred, ytst)


def grid_search(C_range, gamma_range,X,Y):
    best_accuracy = 0.0
    best_sol = np.zeros(4)
    Xtr, Xtst, Ytr, Ytst = split_data(X, Y)
    for C in C_range:
        for gamma in gamma_range:
            print("Best accuracy: ", best_accuracy)
            print("\nC: ", C, "\tgamma: ", gamma)
            t = time.time()
            alpha_star, _ = find_alpha_star(Xtr, Ytr, C, gamma)
            totalTime = time.time() - t

            #print("alpha_star.shape: ",alpha_star.shape,"\nalpha_star\n", alpha_star)
            b_star = find_b_star(alpha_star, Xtr, Ytr, C, gamma)

            ypred = predict(alpha_star, b_star, Xtr, Ytr, Xtst, gamma)
            # print("Accuracy score on test: %2f" % (f.acc_score(ypred, Ytst)))

            ytr = predict(alpha_star, b_star, Xtr, Ytr, Xtr, gamma)
            # print("Accuracy score on training: %2f" % (f.acc_score(ytr, Ytr)))
            accuracy = acc_score(ypred, Ytst)
            print("Accuracy on training: ", acc_score(ytr, Ytr))
            print("Accuracy on test set: ", accuracy)
            print("TOTAL TIME: ", totalTime)
            if accuracy >= best_accuracy:
                best_sol[0] = C
                best_sol[1] = gamma
                best_sol[2] = accuracy
                best_sol[3] = totalTime
                best_accuracy = accuracy

    return best_sol



def scikit_check(X, y, Xtst, ytst, gamma):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    clf = SVC(gamma=gamma)
    clf.fit(X, y)
    ytrpred = clf.predict(X)
    ytstpred = clf.predict(Xtst)
    return clf.score(Xtst,ytst), acc_score(ytrpred,y), acc_score(ytstpred,ytst)


def get_RS(alpha, gradient, y_train, C):
    R,S = [],[]
    for i in range(len(y_train)):
        if np.abs(alpha[i])<tol:
            if y_train[i]== 1:
                R.append((np.asscalar(-1*y_train[i]*gradient[i]),i))
            if y_train[i]==-1:
                S.append((np.asscalar(-1*y_train[i]*gradient[i]),i))
        if np.abs(alpha[i]-C)<tol:
             if y_train[i]==-1:
                R.append((np.asscalar(-1*y_train[i]*gradient[i]),i))
             if y_train[i]==1:
                S.append((np.asscalar(-1*y_train[i]*gradient[i]),i))
        if alpha[i]>=tol and alpha[i]<=C-tol:
            R.append((np.asscalar(-1*y_train[i]*gradient[i]),i))
            S.append((np.asscalar(-1*y_train[i]*gradient[i]),i))
    return R, S

def select_W_k(alpha, gradient, y_train, C, q):
    R,S = get_RS(alpha, gradient, y_train,C)
    R.sort()
    R.reverse()
    S.sort()
    W_k = np.array([it[1] for it in R[:int(q/2)]] + [it[1] for it in S[:int(q/2)]])
    not_W_k = np.array([it for it in [idx for idx in range(len(y_train))] if it not in W_k])
    M = min(S)[0]#[it[0] for it in S[:int(q/2)]]
    m = max(R)[0]#[it[0] for it in R[:int(q/2)]]
    opt = np.isclose(m,M,opt_tol)
    return W_k, not_W_k, opt, m, M

def objective_func_w(alpha_w, alpha_not_w, W_k, W_not_k, Q):
    Q_w = Q[:, W_k]
    Q_ww = Q_w[W_k, :]
    Q_notww = Q_w[W_not_k, :]
    e_w_T = np.ones((1, Q_ww.shape[0]))
    awT_Qww = np.dot(np.transpose(alpha_w), Q_ww)
    anotwT_Qnotw = np.dot(np.transpose(alpha_not_w), Q_notww)
    objective_value = (1 / 2) * np.dot(awT_Qww, alpha_w) + np.dot((anotwT_Qnotw - e_w_T), alpha_w)

    grad = awT_Qww + anotwT_Qnotw - e_w_T
    return objective_value, grad



def SVMlight(X, Y, C, gamma, q):
    Q = gram_matrix(X, Y, gamma)
    alpha = np.zeros((X.shape[0], 1))
    total_fun_eval = 0
    total_jac_eval = 0

    for i in range(max_iter):
        #        print "outer iteration: ", k+1
        gradient = func_deriv(alpha, Q)
        W_k, not_W_k, optimality, _, _ = select_W_k(alpha, gradient, Y, C, q)

        if optimality:
            break

        alpha_w = alpha[W_k]
        y_w = Y[W_k]
        y_not_w = Y[not_W_k]
        alpha_not_w = alpha[not_W_k]
        boxbnds = np.tile([0, C], q).reshape(-1, 2)
        constant = np.dot(np.transpose(y_not_w), alpha_not_w).ravel()
        cons = {'type': 'eq', 'fun': lambda alpha_w: np.dot(np.transpose(y_w), alpha_w).ravel() + constant}
        res = minimize(objective_func_w, x0=alpha_w, args=(alpha_not_w, W_k, not_W_k, Q),
                       method='SLSQP', jac=True, bounds=boxbnds, constraints=cons, options={'disp': False}, tol=tol)

        alpha[W_k] = res.x.reshape(-1,1)

        total_fun_eval += res.nfev
        total_jac_eval += res.njev
    opt_obj,_ = objective_func(alpha,Q)

    return alpha, i, total_fun_eval, total_jac_eval, opt_obj



def find_tmax(di, dj, C, alpha_i, alpha_j):
    if np.all(np.greater([di,dj],tol)):
        tmax = min(C - alpha_i, C - alpha_j)
    elif np.all(np.less([di,dj],tol)):
        tmax = min(alpha_i, alpha_j)
    elif np.greater(di,tol) and np.less(dj,tol):
        tmax = min(C - alpha_i, alpha_j)
    else:
        tmax = min(alpha_i, C - alpha_j)
    return tmax


def PSA(W_k, Y, alpha_w, grad, Q_ww, C):
    d = np.take(Y, W_k).reshape(-1,1)
    d[1] = -d[1]
    direct = np.dot(np.take(grad,[W_k]), d)
    tmax = find_tmax(d[0], d[1], C, alpha_w[0], alpha_w[1])

    if np.isclose(direct,0.0,rtol=tol):
        return alpha_w
    if(np.greater(direct,tol)):
        d = -d

    dQwd = np.dot(np.dot(np.transpose(d), Q_ww), d)

    if np.isclose(dQwd,0.0,rtol=tol):
        t_star = tmax
    else:
        t_nv = -direct / dQwd
        t_star = min(tmax, t_nv)

    return alpha_w + t_star * d

def MVP(X, Y, C, gamma, max_it=max_iter):
    Q = gram_matrix(X, Y, gamma)
    alpha = np.zeros((X.shape[0], 1))
    for i in range(max_it):
        gradient = func_deriv(alpha,Q)
        W_k,_, optimality, _, _ = select_W_k(alpha, gradient, Y, C, 2)

        if optimality:
            break

        Q_w = Q[:, W_k]
        Q_ww = Q_w[W_k, :]
        alpha_w = np.take(alpha, W_k).reshape(-1,1)
        alpha_w_star = PSA(W_k,Y,alpha_w,gradient,Q_ww,C)
        alpha[W_k] = alpha_w_star
    objective_value, _ = objective_func(alpha, Q)
    return alpha, objective_value, i
