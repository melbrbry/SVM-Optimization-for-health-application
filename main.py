import function as f
import time

filename = '2017.12.11 Dataset Project 2.csv'
C = 6.5
gamma = 0.24

X,Y = f.load_data(filename)
Xtr, Xtst, Ytr, Ytst = f.split_data(X, Y)


def main():

    t = time.time()
    alpha_star,res = f.find_alpha_star(Xtr, Ytr, C, gamma)
    totalTime = time.time() - t

    b_star = f.find_b_star(alpha_star,Xtr,Ytr,C,gamma)
    #print("b_star: ", b_star)

    ytstpred = f.predict(alpha_star, b_star, Xtr, Ytr, Xtst, gamma)
    ytrpred = f.predict(alpha_star, b_star, Xtr, Ytr, Xtr, gamma)

    test_accuracy = f.acc_score(ytstpred,Ytst)

    #print("Main train accuracy: ",f.acc_score(ytrpred,Ytr))

    output = open("output_homework2_28.txt", "a")  # instead of 99, number of the team
    output.write("Homework 2, question 1")
    output.write("\nTraining objective function," + "%f" %res.fun)
    output.write("\nTest accuracy," + "%f" % test_accuracy)
    output.write("\nTraining computing time," + "%f" % totalTime)
    output.write("\nFunction evaluations," + "%i" % res.nfev)
    output.write("\nGradient evaluations," + "%i" % res.njev)
    output.close()

def mainMVP():
    t = time.time()
    alpha_star, objective_value, i = f.MVP(Xtr, Ytr, C, gamma)
    total_time = time.time() - t

    b_star = f.find_b_star(alpha_star, Xtr, Ytr, C, gamma)


    ytrpred = f.predict(alpha_star, b_star, Xtr, Ytr, Xtr, gamma)
    ytstpred = f.predict(alpha_star, b_star, Xtr, Ytr, Xtst, gamma)

    test_accuracy = f.acc_score(ytstpred, Ytst)
    #print("Main train accuracy: ", f.acc_score(ytrpred, Ytr))
    output = open("output_homework2_28.txt", "a")  # instead of 99, number of the team
    output.write("\nHomework 2, question 3")
    output.write("\nTraining objective function," + "%f" % objective_value)
    output.write("\nTest accuracy," + "%f" % test_accuracy)
    output.write("\nTraining computing time," + "%f" % total_time)
    output.write("\nOuter iterations," + "%i" % i)
    output.close()


def mainSVML():
    q = 12
    t1 = time.time()
    alpha_star, k, total_fun_eval, total_jac_eval, opt_obj = f.SVMlight(Xtr, Ytr, C, gamma, q)

    total_time = time.time() - t1
    b_star = f.find_b_star(alpha_star, Xtr, Ytr, C, gamma)



    y_pred = f.predict(alpha_star, b_star, Xtr, Ytr, Xtst, gamma)
    test_acc = f.acc_score(y_pred, Ytst)

    ytrpred = f.predict(alpha_star, b_star, Xtr, Ytr, Xtr, gamma)
    #print("Main train accuracy: ", f.acc_score(ytrpred, Ytr))

    output = open("output_homework2_28.txt", "a")  # instead of 99, number of the team
    output.write("\nHomework 2, question 2")
    output.write("\nTraining objective function," + "%f" % opt_obj)
    output.write("\nTest accuracy," + "%f" % test_acc)
    output.write("\nTraining computing time," + "%f" % total_time)
    output.write("\nOuter iterations," + "%i" % k)
    output.write("\nFunction evaluations," + "%i" % total_fun_eval)
    output.write("\nGradient evaluations," + "%i" % total_jac_eval)
    output.close()



main()
mainSVML()
mainMVP()

