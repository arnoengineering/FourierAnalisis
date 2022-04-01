from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from time import time
import math
from scipy.optimize import curve_fit


def im_2_points(mat, save=False):
    print('NEAREST NEIGHBOUR')
    print('searching ', mat.size)
    # converts image to list of points then does nerest nabour
    index = np.where(mat >= 1)
    not_search = list(zip(index[0], index[1]))
    searched = np.array([0, 0])
    index_l = 0
    max_d = np.linalg.norm(mat.shape)

    while not_search != 0:
        searched = np.vstack((searched, not_search[index_l, :]))
        not_search = np.delete(not_search, index_l,0)
        cur_point = searched[-1]
        dist = max_d
        n = 0
        for x in not_search:
            d = np.linalg.norm(x-cur_point)
            if d < dist:
                dist = d
                index_l = n
            n += 1

    # removes initial empty row
    searched = np.delete(searched, 0, 0)
    plt.plot(searched[:, 1], searched[:, 0])
    plt.show()
    if save:
        save_f(searched)
    return searched


def save_f(dat):
    with open('GreedyData.npy', 'wb') as fi:
        np.save(fi, dat)


def load_f():
    with open('GreedyData.npy', 'rb') as fi:
        dat = np.load(fi)

    return dat


def dist(x,y):
    # print(f'x={x}; y={y}')
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def image_to_p(mat):
    index = np.where(mat >= 1)
    not_search = list(zip(index[0], index[1]))

    last = not_search[0]
    searched = [not_search[0]]
    del not_search[0]
    while not_search:
        # last_ind = 0
        # print(not_search)
        nextt = not_search[0]
        min_dist = dist(last, nextt)
        # for n, i in enumerate(not_search[1:],1):
        for i in not_search[1:]:
            # print(f'last={last}; n={n}')
            d = dist(last, i)
            if d < min_dist:
                nextt = i
                # last_ind = n
                min_dist = d
        searched.append(nextt)
        not_search.remove(nextt)
        last = nextt
    # plt.plot(searched[:][1], searched[:][0])
    # plt.show()
    return searched

def size_scale(size, points):
    time_es = size ** 2 * 1.59 * (points ** 1.5 * 1.24) + 4.6  # todo fix, example
    return time_es


def image_to_p10(mat, timer):
    index = np.where(mat >= 1)
    not_search = list(zip(index[0], index[1]))
    time_es = size_scale(mat.size, len(not_search))
    timer.est_nn(time_es)  # todo set len as 100, timer count elapsed and ron estimater there, maybe onlt at start
    # todo find percentage don and elapsed time, then can reset

    last = not_search[0]
    searched = [not_search[0]]
    del not_search[0]
    while not_search:
        last_ind = 0
        # print(not_search)
        nextt = not_search[0]
        min_dist = dist(last, nextt)
        for n, i in enumerate(not_search[1:],1):
        # for i in not_search[1:]:
            # print(f'last={last}; n={n}')
            d = dist(last, i)
            if d < min_dist:
                nextt = i
                last_ind = n
                min_dist = d
                if d < 1.5:
                    break
        searched.append(nextt)
        del not_search[last_ind]
        timer.up_nn()
        last = nextt
    # plt.plot(searched[:][1], searched[:][0])
    # plt.show()
    return searched

def image_to_sorted(mat):
    #todo nonzero vs where , sort, transpose
    index = np.array(np.nonzero(mat)).T
    index = index[np.argsort(index[:,1])]
    not_search = list(zip(index[:, 0], index[:, 1]))

    index2 = np.where(mat >= 1)
    not_search2 = list(zip(index2[0], index2[1]))
    last = not_search[0]
    searched = [not_search[0]]
    del not_search[0]
    while not_search:
        last_ind = 0
        # print(not_search)
        nextt = not_search[0]
        min_dist = dist(last, nextt)
        for n, i in enumerate(not_search[1:],1):
        # for i in not_search[1:]:
            # print(f'last={last}; n={n}')
            d = dist(last, i)
            if d < min_dist:
                nextt = i
                last_ind = n
                min_dist = d
        searched.append(nextt)
        del not_search[last_ind]
        last = nextt
    # plt.plot(searched[:][1], searched[:][0])
    # plt.show()
    return searched


def log_f(x, a, b):
    return x**b*a*np.log(x)
def scale_op():
    #todo scaling
    coords_list = np.random.randint()
    scale = 100  # todo points
    cur = []
    sc_ls = np.arange(7, scale, 2)
    print('scale, mean list: ', '   mean index: ')
    for iii in sc_ls:
        mati = np.zeros((iii,iii))
        for xi, y in coords_list:
            mati[xi, y] = 1
        t = time()
        for _ in range(50):

            image_to_p10(mati)
            # im_2_points(mati)
        t2 = time()
        for _ in range(50):

            image_to_p(mati)
        t3 = time() - t2
        print(iii, " ",t3 , " ", t2 - t)

def scale_2():
    #todo scaling

    scale = 100  # todo points
    cur = []
    sc_ls = np.arange(5, scale, 2)
    print('scale, mean list: ', '   mean index: ')
    for iii in sc_ls:
        coords_list = np.random.randint(50,size=(iii, 2))
        mati = np.zeros((50, 50))
        for xi in coords_list:
            mati[xi[0], xi[1]] = 1

        t = time()
        for _ in range(50):

            image_to_p10(mati)
            # im_2_points(mati)
        t2 = time()
        for _ in range(50):

            image_to_p(mati)
        t3 = time() - t2
        print(iii, " ",t3 , " ", t2 - t)
        cur.append(t3)
    plt.plot(sc_ls, cur)
    # fit = curve_fit(log_f,sc_ls, cur)
    # print('fit', fit)
    # plt.plot(sc_ls, log_f(sc_ls, *fit[0]), 'g--')
    plt.show()

def scale_3():
    mati = np.zeros((50, 50))
    scale = 50  # todo points
    cur = []
    sc_ls = np.arange(5, scale, 2)
    print('scale, mean list: ', '   mean index: ')
    for iii in sc_ls:
        coords_list = np.random.randint(50,size=(iii, 2))
        mati = np.zeros((50, 50))
        for xi in coords_list:

            mati[xi[0], xi[1]] = 1
        t = time()
        for _ in range(50):
            image_to_p10(mati)
            # im_2_points(mati)
        t2 = time()
        for _ in range(50):

            sh = image_to_sorted(mati)
        # print(sh)
        t3 = time() - t2
        print(iii, " ",t3 , " ", t2 - t)
        cur.append(t3)
    plt.plot(sc_ls, cur)
    # fit = curve_fit(log_f,sc_ls, cur)
    # print('fit', fit)
    # plt.plot(sc_ls, log_f(sc_ls, *fit[0]), 'g--')
    plt.show()


if __name__ == '__main__':
    #todo scaling
    scale_3()
    """blur
    change screen size, 
    make freqency smaller,
    n largest,
    path"""