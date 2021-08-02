# encoding=utf8
import numpy as np
import unittest
from datap import molecule,mol
from itertools import combinations_with_replacement,combinations,product
from pyquante2.grid.atomic_grid import lebedev
from pyquante2.grid.atomic_grid import atomic_grid
from pyquante2.grid.grid import grid
from pyquante2.basis.basisset import basisset
from pyquante2.dft.functionals import xs,cvwn5,xb88,xpbe,clyp,cpbe
from pyquante2.dft.reference import data
from pyquante2.ints.hgp import ERI_hgp as ERI
from pyquante2.ints.integrals import twoe_integrals,onee_integrals,iiterator
from pyquante2.ints.one import S,T,V
from pyquante2.utils import trace2, geigh,pairs,binomial,norm2,fact2,Fgamma,dmat
from pyquante2.scf.iterators import SCFIterator,USCFIterator,AveragingIterator,ROHFIterator
from pyquante2.scf.hamiltonians import hamiltonian,rhf,dft
from numpy import pi,exp,floor,array,isclose
from math import factorial,lgamma
import multiprocessing
import time

times = []
names= []
energias= []

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        times.append((time2-time1)*1000.0)
        return ret
    return wrap

def tsvwn(molecula):
    bfs = basisset(molecula,'sto3g')
    solver = dft(molecula ,bfs,'svwn')
    ens = solver.converge()
    print "energy svmw for  \n"
    print molecula.name
    print "Hartees"
    print solver.energy
    tup=(molecula.name,solver.energy)
#    energias.append(solver.energy)
#    names.append(mol[n].name)
    return tup

def tiempores(molecula):
    time1 = time.time()
    xx=tsvwn(molecula)
    time2 = time.time()
    timex= round( (time2-time1)*1000.0, 3)
    a=list(xx)
    a.append(timex)
    b=tuple(a)
    #xx.append(timex)
    return b

#@timing
def mediador():
    con=0
    f = open('energies_1es7list.txt','w+')
    r=len(mol)
    d=0
    print "cuantas moleculas"
    print r
    l=21
    scalar=0
    dif=(l-d)
    print "dif"
    print dif
    print "inicio"
    print d
    print "final"
    print l
    print "cuantos cores"
    print multiprocessing.cpu_count()
    max=dif // multiprocessing.cpu_count()
    print "el max"
    print max
    tem=0
    for i in range(d,l,multiprocessing.cpu_count()):
        con=con+1
        print "el contador va en"
        print con
        if (con <= max):
            aa=mol[i]
            bb=mol[i+1]
            cc=mol[i+2]
            dd=mol[i+3]
            tup=(aa,bb,cc,dd)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            energies = pool.map(tiempores,tup)
            pool.close()
            pool.join()
            for e in energies:
                scalar=scalar+e[2]
                f.write('[')
                for value in e:
                    f.write(str(value) + ', ')
                f.write('] \n')


    return scalar

if __name__ == '__main__':
#    jobs = []
#    n=1
    time1 = time.time()
    scalar=mediador()
    time2 = time.time()
    tiempo_total= round( (time2-time1)*1000.0, 3)
    print "tiempo escalar"
    print scalar
    print "tiempo paralelo"
    print tiempo_total
