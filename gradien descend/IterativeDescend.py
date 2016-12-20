import os, numpy as np

class IterativeDescend:
    def __init__(self):
        pass

    w = 0.0000001

    def SteepestDescendDemo(self, n1 = 1301.0):
        k = 3.0
        n = n1
        s1 = 0.0
        s2 = 1.0

        #self.w = 0.0000001#1.0 / (n ** 2.0)
        t = self.w
        i = 0
        while np.abs(s1 - s2) > t:
            i += 1
            s1 = s2
            dif = - self.w * 2.0 * (s2 ** 3.0 - n) * (3.0 * s2 ** 2.0)
            s2 += dif
            #print "s1 is %f, s2 is %f, dif is %f" % (s1, s2, dif)
        print "cubic root of %f is %f, iterate is %d" % (n, s2, i)

    def DescendDemo(self, n1 = 1301.0):
        k = 3.0
        n = n1
        s1 = 0.0
        s2 = 1.0
        t = self.w
        i = 0
        while np.abs(s1 - s2) > t:
            i += 1
            s1 = s2
            dif = self.w * (n - s2 ** 3.0)#* 2.0 * (s2 ** 3.0 - n) * (3.0 * s2 ** 2.0)
            s2 += dif
            #print "s1 is %f, s2 is %f, dif is %f" % (s1, s2, dif)
        print "cubic root of %f is %f, iterate is %d" % (n, s2, i)

    def NewtonDescendDemo(self, n1 = 1301.0):
        k = 3.0
        n = n1
        s1 = 0.0
        s2 = 1.0
        t = self.w
        i = 0
        while np.abs(s1 - s2) > t:
            i += 1
            s1 = s2
            dif = - (s2 ** 3.0 - n) / (3.0 * s2 ** 2.0)#* 2.0 * (s2 ** 3.0 - n) * (3.0 * s2 ** 2.0)
            s2 += dif
            #print "s1 is %f, s2 is %f, dif is %f" % (s1, s2, dif)
        print "cubic root of %f is %f, iterate is %d" % (n, s2, i)
