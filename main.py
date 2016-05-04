import KMeans, LineaRegression, IterativeDescend
import numpy as np

#reger = LineaRegression.LinearRegression()
#reger.ExponentLinearRegressionDemo(6, (2, 4, 1, 7, 3), 100, (-1, 1), 5)
#reger.ExponentLinearRegressionDemo(5, (1, 4, 4), 500, (-10, 10), 5)
#reger.RegularExponentLinearRegressionDemo(3, (1, 4, 4), 500, (-20, 20), 5, 1000)

#kmeaner = KMeans.KMeans()
#kmeaner.KMeansDemo()
print np.power(9999.0, 1.0/3.0)
descender = IterativeDescend.IterativeDescend()
descender.DescendDemo(9999.0)
descender.SteepestDescendDemo(9999.0)
descender.NewtonDescendDemo(9999.0)