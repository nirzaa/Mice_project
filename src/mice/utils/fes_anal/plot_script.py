import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import cm
#from mpl_toolkits.mplot3d import axes3d
from scipy import interpolate

fileBase='fes_'
kT = 320.0

for fileNumber in range(0,2,1):
	x, y, z, dummy1, dummy2 = np.loadtxt(fileBase + str(fileNumber) + ".dat", unpack=True)
	print(f'working on: {fileBase}{str(fileNumber)}.dat')
	xn = np.array(x)
	yn = np.array(y)
	zn = np.array(z)
	shenet_num = int(np.sqrt(xn.shape))
	my_x = xn[0:shenet_num] / kT
	my_y = yn[::shenet_num]

	[X,Y] = np.meshgrid(my_x, my_y)
	Z = -zn.reshape((shenet_num, shenet_num)) / 10
	print(Z.shape)
	levels = np.linspace(-20,0,15)
	plt.contourf(X,Y,Z,levels)
	plt.colorbar()
	plt.show()

	# x_mat = xn.reshape((shenet_num, shenet_num))
	# np.savetxt('x_mat.txt', x_mat)



