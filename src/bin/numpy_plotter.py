# d is the numpy array we want to plot

d = my_tensor

z,x,y = d.nonzero() # find where the nonzero elements are located

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'red')
plt.savefig(os.path.join("D:", os.sep, "local_github", "yohai_nir_repo", "mice_project", "demo.png"))