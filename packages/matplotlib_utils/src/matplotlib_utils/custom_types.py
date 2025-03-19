import matplotlib.pyplot as plt
import mpl_toolkits


Axes2D = plt.Axes
Axes3D = mpl_toolkits.mplot3d.Axes3D
Axes = Axes2D | Axes3D
