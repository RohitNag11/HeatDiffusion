import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Define COnstants
Tf = 0  # Fluid Temp
h = 500  # Convection Coefficient
K = 385.0  # Thermal Conductivity
a1 = 1.88*10**(-5)  # Diffusivity of material 1
a2 = 5.2*10**(-7)  # Diffusivity of material 2

# Set the domain of solution and the discretised grid
t = 300000  # Total Time

# Select number of nodes:
# For 4 spatial nodes:
# nx = 3  # Number of x, y increments
# nt = 3000  # Number of time steps

# For 20 spatial nodes:
# nx=19          #Number of x, y increments
# nt=30000            #Number of time steps

# #For 40 spatial nodes(for grid ten times finer):
nx = 39  # Number of x, y increments
nt = 60000  # Number of time steps

dt = t/nt  # Time step

L = 1  # Length of x domain (l) = length of y domain (w)
dx = L/nx  # Space step
N = int(nx+1)  # Number of nodes in x or y domain
r = int(nt+1)  # Number of time nodes

G = K/(h*dx)  # Useful constant used for internal nodes calculation

tlist = np.arange(0, t+dt, 100)  # temp domain
xlist = np.arange(0, L+dx, dx)  # x domain
ylist = np.arange(0, L+dx, dx)  # y domain

# Define matrix for constant d=d(x,y) as defined in summary document:
D = np.zeros((N, N))
D[:, :] = a1*dt*(1/(dx**2))  # d for material 1
D[int(0.6*N):int(0.8*N), int(0.2*N):int(0.8*N)] = a2 * \
    dt*(1/(dx**2))  # d and location of material 2
# Establish stability conditions:
if np.max(D[:, :]) <= 0.25:  # stable for d<=0.25
    print('Solution is stable')
else:
    print('Solution is unstable!')

# Solve for temperature:
T = np.zeros((N, N, r))  # Empty temperature matrix, T

# Set the boundary conditions/initial values
# Initial Condition:
T0 = 1173  # Initial Temp of whole shape
T[:, :, :] = T0

# Boundary Conditions:
for k in range(0, r-1):

    for i in range(0, N):
        # Left Face
        T[i, 0, k+1] = (1/(1+G))*(G*T[i, 1, k]+Tf)  # Left Face
        T[i, N-1, k+1] = (1/(1+G))*(-Tf+G*T[i, N-2, k])  # Right Face

        for j in range(0, N):
            T[0, j, k+1] = (1/(1+G))*(-Tf+(G*T[1, j, k]))  # Top Face
            T[N-1, j, k+1] = (1/(1+G))*(-Tf+(G*T[N-2, j, k]))  # Bottom Face


# Implement the numerical method for internal nodes:
            if 0 < i < N-1 and 0 < j < N-1:  # Exclude boundary nodes for calculation
                # Internal Nodes using discretisation
                T[i, j, k+1] = T[i, j, k]+D[i, j] * \
                    (T[i+1, j, k]+T[i-1, j, k] +
                     T[i, j+1, k]+T[i, j-1, k]-4*T[i, j, k])

# Plot results:
# Create Meshgrid
X, Y = np.meshgrid(xlist, ylist)

# Automated Plot:
ti = 0
if np.max(D[:, :]) <= 0.25:
    print("Please wait")  # Stability Condition
    for i in range(0, nt):
        if ti < nt:
            plt.style.use('dark_background')
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # fig.set_facecolor('black')
            # ax.set_facecolor('black')
            ax.plot_surface(X, Y, T[:, :, ti], vmin=0., vmax=np.max(
                [T0, Tf])+10, cmap=cm.gnuplot2, linewidth=0, antialiased=False)
            ax.set_title('Temperature Distribution for t=' +
                         str(ti*dt)+'s')  # Label Title
            ax.set_xlabel('x [m]')  # label x axis
            ax.set_ylabel('y [m]')  # Label y axis
            ax.set_zlabel('T (°C)')  # Label z axis
            # Set z axis range (0, max(T0,Tf))
            ax.set_zlim(0, np.max([T0, Tf])+100)
            ax.view_init(30, (5*i)*(2000/nt))  # Rotate view incrementally
            plt.savefig(f'plots/{i}.png')  # Save each plot as a new image
            # plt.show()  # Show Plot
            # plt.pause(0.0001)  # Pause between each frame
            # Time instances between each frame (not real time in seconds)
            plt.close()
            ti += 10


print('Finished')
