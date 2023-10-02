import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation, rc
matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
from IPython.display import HTML
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\FFMPEG\bin\ffmpeg.exe'

#############################################################
##############  définition des paramètres   #################
#############################################################

T = 1               #période
omega = 2*np.pi/T   #pulsation
R = 0.1             #amplitude du dipole
c = 5                #vitesse lumière
Q = 1               #charge particule
p0 = Q * R          #moment dipolaire

tf = 5
pas = 0.02
t = np.arange(0,tf,pas)

d = 20              #dimension fig
n = 100              #nombre de bruh

#############################################################
##############  calcul CE (indep du temps)  #################
#############################################################

X = np.zeros(len(t))            #position dipole
Z = R * np.sin(omega * pas * t)

x, z = np.meshgrid(np.linspace(-d,d,n),np.linspace(-d,d, n))

r = np.sqrt(z**2 + x**2)            #distance de chaque point du meshgrid
theta = np.arctan(x/z)              #theta de chaque point du meshgrid

Ex = omega**2/(r * c**2) * p0 * np.sin(theta)**2
Ez = - omega**2/(r * c**2) * p0 * np.sin(theta)*np.cos(theta)


Ex = Ex/np.sqrt(Ex**2+Ez**2+1e-100)        #normalisation
Ez = Ez/np.sqrt(Ex**2+Ez**2+1e-100)
#M = np.log((np.hypot(u,v))**4)

#############################################################
#####################  paramètre fig  #######################
#############################################################

fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('white')

ax = plt.gca()
ax.set_xlim(-d, d)
ax.set_ylim(-d, d)
ax.axis('off')

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_aspect('equal', adjustable='box')
ax.set_facecolor('white')

#q = ax.quiver(x,y,Ex_tot[0],Ey_tot[0], M_tot[0], cmap = "inferno")
#plt.scatter(X,Y, c = Q, cmap = "jet", vmin = -2, vmax = 2)
q = ax.quiver(x,z,Ex,Ez, color = "black")
masse, = ax.plot([], [],"o",color = "red", markersize = 3)  #initialisation masse 1

#############################################################
#######################  Animation  #########################
#############################################################

def animate(n,X,Z):

    masse.set_data([X[n],Z[n]])

    #label.set_text(str(10 * n) + " ms")
    q.set_UVC(Ex * np.cos( omega * (t[n] - r/c )) ,Ez * np.cos( omega * (t[n] - r/c )) )

    return (masse, q)

anim = animation.FuncAnimation(fig, animate, frames = int(len(t)), fargs = (X,Z),
                               interval = 1, blit = False)

#writervideo = animation.FFMpegWriter(fps=40)
#anim.save('anim_ce_bat.mp4',writer=writervideo)

plt.show()