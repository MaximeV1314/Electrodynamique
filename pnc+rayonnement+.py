import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import os
import glob
import pandas as pd

mpl.use('Agg')

#############################################################
#####################  fonctions  ###########################
#############################################################

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = 'img/'+i+'.png'

    return(i)

def a(vi, v, m_tabi, i):
    ax = 0
    ay = 0
    k = 0
    eps = 0.1
    for p in v:
        r2 = np.sqrt((vi[2]-p[2][i])**2 + (vi[3]-p[3][i])**2+eps)

        ax = ax - m_tabi[k] * (vi[2] - p[2][i])/r2**3
        ay = ay - m_tabi[k] * (vi[3] - p[3][i])/r2**3
        k = k+1


    return np.array([ax, ay, vi[0], vi[1]])

#@jit(nopython=True)
def rkd(derivee, step, debut, fin, v_int, m_tab, Ex, Ey):
    """
    Méthode Runge-Kutta à l'ordre 4 pour un système de deux équations différentielles
    """
    N = len(v_int)
    t = np.arange(debut,fin,step)
    v_tot = []
    b = np.zeros((2, t.shape[0]))

    for k in range(N):

        vk = np.zeros((4, t.shape[0]))
        vk[:, 0] = v_int[k]             #initialisation masse 1
        v_tot.append(vk)

    for i in range(t.shape[0] - 1):
        l = 0
        xb = 0
        yb = 0

        for vi in v_tot:

            v = list(v_tot)
            del v[l]
            m_tabi = list(m_tab)
            del m_tabi[l]

            d1 = derivee(vi[:, i], v, m_tabi, i, Ex, Ey)
            d2 = derivee(vi[:, i] + d1 * step / 2, v, m_tabi, i, Ex, Ey)
            d3 = derivee(vi[:, i] + d2 * step / 2, v, m_tabi, i, Ex, Ey)
            d4 = derivee(vi[:, i] + d3 * step, v, m_tabi, i, Ex, Ey)
            vi[:, i + 1] = vi[:, i] + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)

            xb = xb + vi[2][i] * m_tab[l]
            yb = yb + vi[3][i] * m_tab[l]

            if abs(vi[2, i + 1]) >= d:
                vi[0, i + 1] = -vi[0, i + 1]

            if abs(vi[3, i + 1]) >= d:
                vi[1, i + 1] = -vi[1, i + 1]

            l = l+1

        b[:, i] = [1/np.sum(m_tab) * xb, 1/np.sum(m_tab) * yb]

    # Argument de sortie
    return v_tot, b

#############################################################
##############  définition des paramètres   #################
#############################################################
"""Attention, tout est en cs ! """

c = 0.05                      #vitesse lumière
Q = 1                      #charge particule

l1 = 1
l2 = 1

#print("v/c = ", R * omega/c)

tf = 2000               #temps détude
pas = 1
t = np.arange(0,tf,pas, dtype = 'int64')
pas_img = 10                 #pas des images. FPS = 1s / (pas * pas_img)
digit = 4

d = 5              #dimension fig
N = 300              #dimension meshgrid

x_init = np.array([-1, 0, 1])
y_init = np.array([-0.5, 0, 0.5])

vx_init = np.array([0, 0, 0])
vy_init = np.array([0, 0, 0])



n_part = 3          #nombre de particule

print("initialization of variables successed")

#############################################################
#######################  calcul CE  #########################
#############################################################

t_prime_i_min2 = [np.zeros((N, N)), np.zeros((N, N))]
t_prime_i_min = [np.zeros((N, N)), np.zeros((N, N))]

x, y = np.meshgrid(np.linspace(-d,d,N),np.linspace(-d,d, N))

X_sys = np.zeros((n_part, len(t)))
Y_sys = np.zeros((n_part, len(t)))

VX_sys = np.zeros((n_part, len(t)))
VY_sys = np.zeros((n_part, len(t)))

X_min = np.zeros(n_part)
Y_min = np.zeros(n_part)
X_min2 = np.zeros(n_part)
Y_min2 = np.zeros(n_part)

Ex = 0
Ey = 0
for i in range(n_part):

    v_init.append([vx_init[i],vy_init[i],x_init[i], y_init[i]])
    #vecteur initial : [vx, vy, x, y],...
    X_sys[k, 0] = x_init[k]
    X_sys[k, 0] = y_init[k]

    VX_sys[k, 0] = vx_init[k]
    VY_sys[k, 0] = vy_init[k]
    m_tab.append((-1))

    Ex += m_tab[i] * (x - X_sys[k, 0, i]) / ((x - X_sys[k, 0, i])**2 + (y - Y_sys[k, 0, i])**2)**(3/2)
    Ey += m_tab[i] * (y - Y_sys[k, 0, i]) / ((x - X_sys[k, 0, i])**2 + (y - Y_sys[k, 0, i])**2)**(3/2)

Ex_arr.append(Ex)
Ey_arr.append(Ey)
E_arr = [(Ex_arr[0]**2 + Ey_arr[0]**2)**(1/2)]

for k in range(n_part):

    V = np.array(rkd(a, pas, 0, pas, v_init, m_tab, Ex, Ey))

    X_sys[k, 1] = V[i, 2, -1]
    Y_sys[k, 1] = V[i, 3, -1]

    VX_sys[k, 1] = V[i, 0, -1]
    VY_sys[k, 1] = V[i, 1, -1]

    t_prime_i_min2.append(np.zeros(len(OM_file)))
    t_prime_i_min.append(np.zeros(len(OM_file)))

    r0 = ((x - X_sys[k][0])**2 + (y - Y_sys[k][0])**2)**(1/2)
    r1 = ((x - X_sys[k][1])**2 + (y - Y_sys[k][1])**2)**(1/2)

    t_prime_i_min2[k] = ((np.ones((N, N)) * t[0] - r0/c)>0) * (np.ones((N, N)) * t[0] - r0/c)
    t_i_int = t_prime_i_min2[k].astype(int)
    t_i_int1 = np.copy(t_i_int) + 1

    X_min2[k] = np.take(X_sys[k], t_i_int) * (1 - (t_prime_i_min2[k] - t_i_int)) + np.take(X_sys[k], t_i_int1) * (t_prime_i_min2[k] - t_i_int)
    Y_min2[k] = np.take(Y_sys[k], t_i_int) * (1 - (t_prime_i_min2[k] - t_i_int)) + np.take(Y_sys[k], t_i_int1) * (t_prime_i_min2[k] - t_i_int)


    t_prime_i_min[k] = ((np.ones((N, N)) * t[1] - r1/c)>0) * (np.ones((N, N)) * t[1] - r1/c)
    t_i_int = t_prime_i_min[k].astype(int)
    t_i_int1 = np.copy(t_i_int) + 1

    X_min[k] = np.take(X_sys[k], t_i_int) * (1 - (t_prime_i_min[k] - t_i_int)) + np.take(X_sys[k], t_i_int1) * (t_prime_i_min[k] - t_i_int)
    Y_min[k] = np.take(Y_sys[k], t_i_int) * (1 - (t_prime_i_min[k] - t_i_int)) + np.take(Y_sys[k], t_i_int1) * (t_prime_i_min[k] - t_i_int)

for i in range(2, len(t)):
    Ex = 0
    Ey = 0

    for k in range(n_part):

        r = ((x - X_sys[k][i])**2 + (y - Y_sys[k][i])**2)**(1/2)

        rx = (x - X_sys[k][i])/r
        ry = (y - Y_sys[k][i])/r

        t_prime_i = ((np.ones((N, N)) * t[i] - r/c)>0) * (np.ones((N, N)) * t[i] - r/c)
        t_i_int = t_prime_i.astype(int)
        t_i_int1 = np.copy(t_i_int) + 1

        X = np.take(X_sys[k], t_i_int) * (1 - (t_prime_i - t_i_int)) + np.take(X_sys[k], t_i_int1) * (t_prime_i - t_i_int)
        Y = np.take(Y_sys[k], t_i_int) * (1 - (t_prime_i - t_i_int)) + np.take(Y_sys[k], t_i_int1) * (t_prime_i - t_i_int)

        Vx = (X - X_min[k])/pas
        Vy = (Y - Y_min[k])/pas
        Ax = (Vx - (X_min[k]-X_min2[k])/pas)/pas
        Ay = (Vy - (Y_min[k]-Y_min2[k])/pas)/pas

        gamma_sq = 1/(1 - (Vx**2 + Vy**2)/c**2)

        #print(np.max((Vx**2 + Vy**2)/c**2))

        Ex += Q/(r*c**2) * ((rx * Ax + ry * Ay) * (rx - Vx/c) - (rx * (rx - Vx/c) + ry * (ry - Vy/c)) * Ax)/(1 - (rx * Vx + ry * Vy)/c)**3    #terme lointain
        Ex += Q/r**2 * (rx - Vx/c)/( gamma_sq * (1 - (rx * Vx + ry * Vy)/c)**3)                                                             #terme proche
        Ey += Q/(r*c**2) * ((rx * Ax + ry * Ay) * (ry - Vy/c) - (rx * (rx - Vx/c) + ry * (ry - Vy/c)) * Ay)/(1 - (rx * Vx + ry * Vy)/c)**3
        Ey += Q/r**2 * (ry - Vy/c)/( gamma_sq * (1 - (rx * Vx + ry * Vy)/c)**3)

        t_prime_i_min2[k] = np.copy(t_prime_i_min[k])
        t_prime_i_min[k] = np.copy(t_prime_i)
        X_min2[k] = np.copy(X_min[k])
        X_min[k] = np.copy(X)
        Y_min2[k] = np.copy(Y_min[k])
        Y_min[k] = np.copy(Y)

    E = (Ex**2 + Ey**2) **(1/2)
    Ex_arr.append(Ex)
    Ey_arr.append(Ey)
    E_arr.append(E)


E_arr = np.array(E_arr)

print("Calculation of Electric field successed")

#############################################################
#######################  Animation  #########################
#############################################################


extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

#Emin = np.mean(np.min(E_arr, axis = 1))
#Emax = np.mean(np.max(E_arr, axis = 1))

Emin = np.min(E_arr[:])
Emax = np.max(E_arr[:])

for i in range(0, len(t)-2, pas_img):

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('black')

    ax = plt.subplot(111)
    #im = ax.imshow(E_arr[i], cmap = "seismic", vmin = Emin, vmax =  Emax, extent = (-d, d, -d, d), interpolation = "gaussian")
    im = ax.imshow(E_arr[i], cmap = "seismic", norm=colors.LogNorm(vmin = Emin, vmax = Emax), extent = (-d, d, -d, d), interpolation = "gaussian")    #CMRmap, turbo, interpolation = "gaussian"

    #ax.plot([0, R * np.sin( theta_0 * np.sin(omega * t[i+2]))], [0,  -R * np.cos( theta_0 * np.sin(omega * t[i+2]))], "-", color = "black")
    #ax.plot([0,X_sys[0][i]],[0,-Y_sys[0][i]], "-o", markeredgecolor = "black", markerfacecolor = "white", color = "black")
    #ax.plot([X_sys[0][i],X_sys[1][i]],[-Y_sys[0][i],-Y_sys[1][i]], "-o", markeredgecolor = "black", markerfacecolor = "white", color = "black")

    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_aspect('equal', adjustable='box')
    #ax.set_facecolor('white')

    name_pic = name(int(i/pas_img), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)
    ax.clear()
    plt.close(fig)
    print(i/len(t))

print("Images successed")

  #ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
