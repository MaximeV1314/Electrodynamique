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

#############################################################
##############  définition des paramètres   #################
#############################################################
"""Attention, tout est en cs ! """

c = 0.20                      #vitesse lumière
Q = 1                      #charge particule

l1 = 1
l2 = 1

#print("v/c = ", R * omega/c)

tf = 2000               #temps détude
pas = 1
t = np.arange(0,tf,pas, dtype = 'int64')
pas_img = 1                 #pas des images. FPS = 1s / (pas * pas_img)
digit = 4

d = 10              #dimension fig
N = 300              #dimension meshgrid

print("initialization of variables successed")

#############################################################
#######################  calcul CE  #########################
#############################################################

theta1 = np.genfromtxt('file_dp_1.csv', delimiter=',')
theta2 = np.genfromtxt('file_dp_2.csv', delimiter=',')

t_prime_i_min2 = [np.zeros((N, N)), np.zeros((N, N))]
t_prime_i_min = [np.zeros((N, N)), np.zeros((N, N))]

X_sys = [np.zeros((len(theta1))), np.zeros((len(theta1)))]
Y_sys = [np.zeros((len(theta1))), np.zeros((len(theta1)))]
X_min = [np.zeros((len(theta1))), np.zeros((len(theta1)))]
Y_min = [np.zeros((len(theta1))), np.zeros((len(theta1)))]
X_min2 = [np.zeros((len(theta1))), np.zeros((len(theta1)))]
Y_min2 = [np.zeros((len(theta1))), np.zeros((len(theta1)))]

X_sys[0] = l1*np.sin(theta1)
Y_sys[0] = l1*np.cos(theta1)
X_sys[1] = X_sys[0] + l2*np.sin(theta2)
Y_sys[1] =  Y_sys[0] + l2*np.cos(theta2)

x, y = np.meshgrid(np.linspace(-d,d,N),np.linspace(-d,d, N))

Ex_arr = []
Ey_arr = []
E_arr = []


for k in range(2):

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

    for k in range(2):

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

Emin = np.min(E_arr)
Emax = np.max(E_arr)

for i in range(0, len(t)-2, pas_img):

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('black')

    ax = plt.subplot(111)
    #im = ax.imshow(E_arr[i], cmap = "seismic", vmin = Emin, vmax =  Emax, extent = (-d, d, -d, d), interpolation = "gaussian")
    im = ax.imshow(E_arr[i], cmap = "seismic", norm=colors.LogNorm(vmin = Emin, vmax = Emax), extent = (-d, d, -d, d), interpolation = "gaussian")    #CMRmap, turbo, interpolation = "gaussian"

    #ax.plot([0, R * np.sin( theta_0 * np.sin(omega * t[i+2]))], [0,  -R * np.cos( theta_0 * np.sin(omega * t[i+2]))], "-", color = "black")
    ax.plot([0,X_sys[0][i]],[0,-Y_sys[0][i]], "-o", markeredgecolor = "black", markerfacecolor = "white", color = "black")
    ax.plot([X_sys[0][i],X_sys[1][i]],[-Y_sys[0][i],-Y_sys[1][i]], "-o", markeredgecolor = "black", markerfacecolor = "white", color = "black")

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
