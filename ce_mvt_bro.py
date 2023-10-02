import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import os
import glob
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
import scipy.signal as signal

#mpl.use('Agg')

_, _, files = next(os.walk("file"))
file_count = int(20000/20)

X_sys = np.zeros((file_count,))
Y_sys = np.zeros((file_count,))
X_tot = []
Y_tot = []
k = 0
for i in range(0, 20000, 20):
    OM = np.genfromtxt('file/' + str(i) + ".csv", delimiter=',')

    X_sys[k] = OM[0, 0]
    Y_sys[k] = - OM[0, 1]

    X_tot.append(OM[:, 0])
    Y_tot.append(-OM[:, 1])

    k += 1


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
c = 0.13                      #vitesse lumière
Q = 1                      #charge particule

tf = int(len(X_sys)/1)               #temps détude
pas = 1
t = np.arange(0,tf,pas, dtype = 'int64')
pas_img = 1                 #pas des images. FPS = 1s / (pas * pas_img)
digit = 4

d = 5              #dimension fig
N = 300              #dimension meshgrid

print("initialization of variables successed")

#############################################################
#######################  calcul CE  #########################
#############################################################


x, y = np.meshgrid(np.linspace(-d,d,N),np.linspace(-d,d, N))
r0 = ((x - X_sys[0])**2 + (y - Y_sys[0])**2)**(1/2)
r1 = ((x - X_sys[1])**2 + (y - Y_sys[1])**2)**(1/2)

Ex_arr = []
Ey_arr = []
E_arr = []

t_prime_i_min2 = ((np.ones((N, N)) * t[0] - r0/c)>0) * (np.ones((N, N)) * t[0] - r0/c)
t_i_int = t_prime_i_min2.astype(int)
t_i_int1 = np.copy(t_i_int) + 1
X_min2 = np.take(X_sys, t_i_int) * (1 - (t_prime_i_min2 - t_i_int)) + np.take(X_sys, t_i_int1) * (t_prime_i_min2 - t_i_int)
Y_min2 = np.take(Y_sys, t_i_int) * (1 - (t_prime_i_min2 - t_i_int)) + np.take(Y_sys, t_i_int1) * (t_prime_i_min2 - t_i_int)

t_prime_i_min = ((np.ones((N, N)) * t[1] - r1/c)>0) * (np.ones((N, N)) * t[1] - r1/c)
t_i_int = t_prime_i_min.astype(int)
t_i_int1 = np.copy(t_i_int) + 1
X_min = np.take(X_sys, t_i_int) * (1 - (t_prime_i_min - t_i_int)) + np.take(X_sys, t_i_int1) * (t_prime_i_min - t_i_int)
Y_min = np.take(Y_sys, t_i_int) * (1 - (t_prime_i_min - t_i_int)) + np.take(Y_sys, t_i_int1) * (t_prime_i_min - t_i_int)


for i in range(2, len(t)):
    r = ((x - X_sys[i])**2 + (y - Y_sys[i])**2)**(1/2)

    rx = (x - X_sys[i])/r
    ry = (y - Y_sys[i])/r

    t_prime_i = ((np.ones((N, N)) * t[i] - r/c)>0) * (np.ones((N, N)) * t[i] - r/c)
    t_i_int = t_prime_i.astype(int)
    t_i_int1 = np.copy(t_i_int) + 1

    X = np.take(X_sys, t_i_int) * (1 - (t_prime_i - t_i_int)) + np.take(X_sys, t_i_int1) * (t_prime_i - t_i_int)
    Y = np.take(Y_sys, t_i_int) * (1 - (t_prime_i - t_i_int)) + np.take(Y_sys, t_i_int1) * (t_prime_i - t_i_int)

    Vx = (X - X_min)/pas
    Vy = (Y - Y_min)/pas
    Ax = (Vx - (X_min-X_min2)/pas)/pas
    Ay = (Vy - (Y_min-Y_min2)/pas)/pas

    gamma_sq = 1/(1 - (Vx**2 + Vy**2)/c**2)

    #print(np.max((Vx**2 + Vy**2)/c**2))

    Ex = Q/(r*c**2) * ((rx * Ax + ry * Ay) * (rx - Vx/c) - (rx * (rx - Vx/c) + ry * (ry - Vy/c)) * Ax)/(1 - (rx * Vx + ry * Vy)/c)**3    #terme lointain
    Ex += Q/r**2 * (rx - Vx/c)/( gamma_sq * (1 - (rx * Vx + ry * Vy)/c)**3)                                                             #terme proche
    Ey = Q/(r*c**2) * ((rx * Ax + ry * Ay) * (ry - Vy/c) - (rx * (rx - Vx/c) + ry * (ry - Vy/c)) * Ay)/(1 - (rx * Vx + ry * Vy)/c)**3
    Ey += Q/r**2 * (ry - Vy/c)/( gamma_sq * (1 - (rx * Vx + ry * Vy)/c)**3)

    E = (Ex**2 + Ey**2) **(1/2)
    Ex_arr.append(Ex)
    Ey_arr.append(Ey)
    E_arr.append(E)

    t_prime_i_min2 = np.copy(t_prime_i_min)
    t_prime_i_min = np.copy(t_prime_i)
    X_min2 = np.copy(X_min)
    X_min = np.copy(X)
    Y_min2 = np.copy(Y_min)
    Y_min = np.copy(Y)


E_arr = np.array(E_arr)

print("Calculation of Electric field successed")

l = len(t) - 2
T = 20/l
tt = np.linspace(0, 20, l)

xf = fftfreq(l, T)
yf = np.abs(fft(E_arr[:, 150, 50]))**2

#f, Pwelch_spec = signal.welch(E_arr[:, 150, 50], l, scaling='spectrum')

#plt.semilogy(f, Pwelch_spec)
plt.plot(xf[:l//2], yf[:l//2], "-", color = "blue")
#plt.xlabel("temps")
#plt.ylabel("||E||")
#plt.title("intensité CE point (150,50) en fonction du temps")
plt.grid()
plt.show()

#############################################################
#######################  Animation  #########################
#############################################################

"""
extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

#Emin = np.mean(np.min(E_arr, axis = 1))
#Emax = np.mean(np.max(E_arr, axis = 1))

Emin = np.min(E_arr)
Emax = np.max(E_arr)

ones = np.ones((len(X_tot[0]) - 1,))
m = np.concatenate((np.array([2300]), ones * 35))
col = np.concatenate((np.array([0]), ones *0.3))

for i in range(0, len(t)-2, pas_img):

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('black')

    ax = plt.subplot(121)
    #im = ax.imshow(E_arr[i], cmap = "seismic", vmin = Emin, vmax = 1/600 * Emax, extent = (-d, d, -d, d), interpolation = "gaussian")
    im = ax.imshow(E_arr[i], cmap = "seismic", norm=colors.LogNorm(vmin = Emin, vmax = Emax), extent = (-d, d, -d, d), interpolation = "gaussian")    #CMRmap, turbo, interpolation = "gaussian"
    #ax.plot(X_sys[i], Y_sys[i], "o", markeredgecolor = "black", markerfacecolor = "white", markersize = 20)

    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #ax.set_title("Radiation of brownian particle", color = "white")
    ax.set_aspect('equal', adjustable='box')
    #ax.set_facecolor('white')


    ax1 = plt.subplot(122)

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim( -d, d)
    ax1.set_ylim( -d, d)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    #ax1.set_title("Motion of brownian particle", color = "white")

    circle1 = plt.Circle((0, 0), 30, color='black', ec = "white", lw = 10, zorder = -2)    #cercle
    ax1.add_patch(circle1)

    ax1.plot(X_sys[:i], -Y_sys[:i], "-", color = 'white', zorder = 1)
    ax1.scatter(X_tot[i][:], -Y_tot[i][:],c = col, s = m, cmap = "rainbow", ec = "white", vmin = 0, vmax = 2)

    plt.subplots_adjust(wspace=0, hspace=0)

    name_pic = name(int(i/pas_img), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    ax1.clear()
    plt.close(fig)
    print(i/len(t))

"""
print("Images successed")

  #ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
