import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

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

freq = 2                    #période
omega = 2*np.pi * freq      #pulsation
R = 0.1                     #amplitude du dipole
c = 10                       #vitesse lumière
Q = -1                      #charge particule


"""Attention :
    pour être dans l'approximation dippolaire, R << d
    pour être dans le cadre non relativiste : omega * R << c"""

tf = 2*np.pi                #temps détude
pas = 0.1
t = np.arange(0,tf,pas)
pas_img = 5                 #pas des images. FPS = 1s / (pas * pas_img)
digit = 4

d = 1000               #dimension fig
n = 500              #dimension meshgrid

print("initialization of variables successed")

#############################################################
#######################  calcul CE  #########################
#############################################################

x, y = np.meshgrid(np.linspace(-d,d,n),np.linspace(-d,d, n))

X = np.zeros(len(t))
Y = R * np.sin(omega * pas * t)

Ex_arr = []
Ey_arr = []
E_arr = []

r_prime_m2 = ((X[0] - x)**2 + (Y[0] - y)**2 )**(1/2)
ur_prime_x_m2 = (x - X[0])/r_prime_m2
ur_prime_y_m2 = (y - Y[0])/r_prime_m2

r_prime_m1 = ((X[1] - x)**2 + (Y[1] - y)**2 )**(1/2)
ur_prime_x_m1 = (x - X[1])/r_prime_m1
ur_prime_y_m1 = (y - Y[1])/r_prime_m1

for i in range(1, len(t)):

    r_prime = ((X[i] - x)**2 + (Y[i] - y)**2 )**(1/2)   #distance meshgrid
    t_prime = t[i] - r_prime/c                          #temps meshgrid

    ur_prime_x = (x - X[i])/r_prime
    ur_prime_y = (y - Y[i])/r_prime

    ur_prime_mid_x = ((ur_prime_x/r_prime**2) -  (ur_prime_x_m1/r_prime_m1**2))/pas
    ur_prime_mid_y = ((ur_prime_y/r_prime**2) -  (ur_prime_y_m1/r_prime_m1**2))/pas

    ur_prime_far_x = (ur_prime_x_m1 + ur_prime_x_m2 - 2 * ur_prime_x)/pas**2
    ur_prime_far_y = (ur_prime_y_m1 + ur_prime_y_m2 - 2 * ur_prime_y)/pas**2

    #Ex = Q * ( (ur_prime_x / r_prime**2) + (r_prime/c * ur_prime_mid_x) + 1/c**2 * ur_prime_far_x)
    #Ey = Q * ( (ur_prime_y / r_prime**2) + (r_prime/c * ur_prime_mid_y) + 1/c**2 * ur_prime_far_y)
    Ex = Q * 1/c**2 * ur_prime_far_x
    Ey = Q * 1/c**2 * ur_prime_far_y
    E = (Ex**2 + Ey**2)**(1/2)

    E_arr.append(E)

    ur_prime_x_m2 = ur_prime_x_m1
    ur_prime_x_m1 = ur_prime_x
    ur_prime_y_m2 = ur_prime_y_m1
    ur_prime_y_m1 = ur_prime_y

    r_prime_m2 = r_prime_m1
    r_prime_m1 = r_prime


E_arr = np.array(E_arr)

print("Calculation of Electric field successed")

#############################################################
#######################  Animation  #########################
#############################################################

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

Emin = 0
Emax = 1/100 * np.max(E_arr)
for i in range(0, len(t), pas_img):
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('black')

    ax = plt.gca()
    ax.imshow(E_arr[i], vmin = Emin, vmax = Emax)

    ax.axis('off')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')

    name_pic = name(int(i/pas_img), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    plt.close(fig)
    print(i/len(t))

print("Images successed")

  #ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"


