import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
txt = "C:\ENSTA\Projet_Description_des_fonds_sous_marin\Code\Donnees_artificielles-20250507\double_sin.txt"

mnt = np.loadtxt(txt)
print(mnt)
print(mnt.shape)

taille = mnt.shape[0]

x = np.arange(0,taille)
y = np.arange(0,taille)
X,Y = np.meshgrid(x,y)

cmap = plt.cm.gist_earth
img = plt.contourf(X, Y, mnt, levels=100, cmap=cmap)
plt.contour(X, Y, mnt, levels=5, colors='black')
plt.title('Double sinus')
plt.colorbar(img, label='Altitude [m]')
#plt.show()

##Calcul des caractéristiques du terrain de double sin

profondeur_min = np.min(mnt)
profondeur_max = np.max(mnt)
profondeur_moyenne = np.mean(mnt)
ecart_type_profondeur = np.std(mnt)
print("__________________________________________________________")
print(f"profondeur_min = {profondeur_min}")
print(f"profondeur_max = {profondeur_max}")
print(f"profondeur_moyenne = {profondeur_moyenne}")
print(f"ecart_type_profondeur = {ecart_type_profondeur}")

##Tracé de l'histogramme

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

taille = mnt.shape[0]
x = np.arange(0, taille)
y = np.arange(0, taille)
X, Y = np.meshgrid(x, y)

ax.plot_surface(X, Y, mnt, cmap='gist_earth', edgecolor='none')
ax.set_title('Relief sous-marin en 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Profondeur [m]')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Réduction de taille pour lisibilité
step = 5
mnt_red = -np.nan_to_num(mnt[::step, ::step], nan=0)  # on inverse pour que ce soit en hauteur

X, Y = np.meshgrid(np.arange(mnt_red.shape[1]), np.arange(mnt_red.shape[0]))
Z = mnt_red
Z = np.nan_to_num(Z, nan=0)
Z[Z <= 0] = 0

# Préparer les données pour bar3d (1 valeur par barre)
x = X.flatten()
y = Y.flatten()
z = np.zeros_like(x)
dz = Z.flatten()

# Normalisation pour la colormap
norm = plt.Normalize(dz.min(), dz.max())
colors = cm.viridis(norm(dz))  # Remplace viridis par la cmap de ton choix

# Tracé
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x, y, z, 1, 1, dz, shade=True, color=colors)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Profondeur')
ax.set_title('Histogramme 3D coloré selon la profondeur')
plt.tight_layout()
plt.show()



##calcul du gradient sur 3 points (TPP)

def TPP(M): #avec M une matrice de representation des données batymétriques
    res = np.zeros(M.shape)
    for i in range(M.shape[0]):
        res[i,0]=np.nan
        res[i,M.shape[0]-1]=np.nan
    for j in range(M.shape[1]):
        res[0,j]=np.nan
        res[M.shape[0]-1,j]=np.nan
    for i in range(1,M.shape[0]-1):
        for j in range(1,M.shape[1]-1):
            f_x=(M[i,j+1]-M[i,j])/1
            f_y=(M[i-1,j]-M[i,j])/1
            p=np.sqrt(f_x**2+f_y**2)
            res[i,j]=np.arctan(p)*(180/np.pi)
    return res

#print(TPP(mnt))

def FCN(M):
    res = np.zeros(M.shape)
    for i in range(M.shape[0]):
        res[i, 0] = np.nan
        res[i, M.shape[0] - 1] = np.nan
    for j in range(M.shape[1]):
        res[0, j] = np.nan
        res[M.shape[0] - 1, j] = np.nan
    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            f_x = (M[i,j+1]-M[i,j-1])/2
            f_y = (M[i-1,j]-M[i+1,j])/2
            p = np.sqrt(f_x**2 + f_y**2)
            res[i,j] = np.arctan(p)*(180/np.pi)
    return res

#print(FCN(mnt))

def Evans(M):
    res = np.zeros(M.shape)
    for i in range(M.shape[0]):
        res[i, 0] = np.nan
        res[i, M.shape[0] - 1] = np.nan
    for j in range(M.shape[1]):
        res[0, j] = np.nan
        res[M.shape[0] - 1, j] = np.nan
    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            f_x = (M[i+1,j+1] + M[i,j+1] + M[i-1,j+1] - (M[i-1,j-1] + M[i,j-1] + M[i+1,j-1]))/6
            f_y = (M[i-1,j-1] + M[i-1,j] + M[i-1,j+1] - (M[i+1,j-1] + M[i+1,j] + M[i+1,j+1]))/6
            p = np.sqrt(f_x**2 + f_y**2)
            res [i,j] = np.arctan(p)*(180/np.pi)
    return res

#print(Evans(mnt))

def affiche_pente():
    # Calcul des pentes avec les trois méthodes
    pente_TPP = TPP(mnt)
    pente_FCN = FCN(mnt)
    pente_Evans = Evans(mnt)

    # Noms et résultats regroupés
    pentes = [pente_TPP, pente_FCN, pente_Evans]
    noms = ['TPP', 'FCN', 'Evans']
    cmap = 'magma_r'

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(3):
        im = ax[i].imshow(pentes[i], origin='lower', cmap=cmap)
        ax[i].set_title(f'{noms[i]} ({cmap})')
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente [°]', cax=cax)

    plt.suptitle('Comparaison des méthodes de calcul de pente')
    plt.tight_layout()
    plt.show()


#affiche_pente()

x = np.arange(0,taille)
y = np.arange(0,taille)
z = 5*np.sin(x/10+3*np.sin(y/20)) + 2*np.sin(y/5)
C = np.array([ [5*np.sin(j/10+3*np.sin(i/20)) + 2*np.sin(i/5) for j in range(0,101)] for i in range(0,101) ])
def courbe(M):
    res = np.zeros(M.shape)
    for i in range(M.shape[0]):
        res[i, 0] = np.nan
        res[i, M.shape[0] - 1] = np.nan
    for j in range(M.shape[1]):
        res[0, j] = np.nan
        res[M.shape[0] - 1, j] = np.nan
    for i in range(1,M.shape[0]-1):
        for j in range(1,M.shape[1]-1):
            f_x = 0.5 * np.cos(j/10 + 3 *np.sin(i/20))
            f_y = (3 / 4) * np.cos(j/20) * np.cos(j/10 + 3*np.sin(i/20)) + (2/3) *np.cos(i/5)
            p = np.sqrt(f_x ** 2 + f_y ** 2)
            res[i, j] = np.arctan(p)*(180/np.pi)
    return res

#print(courbe(C))
#print(C.shape)

def exposition(f_x,f_y):
    return np.arctan2(-f_x,-f_y)


M_erreur = courbe(mnt)-TPP(mnt)
M_erreur1 = courbe(mnt)-FCN(mnt)
M_erreur2 = courbe(mnt)-Evans(mnt)
print("__________________________________________________________")
print("Ecart type de l'erreur entre théorique et expérimental")
print(np.nanstd(M_erreur))
print(np.nanstd(M_erreur1))
print(np.nanstd(M_erreur2))
print("__________________________________________________________")
print("Moyenne de l'erreur entre théorique et expérimental")

print(np.nanmean(M_erreur))
print(np.nanmean(M_erreur1))
print(np.nanmean(M_erreur2))


