import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as sp
from scipy.ndimage import convolve
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
# Fonctions de calcul des indicateurs
#from fonctions_indicateurs import *

txt = "C:\ENSTA\Projet_Description_des_fonds_sous_marin\Code\Donnees_artificielles-20250507\double_sin.txt"
txt1 = "C:\ENSTA\Projet_Description_des_fonds_sous_marin\Code\Donnees_artificielles-20250507\sin_card.txt"
txt2 = "C:\ENSTA\Projet_Description_des_fonds_sous_marin\Code\Donnees_artificielles-20250507\plan.txt"
txt3 = "C:\ENSTA\Projet_Description_des_fonds_sous_marin\Code\Donnees_artificielles-20250507\plateau.txt"
mnt = np.loadtxt(txt3)
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
#print("__________________________________________________________")
#print(f"profondeur_min = {profondeur_min}")
#print(f"profondeur_max = {profondeur_max}")
#print(f"profondeur_moyenne = {profondeur_moyenne}")
#print(f"ecart_type_profondeur = {ecart_type_profondeur}")

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
#plt.show()


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
#plt.show()



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

def exposition(M):
    res = np.zeros(M.shape)
    for i in range(M.shape[0]):
        res[i, 0] = np.nan
        res[i, M.shape[0] - 1] = np.nan
    for j in range(M.shape[1]):
        res[0, j] = np.nan
        res[M.shape[0] - 1, j] = np.nan
    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            f_x = 0.5 * np.cos(j / 10 + 3 * np.sin(i / 20))
            f_y = (3 / 4) * np.cos(j / 20) * np.cos(j / 10 + 3 * np.sin(i / 20)) + (2 / 3) * np.cos(i / 5)
            res [i, j] = np.arctan2(-f_x,-f_y)
    return res

#print(f"l'exposition pour le double sin est {exposition(mnt)}")

M_erreur = courbe(mnt)-TPP(mnt)
M_erreur1 = courbe(mnt)-FCN(mnt)
M_erreur2 = courbe(mnt)-Evans(mnt)
#print("__________________________________________________________")
#print("Ecart type de l'erreur entre théorique et expérimental")
#print(np.nanstd(M_erreur))
#print(np.nanstd(M_erreur1))
#print(np.nanstd(M_erreur2))
#print("__________________________________________________________")
#print("Moyenne de l'erreur entre théorique et expérimental")

#print(np.nanmean(M_erreur))
#print(np.nanmean(M_erreur1))
#print(np.nanmean(M_erreur2))


##Calcul du BPI (forme de disque)

def BPI(M, r):
    x = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, x)
    D = np.sqrt(X**2 + Y**2)
    filtre = (D <= r).astype(float)
    filtre /= np.sum(filtre)  # Normalisation

    # Convolution de l'image avec le noyau
    filtered = convolve(M, filtre, mode='constant')

    # Soustraction : M - composante lissée
    res = M - filtered
    return res

#print(f"Le BPI pour un rayon de 5 est {BPI(mnt,5)}")

##calcul du BPI (forme d'anneau)

def BPI_anneau(M,r1,r2):
    if r1>r2:
        x = np.arange(-r1, r1 + 1)
        X, Y = np.meshgrid(x, x)
        D = np.sqrt(X ** 2 + Y ** 2)
        filtre = ((D <= r1) & (D>=r2)).astype(float)
        filtre /= np.sum(filtre)  # Normalisation

        # Convolution de l'image avec le noyau
        filtered = convolve(M, filtre, mode='constant')

        # Soustraction : M - composante lissée
        res = M - filtered
    else:
        x = np.arange(-r2, r2 + 1)
        X, Y = np.meshgrid(x, x)
        D = np.sqrt(X ** 2 + Y ** 2)
        filtre = ((D <= r2) & (D >= r1)).astype(float)
        filtre /= np.sum(filtre)  # Normalisation

        # Convolution de l'image avec le noyau
        filtered = convolve(M, filtre, mode='constant')

        # Soustraction : M - composante lissée
        res = M - filtered
    return res

#print(BPI_anneau(mnt,20,25))

##calcul du BPI (forme de secteur)
def BPI_secteur(M, r, theta_init, theta_max):
    # Création de la grille
    x = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, x)
    D = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)  # correction ici

    # Création du masque angulaire
    if theta_init <= theta_max:
        masque_angle = (angle >= theta_init) & (angle <= theta_max)
    else:
        # Cas où l’intervalle traverse la discontinuité -π/π
        masque_angle = (angle >= theta_init) | (angle <= theta_max)

    filtre = ((D <= r) & masque_angle).astype(float)
    filtre /= np.sum(filtre)  # normalisation du filtre

    # Application du filtre
    filtered = convolve(M, filtre, mode='reflect')
    res = M - filtered
    return res



def show_BPI(BPI):
    #plt.imshow(BPI, cmap='RdBu')
    #plt.title('BPI')
    #plt.colorbar()
    #plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(BPI.shape[0]), np.arange(BPI.shape[1]))
    surf = ax.plot_surface(X, Y, BPI, cmap='magma', edgecolor='none')
    ax.set_title('BPI')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(surf, ax=ax, label='Intensité')
    plt.show()

#print(f"Le BPI est : {BPI(mnt)}")

theta1 = np.deg2rad(0)
theta2 = np.deg2rad(45)
#show_BPI(BPI(mnt,5))
#show_BPI(BPI_anneau(mnt,20,25))
#show_BPI(BPI_secteur(mnt,5,theta1,theta2))

##Calcul de la rugosité (ecart type des profondeurs)

def rugosite(M,n):
    return sp.generic_filter(M, function=np.std, size=(n,n))

def show_rugosite(M):
    #plt.imshow(M, cmap='RdBu')
    #plt.title('Rugosité')
    #plt.colorbar()
    #plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))
    ax.plot_surface(X, Y, M, cmap='RdBu', edgecolor='none')
    ax.set_title('Rugosité')
    plt.show()

#print(rugosite(mnt,3))
#show_rugosite(rugosite(mnt,3))

##calcul de la rugosité (écart type des différences entre la profondeur du MNT et la profondeur du MNT lissé)

def terrain_lisse(M):
    return sp.gaussian_filter(M, sigma=1) #sigma =1 est pour une fenetre de 3*3

def rugosite2(M,n):
    diff = M - terrain_lisse(M)
    return sp.generic_filter(diff, function=np.std ,size=(n,n))

def show_rugosite2(M):
    #plt.imshow(M, cmap='RdBu')
    #plt.title('Rugosité')
    #plt.colorbar()
    #plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))
    ax.plot_surface(X, Y, M, cmap='RdBu', edgecolor='none')
    ax.set_title('Rugosité')
    plt.show()

#show_rugosite2(rugosite2(mnt,3))



def classifier_bathymetrie_BBPI(B_BPI, F_BPI, pente ,seuil=0.5, stdv=1):
    """
    Classe les zones bathymétriques en 10 classes selon le B-BPI et F-BPI.

    Paramètres :
    - B_BPI : array 2D du Broad-scale BPI
    - F_BPI : array 2D du Fine-scale BPI
    - seuil : seuil de classification (en unités d'écart-type)
    - stdv : écart-type des BPI (mettre 1 si déjà standardisé)

    Renvoie :
    - classes : array 2D contenant les classes 0 à 9 :
        0 = large dépression
        1 = large crête
        2 = plat / pente intermédiaire
        3 = dépression étroite dans une grande dépression
        4 = crête étroite dans une grande dépression
        5 = plat dans une grande dépression
        6 = crête étroite dans une grande crête
        7 = dépression locale dans une crête
        8 = crête locale dans une pente
        9 = dépression locale dans une pente
    """
    classes = np.full_like(B_BPI, fill_value=2, dtype=int)  # par défaut = plat

    mask_depression = B_BPI <= -seuil * stdv
    mask_crest = B_BPI >= seuil * stdv
    mask_flat = np.abs(B_BPI) < seuil * stdv

    # Large structures
    classes[mask_depression] = 0
    classes[mask_crest] = 1

    # Détailler les dépressions
    classes[np.logical_and(mask_depression, F_BPI <= -seuil * stdv)] = 3
    classes[np.logical_and(mask_depression, F_BPI >= seuil * stdv)] = 4
    classes[np.logical_and(mask_depression, np.abs(F_BPI) < seuil * stdv)] = 5

    # Détailler les crêtes
    classes[np.logical_and(mask_crest, F_BPI >= seuil * stdv)] = 6
    classes[np.logical_and(mask_crest, F_BPI <= -seuil * stdv)] = 7

    # Détail des zones plates
    classes[np.logical_and(mask_flat, F_BPI >= seuil * stdv)] = 8
    classes[np.logical_and(mask_flat, F_BPI <= -seuil * stdv)] = 9

    # Détail des zones plates (ajout pente)
    #mask_flat_crest_steep = np.logical_and.reduce((mask_flat, F_BPI >= seuil * stdv, pente > 5))
    #mask_flat_crest_gentle = np.logical_and.reduce((mask_flat, F_BPI >= seuil * stdv, pente <= 5))
    #mask_flat_depr_steep = np.logical_and.reduce((mask_flat, F_BPI <= -seuil * stdv, pente > 5))
    #mask_flat_depr_gentle = np.logical_and.reduce((mask_flat, F_BPI <= -seuil * stdv, pente <= 5))

    #classes[mask_flat_crest_steep] = 10
    #classes[mask_flat_crest_gentle] = 11
    #classes[mask_flat_depr_steep] = 12
    #classes[mask_flat_depr_gentle] = 13

    return classes



def show_pente(M):
    pente = Evans(M)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pente, cmap=cmap)
    plt.colorbar(im, label='Pente (°)')
    plt.title('Carte des pentes')
    plt.xlabel('X (colonne)')
    plt.ylabel('Y (ligne)')
    plt.tight_layout()
    plt.show()


####Travail sur zone1

##Extraction des données

zone1=np.loadtxt("C:\ENSTA\Projet_Description_des_fonds_sous_marin\Code\Zone 1-20250516\z_Zone1_8m.txt")
zone_1 = zone1[:,:-1]
taille1 = zone_1.shape[0]

x1 = np.arange(0,taille1)
y1 = np.arange(0,taille1)
X1,Y1 = np.meshgrid(x1,y1)

print(zone_1.shape)
cmap = plt.cm.magma
img = plt.contourf(X1, Y1, zone_1, levels=100, cmap=cmap)
plt.contour(X1, Y1, zone_1, levels=8, colors='black')
plt.title('Zone 1')
plt.colorbar(img, label='Altitude [m]')
plt.show()

print(f"La pente est {Evans(zone_1)}")
print(f"Le BPI est {BPI(zone_1, 50)}")

show_BPI(BPI(zone_1, 50))
show_pente(zone_1)


from matplotlib.colors import ListedColormap

from matplotlib.patches import Patch

def afficher_classes(classes):
    """
    Affiche une matrice de classes avec des couleurs personnalisées (10 classes).
    Une légende est ajoutée à droite avec le nom de chaque classe.
    """
    couleurs = ([
        'darkred', 'gold', 'white', 'red', 'orange',
        'pink', 'darkorange', 'purple', 'lightgreen', 'lightblue',
        'lime'])
        #, 'lightyellow', 'teal', 'skyblue')]

    labels = ([
        'Large dépression', 'Large crête', 'Plat',
        'Narrow dépression', 'Narrow crête in dépression',
        'Plat in dépression', 'Narrow crête',
        'Dépression sur crête', 'Crête dans pente',
        'Dépression dans pente',
        'Crête pente forte'])
        #, 'Crête pente faible'])
        #'Dépression pente forte', 'Dépression pente faible']

    cmap = ListedColormap(couleurs)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(classes, cmap=cmap, origin='lower')
    plt.title('Classification bathymétrique')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Créer les patches pour la légende
    legend_elements = [Patch(facecolor=couleurs[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]

    # Afficher la légende à droite
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()


#for i in range(5,21):
#    afficher_classes(classifier_bathymetrie_BBPI(BPI(zone_1, i)))

#afficher_classes(classifier_bathymetrie_BBPI(BPI(zone_1, 20), BPI(zone_1, 10)))
#afficher_classes(classifier_bathymetrie_BBPI(BPI(zone_1, 90), BPI(zone_1, 11),Evans(zone_1)))

#for i  in range(1,15):
#    afficher_classes(classifier_bathymetrie_BBPI(BPI(zone_1, 40), BPI(zone_1, i), Evans(zone_1)))


##Machine learning

x = np.arange(zone_1.shape[1])
y = np.arange(zone_1.shape[0])
X, Y = np.meshgrid(x, y)

print('Pente FCN')
p_evans = Evans(zone_1)

print('W BPI')
wbpi = BPI(zone_1, 20)#B_BPI pour rayon de 20
print('F BPI')
fbpi = BPI(zone_1, 10)##F_BPI pour rayon de 10
print('Rugosité')
rug = rugosite(zone_1, 15)
# Création d'un DataFrame pour stocker l'ensemble des colonnes
df = pandas.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': zone_1.flatten(),
                       'p': p_evans.flatten(), 'wbpi': wbpi.flatten(),
                       'fbpi': fbpi.flatten(),'rug' : rug.flatten()})
# Suppression des lignes contenant des valeurs non définies
data = df.dropna().copy()

# Choix du nombre de classes :
n = 11
# Le paramètre random_state permet d'avoir des classifications reproductibles
kmeans = cluster.KMeans(n_clusters=n, n_init='auto', random_state=42)
# Colonnes sélectionnées pour la classification
select = data[['z', 'p', 'wbpi', 'fbpi', 'rug']]

# Mise à l'échelle des données
scaler = StandardScaler()
data_ok = scaler.fit_transform(select)

# Création des n classes
kmeans.fit(data_ok)
# Ajout du résultat comme nouvelle colonne
df[f'kmeans_{n}'] = pandas.Series(kmeans.labels_, index=data.index)
# Mise en forme des résultats sous forme de matrice
mat = df.pivot(columns='x', index='y')
# Répartition des classes :
print('Classes : ', df[['x', f'kmeans_{n}']].groupby(f'kmeans_{n}').count())
# Mise en forme des résultats sous forme de matrice
mat = df.pivot(columns='x', index='y')

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

cmap = plt.cm.cubehelix
ls = LightSource(azdeg=-45, altdeg=35)
# Si la matrice contient des NaN, utiliser un tableau masqué
mnt_mask = np.ma.masked_invalid(mnt)
im = ax[0].imshow(mnt, origin='lower', cmap=cmap)
# Création de la carte ombrée
rgb = ls.shade(mnt_mask, cmap=cmap, vert_exag=4, blend_mode='soft')
p = ax[0].imshow(rgb, origin='lower', cmap=cmap)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
# Pour la colorbar, reprendre le "im" créé avec le 1er imshow
plt.colorbar(im, label='z[m]', cax=cax)
ax[0].set_title('Terrain')

# Palette de couleurs discrète
cm = plt.get_cmap('tab20', n)
# Palette de couleurs personnalisée
#cm = ListedColormap(["red", "lightblue", "gray", "#E0D010", "darkgreen"])
cl = ax[1].imshow(mat[f'kmeans_{n}'], origin='lower', cmap=cm, vmin=-0.5, vmax=n-0.5)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cl, label='classe', cax=cax)
ax[1].set_title(f'kmeans_{n}')
plt.show()