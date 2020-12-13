from matplotlib import pyplot as plt
import numpy as np
import math
import sympy as sym
from sympy import solvers as sol
from sympy import Symbol

#-----------------------------------------------------------------------------------------------------------------#


class P1:
    
    précision = 1000   # Nombre de valeurs que vont renvoyer les fonctions 
                       # (PAR SECONDE dans le cas des fonction temporelles)
    
    def __init__(self, lst, larg, H1, mdép, I, D):
        self.largeur = larg        #Largeur de la barge [m]
        self.longueur = larg       #Longueur de la barge (longueur=largeur) [m]
        self.H1 = H1               #Hauteur de la barge [m]
        self.mdéplacée = mdép      #Masse déplacée [kg]
        self.I = I                 #Moment d'inertie [gmm²]
        self.D = D                 #Coefficient d'amortissement
        self.lst = lst             #Liste reprenant les différentes masses (m1 et m2)
        
        
        # Initialisation des autres variables -----------------------------#
        self.mtot = mdép
        for i in range(len(lst)):
            self.mtot += self.lst[i]     
        
        self.Hc = self.mtot/(self.largeur*self.longueur*1000)
        
        self.dist_initial = 0.3           # le repère se trouve en (0,0,0) on suppose donc que le grappin prendra le morceau d'éolienne
                                          # en (0 ; 0,3 ; H1-Hc) [m]
        self.dist_max = 0.5               # Pour notre simulation, nous avons spécifié que la grue devait porter le morceau d'éolienne à 20cm de la barge
        self.G_initial = [0.001, self.H1 - self.Hc -0.005]    
        self.G_final = [0.06, self.H1 - self.Hc + 0.01]     # G_initial et G_final ont été obtenu sur Fusion360 pour plus de simplicité
                                                            # La variable contient d'ailleur la composante en y et ensuite la composante en z 
                                                            # et ce, car la composante en x est supposée nulle (hypothèse du modèle)
        
        self.theta_soulevement = math.atan(self.Hc / (self.longueur/2))
        self.theta_submersion = math.atan((self.H1 - self.Hc) / (self.longueur/2))
        
        print("Theta_soulevement : "+ str(self.theta_soulevement*180/math.pi) + " degrés")    
        print("Theta_submersion : " + str(self.theta_submersion*180/math.pi) + " degrés")     

    #--------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------- Fonctions usuelles --------------------------------------------#
        
    def G_dist(self, dist_d): # Cette fonction permet de calculer les coordonnées de G (en y et en z) en fonction de le distance "dist_d". 
                              # dist_d correspond à la distance à laquelle le morceau est déplacé, et est compris dans [dist_initial; dist_max]
                              # A dist_inital est associé G_initial, A dit_max est associé G_final, grâce à Fusion.
                              # Nous considérons une progression linéaire de G en fonction de dist_d.
        
        y_new_G = self.G_initial[0] + (self.G_final[0] - self.G_initial[0])*(dist_d - self.dist_initial)/(self.dist_max - self.dist_initial)
        z_new_G = self.G_initial[1] + (self.G_final[1] - self.G_initial[1])*(dist_d - self.dist_initial)/(self.dist_max - self.dist_initial)
        new_G = [y_new_G, z_new_G]
        return new_G
    
    
    def theta_1d(self, distance, masse = -1): # Cette fonction calcul thêta par itération. 
        """
        Input : la distance à laquelle on porte une charge d'une certaine masse (masse = self.mdéplacée par défaut)
        Output : l'angle theta à cette distance
        """
        
        if masse != -1 :  # Ce if/else permet de calculer thêta pour une masse différente de self.mdéplacée 
            mtot = self.mtot - self.mdéplacée + masse
            Ca = masse*9.81*distance
        else:
            mtot = self.mtot
            Ca = self.mdéplacée*9.81*distance
        
        Cr = 0
        interval = [-(math.pi)/2, (math.pi)/2]
        theta = (interval[0] + interval[1])/2
        
        G = self.G_dist(distance)
        
        while abs(Cr - Ca) > 0.00000001 :          # Tant qu'on a pas une différence inférieure à 0.00000001 la boucle continue de tourner
                                                   # Cela nous permet d'avoir une très bonne précision sans pour autant rechercher la valeur exacte
            
            deltaH = (2/self.largeur) * math.tan(theta)
            # Sous l'hypothèse que longueur = largeur :
            y_Cp = self.longueur/2 - ((self.longueur/3) * ((3*self.Hc -deltaH)/(2*self.Hc))) #déplacement du centre de poussée selon l'axe y.
            #    = self.longueur * deltaH / (6*self.Hc)
            y_G = G[0]*math.cos(theta) - G[1] *math.sin(theta)
            Cr = (y_Cp-y_G)*9.81*mtot
            if Cr > Ca :
                interval[1] = theta
            else:
                interval[0] = theta
            theta = (interval[0] + interval[1])/2

        return theta
    
    
    def theta_dist(self, end, masse = -1):
        """
        Input : la distance maximale à laquelle on porte une certaine masse (masse = self.mdéplacée par défaut)
                 
        Output : un array numpy contenant les valeurs de theta pour un déplacement continu de 0 à end, avec une 
                  précision de P1.précision
        """
        
        X = np.linspace(0, end, int(P1.précision))
        Y = np.zeros(len(X))
        for i in range(len(X)):
            Y[i] = self.theta_1d(X[i], masse)      # la fonction self.theta_1d() s'occupe de gérer masse = -1
        
        return Y


    def theta_stab(self, end, distance = 0, masse = -1): # Cette fonction calcule le thêta lors de la mise à l'eau du système barge+grue
        """
        Input : end = fin de la simulation (en sec); distance = distance fixe, ou liste des distances 
                de la masse transportée (int, float ou [end*P1.précision])
        Output : theta = une liste de l'évolution des thetas lors de la stabilisation ; omega = une liste de 
                 l'évolution de la vitesse angulaire lors de la stabilisation
        """
        
        if masse != -1 : # Ce if/else permet de calculer les thêtas pour une masse différente de mdéplacée
            mtot = self.mtot - self.mdéplacée + masse
            mdep = masse
        else:
            mtot = self.mtot
            mdep = self.mdéplacée 
        
        if type(distance) == int or type(distance) == float:
            distances = np.linspace(distance, distance, int(end*P1.précision))
        else :
            distances = distance
        
        # Début des calculs
        temps = np.linspace(0, end, int(end*P1.précision))
        dt = temps[1] - temps[0]
        
        theta = np.empty(int(end*P1.précision))
        omega = np.empty(int(end*P1.précision))
        theta[0] = 0
        omega[0] = 0
        
        F = -9.81*1000*self.largeur*self.longueur*self.Hc       # Formule de la force d'Archimède
        
        for i in range(len(temps)-1):          # Itérer et trouver les valeurs une par une en foction des précédentes !!
            # Données pour trouver y_C
            G = self.G_dist(distances[i])
            deltaH = (2/self.longueur) * math.tan(theta[i])
            y_G = G[0]*math.cos(theta[i]) - G[1]*math.sin(theta[i])
            y_Cp = self.longueur * deltaH / (6*self.Hc)
            #y_Cp = self.longueur/2 - (self.longueur/3) * (3*self.Hc-deltaH)/(2*self.Hc)  # déplacement du centre de poussée selon l'axe y
            Cg = 9.81*mtot*y_G                                                           # Couple dû à la masse "principale"
            Ca = 9.81*mdep*distances[i]                                                  # Couple dû à la masse déplacée
            
            y_C = F*y_Cp - Cg - Ca - self.D*omega[i]
            omega[i+1] = omega[i] + y_C*dt/self.I
            theta[i+1] = theta[i] + omega[i]*dt
            
        return theta, omega
    
    
    
    def theta_global_t(self, start_dép, dist_max, v, masse = -1, durée_rien = 20): # Calcul de l'angle de la barge en tenant compte de la stabilisation 
                                                                                   # Lors de la mise à l'eau ET d'un déplacement continu de la masse
        """
        Input : start = moment où le déplacement de la charge commence ; dist_max = distance à laquelle la masse
                est amenée ; v = vitesse de déplacement de la masse
                durée_rien = Durée avant 0 et après le déplacement (en sec) => max 4 chiffres après la virgule !!
        Output : T = liste des temps (abscisses) auxquels les angles ont été calculés ; theta_total = liste des
                 angles thetas ; theta_stab = liste des angles theta causés par la stabilisation ; 
                 theta_dép = liste des angles theta causés par le déplacement de la masse ;
                 distances = liste des distances de la masse
        """
        
        # Etapes : |   Rien   |   Stabilisation -> début du déplacement   |   Début déplacement -> Fin du déplacement   |   Rien
        n_rien = int(durée_rien*P1.précision)     # Nombre de valeurs pour les phases initiales et terminales (avant 0 et après dép)
        durée_dép = dist_max/v                    # Durée du déplacement
        n_dép = int(durée_dép*P1.précision)       # Nombre de valeurs pour t lors du déplacement
        n_interval = int(start_dép*P1.précision)  # Nombre de valeurs pour t entre la mise à l'eau (t = 0) et le début du déplacement (t = start_dép)
        
        T = list(np.linspace(-durée_rien,  start_dép + durée_dép + durée_rien,  2*n_rien + n_interval + n_dép))
        
        # Evolution de la position de la masse (pour la fonction theta_stab)
        distances = list(np.linspace(self.dist_initial, self.dist_initial, n_interval)) + list(np.linspace(self.dist_initial, dist_max, n_dép))   # Commence en t = 0
        p_sup = np.linspace(dist_max,dist_max, n_rien)
        distances = distances + list(p_sup)
        
        
        # Variations de theta dues à la mise à l'eau
        theta_stab = list(np.zeros(n_rien)) + list(self.theta_stab(start_dép + durée_dép + durée_rien, distances, masse)[0])
        
        # Variations totales
        theta_total = list(np.array(theta_stab))
        distances = list(np.zeros(n_rien)) + distances
        
        return T, theta_total, distances
    


#-----------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#
#-------------------------------------------- Fonctions de PLOT --------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#
    
    def plot_Theta_distance(self, end):
        fig1 = plt.gcf()
        fig1.canvas.set_window_title('Simulation informatique du modèle physique')
        plt.title("Evolution de theta en fonction du déplacement")
        X = np.linspace(0, end, P1.précision)
        Y = self.theta_dist(end)
        
        # Limites
        Ysou = np.zeros(len(X))   # thêta soulevement  
        Ysub = np.zeros(len(X))   # thêta submersion 
        Y2 = self.theta_dist(end, 0.02)
        Y3 = self.theta_dist(end, 2)
        for i in range(len(X)):
            Ysou[i] = self.theta_soulevement
            Ysub[i] = self.theta_submersion
        
        plt.grid(True)
        plt.title("Angle d'inclinaison en fonction de la distance à laquelle on porte la masse")
        plt.xlabel("Distance (m)")
        plt.ylabel("Angle d'inclnaison theta (rad)")
        plt.plot(X, Y2, label = "theta pour m = 0.02 kg", color="hotpink")
        plt.plot(X, Y, label = "theta pour m = " + str(self.mdéplacée) + " kg")
        plt.plot(X, Y3, label = "theta pour m = 2 kg", color="purple")
        plt.plot(X, Ysou, "--", color="green", label = "angle_de_soulèvement")
        plt.plot(X, Ysub, "--", color="red", label = "angle_de_submersion")
        #plt.plot(self.dist_max, self.theta_1d(self.dist_max), "o", color="black", label="Theta final")
        plt.legend()
        plt.show()


    def plot_Theta_stab(self, tps_final, position = False):
        fig2 = plt.gcf()
        fig2.canvas.set_window_title('Simulation informatique du modèle physique')
        plt.title("Evolution de theta en fonction du temps")
        T = np.linspace(0, tps_final, int(tps_final*P1.précision))
        
        if position == False :
            positions = self.dist_initial
        else : positions = position
        
        Y, W = self.theta_stab(tps_final, positions)  
        print("Theta stab = " + str(Y[-1]*180/math.pi) + " degrés")
        # Limites
        Y2 = np.zeros(len(T))
        Y3 = np.zeros(len(T))
        plt.xlabel("temps (sec)")
        plt.ylabel("Angle d'inclinaison theta (rad)")
        for i in range(len(T)):
            Y2[i] = self.theta_soulevement
            Y3[i] = self.theta_submersion
        
        plt.title("Valeur de theta en fonction du temps lors de la stabilisation")
        plt.xlabel("Temps (sec)")
        plt.ylabel("Angle d'inclinaison theta (rad)")
        plt.grid(True)
        plt.plot(T, Y, label="theta")
        plt.plot(T, Y2, "--", color="green", label="angle_de_soulèvement")
        plt.plot(T, Y3, "--", color="red", label="angle_de_submersion")
        plt.legend(loc = 1)
        plt.show()
        
            
            
    def plot_theta_global_t(self, start, dist, v, masses = [], rien = 20):
        fig3 = plt.gcf()
        fig3.canvas.set_window_title('Simulation informatique du modèle physique')
        plt.title("Evolution de theta en fonction du temps pour une charge de " + str(self.mdéplacée) + "kg")
        
        T, th, p = self.theta_global_t(start, dist, v, durée_rien = rien)
        Y2 = np.zeros(len(T))
        Y3 = np.zeros(len(T))
        for i in range(len(T)):
            Y2[i] = self.theta_soulevement
            Y3[i] = self.theta_submersion
        
        plt.xlabel("Temps (sec)")
        plt.ylabel("Angle d'inclinaison theta (rad)")
        plt.grid(True)
        print("Theta final = " + str(th[-1]) + " rad")
        print("Theta final = " + str(th[-1]*360/math.pi) + " degrés")
        
        if masses == [] :
            plt.title("Evolution de theta en fonction du temps pour une charge de " + str(self.mdéplacée) + "kg")
            plt.plot(T, th, label = "theta total", color="blue")
            #plt.plot(T, th1, label = "theta de mise à l'eau", color="orange")
            #plt.plot(T, th2, label = "theta de déplacement", color="m")
        else:
            plt.title("Evolution de theta en fonction du temps")
            for j in masses:
                th_bis = self.theta_global_t(start, dist, v, j)[0]
                plt.plot(T, th_bis, label = "angle d'inclinaison theta pour une charge de " + str(j) + " kg")
        
        
        plt.plot(T, Y2, "--", color="green", label="angle_de_soulèvement")
        plt.plot(T, Y3, "--", color="red", label="angle_de_submersion")
        
        plt.legend(loc=6, fontsize="small")
        plt.show()
            
            
    def plot_fct_sup(self, tps_final, Phase = False):
        fig3 = plt.gcf()
        fig3.canvas.set_window_title('Simulation informatique du modèle physique')
        if Phase:
            positions = np.zeros(int(tps_final*P1.précision))
            Y, W = self.theta_stab(tps_final, positions)
            plt.title("Diagramme de phase")
            plt.xlabel("angle d'inclinaison theta (rad)")
            plt.ylabel("Vitesse angulaire omega (rad/sec)")
            plt.grid(True)
            plt.plot(Y, W)
            plt.show()

        


# liste des poids et positions de centre de masse des éléments du projet [poids]
lst = [3.5, 0.772, 0.175] # => [barge, grue, flotteurs]


# P1(lst, largeur, Htotale, mdéplacée, Inertie, D) 
Syst = P1(lst, 0.6, 0.095, 0.2, 264.6, 20)

# end = distance max
#Syst.plot_Theta_distance(1) 

# tps_final
Syst.plot_Theta_stab(100)

# start (début du délacement), dist (distance max), v, masses = []
#  => /!\ start: pas plus de 4 chiffres après la virgule /!\
Syst.plot_theta_global_t(30, 0.2, 0.2, rien = 100)

Syst.plot_fct_sup(100, True)



