import periodictable.cromermann as ptc

import numpy as np

from collections import OrderedDict
from functools import reduce

class tool():
    
    def tand(self,theta):
        """ tan in degrees """
        return np.tan(np.radians(theta))
    def sind(self,theta):
        """ sin in degrees """
        return np.sin(np.radians(theta))
    
    def dHKL(self,H, K, L):
        """ Calculates dHKL for orthorhombic structure with the HKL and a,c values.
        """
        dHKL = np.sqrt(1./((H/self.a)**2.+(K/self.b)**2.+(L/self.c)**2.))
        return dHKL
    def calc_eta(self,theta,chi,phi = None):
        if self.eta_phi == False:
            return 1 / np.abs(np.sin(np.radians(theta))*np.sin(np.radians(chi)))
        else:
            chi = chi
            phi = phi % 90
            a = 0.1
            return np.abs(self.sind(a/2)/self.sind(theta+a/2)+self.sind(a/2)/self.sind(theta-a/2))*np.abs(1/(np.sin(np.radians(chi))))
            #return np.abs(1/(np.sin(np.radians(theta))*np.sin(np.radians(chi))/(1-0.5*(1-self.tand(phi/2))**2*self.tand(phi))))
            
        
    def LorenP(self,H, K, L):
        """  LP = 1/sin(2theta).  Theta is found from Bragg's law using dHKL, for orthorhombic. Can generalize later
        for monoclinic, but change will be extremely small considering typical deviation of angle is ~ 0.1 degrees.
        """
        if self.LP_mode == 0: #calculating the approximated Result
            LP =  1./np.sin(2.*np.arcsin(self.wavelength/(2.*self.dHKL(H,K,L))))
        elif self.LP_mode == 1: #calculating for Ge(220)-2 bounce monochromatic beam
            A = np.cos(2*self.M*np.pi/180)**2
            theta = np.arcsin(self.wavelength/(2.*self.dHKL(H,K,L)))
            LP = (1+A*np.cos(2*theta)**2)/((1+A)*np.sin(2*theta))
        elif self.LP_mode == 2: #calculating for Small Single crystal
            theta = np.arcsin(self.wavelength/(2.*self.dHKL(H,K,L)))
            LP = (1+np.cos(2*theta)**2)/(2*np.sin(2*theta))
        elif self.LP_mode == 3:
            theta = np.arcsin(self.wavelength/(2.*self.dHKL(H,K,L)))
            LP = (1+np.cos(2*theta)**2)/(np.cos(theta)*np.sin(theta)**2)
        return LP

   
    def CM(self,H, K, L, symbol):
        return np.array(ptc.fxrayatstol(symbol, 1/(2*self.dHKL(H,K,L))))
    
    def Rotation(self,d1,d2,alpha,beta,gamma,direction):
        
        if direction[0] == '-':
            A1 = [1,-1,-1,1,-1,1,1,-1]
            A2 = [-1,1,1,-1,1,-1,-1,1]
        else:
            A1 = [-1,-1,1,1,1,1,-1,-1]
            A2 = [1,1,-1,-1,-1,-1,1,1]
        if direction[1] == '-':
            B1 = [-1,1,1,-1,1,-1,-1,1]
            B2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            B1 = [1,-1,1,-1,-1,1,-1,1]
            B2 = [-1,1,-1,1,1,-1,1,-1]
        if direction[2] == '-':
            C1 = [-1,1,1,-1,1,-1,-1,1]
            C2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            C1 = [1,-1,-1,1,1,-1,-1,1]
            C2 = [-1,1,1,-1,-1,1,1,-1]
        
        cell_1RSL = OrderedDict(
        O1  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[0], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[0], 0.],
        O2  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[1], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[1], 0.],
        O3  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[2], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[2], 0.],
        O4  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[3], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[3], 0.],
        O5  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[4], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[4], 0.5],
        O6  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[5], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[5], 0.5],
        O7  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[6], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[6], 0.5],
        O8  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[7], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[7], 0.5],
        
        O9  = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[0], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[0]],
        O10 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[1], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[1]],
        O11 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[2], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[2]],
        O12 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[3], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[3]],
        O13 = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[4], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[4]],
        O14 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[5], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[5]],
        O15 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[6], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[6]],
        O16 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[7], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[7]],
        
        O17 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[0], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[0]],
        O18 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[1], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[1]],
        O19 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[2], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[2]],
        O20 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[3], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[3]],
        O21 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[4], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[4]],
        O22 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[5], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[5]],
        O23 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[6], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[6]],
        O24 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[7], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[7]],
        
        A1 = [d1, d1 + d2, 0],
        A2 = [0.5+d1+d2, d1, 0],
        A3 = [0.5+d1, 0.5+d1+d2, 0],
        A4 = [d1+d2, 0.5+d1, 0],
        A5 = [-d1, -d1-d2, 0.5],
        A6 = [0.5-d1-d2, -d1, 0.5],
        A7 = [0.5-d1, 0.5-d1-d2, 0.5],
        A8 = [-d1-d2, 0.5-d1, 0.5],
        
        B1 = [0.25, 0.25, 0.25],
        B2 = [0.75, 0.25, 0.25],
        B3 = [0.75, 0.75, 0.25],
        B4 = [0.25, 0.75, 0.25],
        
        C1 = [0.25, 0.25, 0.75],
        C2 = [0.75, 0.25, 0.75],
        C3 = [0.75, 0.75, 0.75],
        C4 = [0.25, 0.75, 0.75]
        )
        return cell_1RSL
    
    def Rotation_X1(self,d1,d2,alpha,beta,gamma,direction):
        
        if direction[0] == '-':
            A1 = [1,-1,-1,1,-1,1,1,-1]
            A2 = [-1,1,1,-1,1,-1,-1,1]
        else:
            A1 = [-1,-1,1,1,1,1,-1,-1]
            A2 = [1,1,-1,-1,-1,-1,1,1]
        if direction[1] == '-':
            B1 = [-1,1,1,-1,1,-1,-1,1]
            B2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            B1 = [1,-1,1,-1,-1,1,-1,1]
            B2 = [-1,1,-1,1,1,-1,1,-1]
        if direction[2] == '-':
            C1 = [-1,1,1,-1,1,-1,-1,1]
            C2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            C1 = [1,-1,-1,1,1,-1,-1,1]
            C2 = [-1,1,1,-1,-1,1,1,-1]
        
        cell_1RSL = OrderedDict(
        O1  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[0], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[0], 0.],
        O2  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[1], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[1], 0.],
        O3  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[2], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[2], 0.],
        O4  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[3], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[3], 0.],
        O5  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[4], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[4], 0.5],
        O6  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[5], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[5], 0.5],
        O7  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[6], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[6], 0.5],
        O8  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[7], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[7], 0.5],
        
        O9  = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[0], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[0]],
        O10 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[1], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[1]],
        O11 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[2], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[2]],
        O12 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[3], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[3]],
        O13 = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[4], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[4]],
        O14 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[5], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[5]],
        O15 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[6], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[6]],
        O16 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[7], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[7]],
        
        O17 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[0], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[0]],
        O18 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[1], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[1]],
        O19 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[2], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[2]],
        O20 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[3], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[3]],
        O21 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[4], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[4]],
        O22 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[5], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[5]],
        O23 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[6], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[6]],
        O24 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[7], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[7]],
        
        A1 = [d1, d1 + d2, 0],
        A2 = [0.5+d1+d2, d1, 0],
        A3 = [0.5+d1, 0.5+d1+d2, 0],
        A4 = [d1+d2, 0.5+d1, 0],
        A5 = [-d1, -d1-d2, 0.5],
        A6 = [0.5-d1-d2, -d1, 0.5],
        A7 = [0.5-d1, 0.5-d1-d2, 0.5],
        A8 = [-d1-d2, 0.5-d1, 0.5],
        
        B1 = [0.25, 0.25, 0.25],
        B2 = [0.75, 0.25, 0.25],
        B3 = [0.75, 0.75, 0.25],
        B4 = [0.25, 0.75, 0.25],
        
        C1 = [0.25, 0.25, 0.75],
        C2 = [0.75, 0.25, 0.75],
        C3 = [0.75, 0.75, 0.75],
        C4 = [0.25, 0.75, 0.75]
        )
        return cell_1RSL
        
    def Rotation_Y1(self,d1,d2,alpha,beta,gamma,direction):
        
        if direction[0] == '-':
            A1 = [1,-1,-1,1,-1,1,1,-1]
            A2 = [-1,1,1,-1,1,-1,-1,1]
        else:
            A1 = [-1,-1,1,1,1,1,-1,-1]
            A2 = [1,1,-1,-1,-1,-1,1,1]
        if direction[1] == '-':
            B1 = [-1,1,1,-1,1,-1,-1,1]
            B2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            B1 = [1,-1,1,-1,-1,1,-1,1]
            B2 = [-1,1,-1,1,1,-1,1,-1]
        if direction[2] == '-':
            C1 = [-1,1,1,-1,1,-1,-1,1]
            C2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            C1 = [1,-1,-1,1,1,-1,-1,1]
            C2 = [-1,1,1,-1,-1,1,1,-1]
        
        cell_1RSL = OrderedDict(
        O1  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[0], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[0], 0.],
        O2  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[1], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[1], 0.],
        O3  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[2], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[2], 0.],
        O4  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[3], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[3], 0.],
        O5  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[4], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[4], 0.5],
        O6  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[5], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[5], 0.5],
        O7  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[6], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[6], 0.5],
        O8  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[7], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[7], 0.5],
        
        O9  = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[0], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[0]],
        O10 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[1], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[1]],
        O11 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[2], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[2]],
        O12 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[3], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[3]],
        O13 = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[4], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[4]],
        O14 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[5], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[5]],
        O15 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[6], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[6]],
        O16 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[7], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[7]],
        
        O17 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[0], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[0]],
        O18 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[1], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[1]],
        O19 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[2], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[2]],
        O20 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[3], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[3]],
        O21 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[4], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[4]],
        O22 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[5], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[5]],
        O23 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[6], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[6]],
        O24 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[7], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[7]],
        
        A1 = [d1+d2, d1, 0],
        A2 = [0.5+d1, d1+d2, 0],
        A3 = [0.5+d1+d2, 0.5+d1, 0],
        A4 = [d1, 0.5+d1+d2, 0],
        A5 = [-d1-d2, -d1, 0.5],
        A6 = [0.5-d1, -d1-d2, 0.5],
        A7 = [0.5-d1-d2, 0.5-d1, 0.5],
        A8 = [-d1, 0.5-d1-d2, 0.5],
        
        B1 = [0.25, 0.25, 0.25],
        B2 = [0.75, 0.25, 0.25],
        B3 = [0.75, 0.75, 0.25],
        B4 = [0.25, 0.75, 0.25],
        
        C1 = [0.25, 0.25, 0.75],
        C2 = [0.75, 0.25, 0.75],
        C3 = [0.75, 0.75, 0.75],
        C4 = [0.25, 0.75, 0.75]
        )
        return cell_1RSL
        
    def Rotation_X2(self,d1,d2,alpha,beta,gamma,direction):
        
        if direction[0] == '-':
            A1 = [1,-1,-1,1,-1,1,1,-1]
            A2 = [-1,1,1,-1,1,-1,-1,1]
        else:
            A1 = [-1,-1,1,1,1,1,-1,-1]
            A2 = [1,1,-1,-1,-1,-1,1,1]
        if direction[1] == '-':
            B1 = [-1,1,1,-1,1,-1,-1,1]
            B2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            B1 = [1,-1,1,-1,-1,1,-1,1]
            B2 = [-1,1,-1,1,1,-1,1,-1]
        if direction[2] == '-':
            C1 = [-1,1,1,-1,1,-1,-1,1]
            C2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            C1 = [1,-1,-1,1,1,-1,-1,1]
            C2 = [-1,1,1,-1,-1,1,1,-1]
        
        cell_1RSL = OrderedDict(
        O1  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[0], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[0], 0.],
        O2  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[1], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[1], 0.],
        O3  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[2], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[2], 0.],
        O4  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[3], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[3], 0.],
        O5  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[4], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[4], 0.5],
        O6  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[5], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[5], 0.5],
        O7  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[6], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[6], 0.5],
        O8  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[7], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[7], 0.5],
        
        O9  = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[0], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[0]],
        O10 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[1], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[1]],
        O11 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[2], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[2]],
        O12 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[3], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[3]],
        O13 = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[4], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[4]],
        O14 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[5], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[5]],
        O15 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[6], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[6]],
        O16 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[7], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[7]],
        
        O17 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[0], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[0]],
        O18 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[1], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[1]],
        O19 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[2], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[2]],
        O20 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[3], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[3]],
        O21 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[4], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[4]],
        O22 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[5], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[5]],
        O23 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[6], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[6]],
        O24 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[7], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[7]],
        
        A1 = [-d1, d1 + d2, 0],
        A2 = [0.5-d1-d2, d1, 0],
        A3 = [0.5-d1, 0.5+d1+d2, 0],
        A4 = [-d1-d2, 0.5+d1, 0],
        A5 = [d1, -d1-d2, 0.5],
        A6 = [0.5+d1+d2, -d1, 0.5],
        A7 = [0.5+d1, 0.5-d1-d2, 0.5],
        A8 = [d1+d2, 0.5-d1, 0.5],
        
        B1 = [0.25, 0.25, 0.25],
        B2 = [0.75, 0.25, 0.25],
        B3 = [0.75, 0.75, 0.25],
        B4 = [0.25, 0.75, 0.25],
        
        C1 = [0.25, 0.25, 0.75],
        C2 = [0.75, 0.25, 0.75],
        C3 = [0.75, 0.75, 0.75],
        C4 = [0.25, 0.75, 0.75]
        )
        return cell_1RSL
        
    def Rotation_Y2(self,d1,d2,alpha,beta,gamma,direction):
        
        if direction[0] == '-':
            A1 = [1,-1,-1,1,-1,1,1,-1]
            A2 = [-1,1,1,-1,1,-1,-1,1]
        else:
            A1 = [-1,-1,1,1,1,1,-1,-1]
            A2 = [1,1,-1,-1,-1,-1,1,1]
        if direction[1] == '-':
            B1 = [-1,1,1,-1,1,-1,-1,1]
            B2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            B1 = [1,-1,1,-1,-1,1,-1,1]
            B2 = [-1,1,-1,1,1,-1,1,-1]
        if direction[2] == '-':
            C1 = [-1,1,1,-1,1,-1,-1,1]
            C2 = [1,-1,-1,1,-1,1,1,-1]
        else:
            C1 = [1,-1,-1,1,1,-1,-1,1]
            C2 = [-1,1,1,-1,-1,1,1,-1]
        
        cell_1RSL = OrderedDict(
        O1  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[0], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[0], 0.],
        O2  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[1], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[1], 0.],
        O3  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[2], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[2], 0.],
        O4  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[3], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[3], 0.],
        O5  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[4], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[4], 0.5],
        O6  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[5], 0.25 + self.c/(4*self.b) *  self.tand(alpha)*A1[5], 0.5],
        O7  = [0.25 + self.c/(4*self.a)* self.tand(beta)*B1[6], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[6], 0.5],
        O8  = [0.75 + self.c/(4*self.a)* self.tand(beta)*B1[7], 0.75 + self.c/(4*self.b) *  self.tand(alpha)*A1[7], 0.5],
        
        O9  = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[0], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[0]],
        O10 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[1], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[1]],
        O11 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[2], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[2]],
        O12 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[3], 0.25 + self.a/(4*self.c) * self.tand(beta)*B2[3]],
        O13 = [0  , 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[4], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[4]],
        O14 = [0.5, 0.25 + self.a/(4*self.b)*self.tand(gamma)*C1[5], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[5]],
        O15 = [0  , 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[6], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[6]],
        O16 = [0.5, 0.75 + self.a/(4*self.b)*self.tand(gamma)*C1[7], 0.75 + self.a/(4*self.c) * self.tand(beta)*B2[7]],
        
        O17 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[0], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[0]],
        O18 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[1], 0  , 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[1]],
        O19 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[2], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[2]],
        O20 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[3], 0.5, 0.25 + self.b/(4*self.c) * self.tand(alpha)*A2[3]],
        O21 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[4], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[4]],
        O22 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[5], 0  , 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[5]],
        O23 = [0.25 + self.b/(4*self.a)*self.tand(gamma)*C2[6], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[6]],
        O24 = [0.75 + self.b/(4*self.a)*self.tand(gamma)*C2[7], 0.5, 0.75 + self.b/(4*self.c) * self.tand(alpha)*A2[7]],
        
        A1 = [-d1-d2, d1, 0],
        A2 = [0.5-d1, d1+d2, 0],
        A3 = [0.5-d1-d2, 0.5+d1, 0],
        A4 = [-d1, 0.5+d1+d2, 0],
        A5 = [d1+d2, -d1, 0.5],
        A6 = [0.5+d1, -d1-d2, 0.5],
        A7 = [0.5+d1+d2, 0.5-d1, 0.5],
        A8 = [d1, 0.5-d1-d2, 0.5],
        
        B1 = [0.25, 0.25, 0.25],
        B2 = [0.75, 0.25, 0.25],
        B3 = [0.75, 0.75, 0.25],
        B4 = [0.25, 0.75, 0.25],
        
        C1 = [0.25, 0.25, 0.75],
        C2 = [0.75, 0.25, 0.75],
        C3 = [0.75, 0.75, 0.75],
        C4 = [0.25, 0.75, 0.75]
        )
        return cell_1RSL
    def calc_FSL(self,H, K, L, cell):
        """  For each element add the contributions to the structure factor for each site. Then total structure factor is the contribution times the elemental form factor. The 2* is for the doubled unit cell as May does in paper.
         """
        def expQr(H, K, L, r):
            return np.exp(2.*np.pi*1j* (H*r[0] + K*r[1] + L*r[2]) )
        A_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] == 'A'))
        B_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] == 'B'))
        C_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] == 'C'))
        O_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] == 'O'))
        
        F_HKL = self.CM(H, K, L, self.symbols[3]) * O_term + self.CM(H, K, L, self.symbols[1]) * B_term + self.CM(H, K, L, self.symbols[2]) * C_term + self.CM(H,K,L,self.symbols[0]) * A_term
        return F_HKL
    
    def calc_FSL_O(self,H, K, L, cell):
        """  For each element add the contributions to the structure factor for each site. Then total structure factor is the contribution times the elemental form factor. The 2* is for the doubled unit cell as May does in paper.
         """
        def expQr(H, K, L, r):
            return np.exp(2.*np.pi*1j* (H*r[0] + K*r[1] + L*r[2]) )
        O_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] == 'O'))
        
        F_HKL = self.CM(H, K, L, self.symbols[3]) * O_term
        return F_HKL
    def Only_F(self,H, K, L, alpha, beta, gamma,X1, Y1, X2,Y2):
        """ Takes all positive alpha beta gamma. In-plane"""
        d1 = 0
        d2 = 0
        direction_O = self.direction[1] + self.direction[0] + self.direction[2]
        Cells =  [self.Rotation_X2(d1, d2, alpha , beta  , gamma,self.direction),
                  self.Rotation_Y2(d1, d2, -beta , alpha , gamma,direction_O),
                  self.Rotation_Y1(d1, d2, -alpha, -beta , gamma,self.direction),
                  self.Rotation_X1(d1, d2, beta  , -alpha, gamma,direction_O),
                  ]
        I = np.zeros(H.shape, dtype='complex128')
        for cell, vol_frac in zip(Cells, [X1, Y1, X2, Y2]):
            if self.Oxygen == True:
                F_HKL = self.calc_FSL_O(H, K, L, cell)
            else:
                F_HKL = self.calc_FSL(H, K, L, cell)
            I += F_HKL * np.conj(F_HKL)*vol_frac
        return np.real(I)
    
    def intensity_ambmcp(self,H, K, L,eta, alpha, beta, gamma, d1, d2,X1, Y1, X2,Y2):
        """ Takes all positive alpha beta gamma. In-plane"""
        direction_O = self.direction[1] + self.direction[0] + self.direction[2]
        Cells =  [self.Rotation_X2(d1, d2, alpha , beta  , gamma,self.direction),
                  self.Rotation_Y2(d1, d2, -beta , alpha , gamma,direction_O),
                  self.Rotation_Y1(d1, d2, -alpha, -beta , gamma,self.direction),
                  self.Rotation_X1(d1, d2, beta  , -alpha, gamma,direction_O),
                  ]
        """
        Cells =  [self.Rotation_X2(d1, d2, alpha , beta  , gamma,"+--"),
                  self.Rotation_Y2(d1, d2, beta , alpha , gamma,"-+-"),
                  self.Rotation_Y1(d1, d2, -alpha, beta , gamma,"+--"),
                  self.Rotation_X1(d1, d2, beta  , -alpha, gamma,"-+-"),
                  ]
        """
        I = np.zeros(H.shape, dtype='complex128')
        for cell, vol_frac in zip(Cells, [X1, Y1, X2, Y2]):
            if self.Oxygen == True:
                F_HKL = self.calc_FSL_O(H, K, L, cell)
            else:
                F_HKL = self.calc_FSL(H, K, L, cell)
            L_P = self.LorenP(H, K, L) + 1e-16*1j
            I += self.I0*eta*L_P* F_HKL * np.conj(F_HKL)*vol_frac
        return np.real(I)
    
    def intensity_ambmcp_test(self,H, K, L,eta, alpha, beta, gamma, d1, d2,X1, Y1, X2,Y2):
        """ Takes all positive alpha beta gamma. In-plane"""
        direction_O = self.direction[1] + self.direction[0] + self.direction[2]
        
        Cells =  [
                    self.Rotation_X1(d1, d2, -alpha , beta  , gamma,'-+-'),
                    self.Rotation_Y1(d1, d2, beta , alpha  , gamma,'+--'),
                    self.Rotation_X2(d1, d2, gamma , beta  , alpha,'-+-'),
                    self.Rotation_Y2(d1, d2, beta , -gamma  , alpha,'+--'),
                  
                  
                  ]

        I = np.zeros(H.shape, dtype='complex128')
        for cell, vol_frac in zip(Cells, [X1, Y1, X2, Y2]):
            if self.Oxygen == True:
                F_HKL = self.calc_FSL_O(H, K, L, cell)
            else:
                F_HKL = self.calc_FSL(H, K, L, cell)
            L_P = self.LorenP(H, K, L) + 1e-16*1j
            I += self.I0*eta*L_P* F_HKL * np.conj(F_HKL)*vol_frac
        return np.real(I)
    
    def intensity_ambmcp_SLs(self,H, K, L,eta, alpha, beta, gamma, d1, d2,X1,Y1,X2,beta2,R):
        """ Takes all positive alpha beta gamma. In-plane"""
        direction_O = self.direction[1] + self.direction[0] + self.direction[2]
        Y2 = 1 - X1 - Y1 - X2
        Cells =  [
                  #SrRuO3 term <- Orthorhombic structure
                  self.Rotation_X2(d1, d2, alpha, beta, gamma,self.direction),
                  self.Rotation_Y2(d1, d2, beta, alpha, gamma,direction_O),
                  self.Rotation_X1(d1, d2, -alpha, beta, gamma,self.direction),
                  self.Rotation_Y1(d1, d2, beta, -alpha, gamma,direction_O),
            
                  #CaRuO3 term <- Tetragonal   structure
                  self.Rotation_X1(0, 0, 0, beta2, 0,"---"),
                  self.Rotation_Y2(0, 0, -beta2, 0, 0,"---"),
                  self.Rotation_Y1(0, 0, 0, -beta2, 0,"---"),
                  self.Rotation_X2(0, 0, beta2, 0, 0,"---"),
                  ]
        I = np.zeros(H.shape, dtype='complex128')
        for cell, vol_frac in zip(Cells, [X1, Y1, X2, Y2,0.45*R,0.05*R,0.45*R,0.05*R]):
            if self.Oxygen == True:
                F_HKL = self.calc_FSL_O(H, K, L, cell)
            else:
                F_HKL = self.calc_FSL(H, K, L, cell)
            L_P = self.LorenP(H, K, L) + 1e-16*1j
            I += self.I0*eta*L_P* F_HKL * np.conj(F_HKL)*vol_frac
        return np.real(I)
    
    
    
    
    
    
    
    
    
    
    
    
    