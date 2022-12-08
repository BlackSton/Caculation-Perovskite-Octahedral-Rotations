# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 02:35:33 2022

@author: rltjr
"""
from scipy import optimize
from scipy.stats import chisquare
import Calc,math
import numpy as np
import matplotlib.pyplot as plt

class Fit(Calc.tool):
    def __init__(self):
        
        #define initial Value
        self.I0 = 1
        self.wavelength = 1.54247
        self.a = 3.905
        self.b = 3.905
        self.c = 3.905
        self.direction = '---' # - : Out-of-Plane rotation. + : In-Plane rotation
        self.symbols = ['Sr2+','Ru4+','Ru4+','O2-'] # [A-Site,B-Site,B-Site,X-site] ABX_3
        #self.symbols = ['La3+','Ni','Ni','O2-'] # [A-Site,B-Site,B-Site,X-site] ABX_3
        self.M = 22.65
        
        # Defualt Setting Value
        self.I0_use     = False
        self.Oxygen     = True
        self.SLs        = False
        self.LP_mode    = 0
        self.eta_phi    = False
        
        self.ED  = [] #Experimental Data
        self.h   = []
        self.k   = []
        self.l   = []
        self.theta = []
        self.chi   = []
        self.phi   = []
        self.eta = []
        
        # alpha, beta, gamma, d1, d2, X,Y,X',Y'
        self.Init         =[0,0,0,
                            0,0,
                            0.25,0.25,0.25,0.25]
        self.Low_Boundary =[-15,-15,-15,
                            -0.05,-0.05,
                            0.,0.,0.,0.]
        self.High_Boundary=[15,15,15,
                            0.05,0.05,
                            1.,1.,1.,1.]
        self.cons = [{'type':'eq','fun':self.Cons_ab_equal},
                     {'type':'eq','fun':self.Cons_x_equal},
                     {'type':'eq','fun':self.Cons_y_equal},
                     {'type':'eq','fun':self.Cons_xy_equal}]
    # Vector Tool for measuring degree
    def VecNorm(self,v):
        if isinstance(v, np.ndarray):
            if len(v.shape) >= 2:
                return np.linalg.norm(v, axis=-1)
        if len(v) != 3:
            raise ValueError("Vector must be of length 3, but has length %d!"
                             % len(v))
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    def VecDot(self,v1, v2):
        if isinstance(v1, np.ndarray):
            if len(v1.shape) >= 2:
                return np.einsum('...i, ...i', v1, v2)
        if len(v1) != 3 or len(v2) != 3:
            raise ValueError("Vectors must be of size 3! (len(v1)=%d len(v2)=%d)"
                             % (len(v1), len(v2)))

        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    def VecAngle(self,v1, v2, deg=False):
        u1 = self.VecNorm(v1)
        u2 = self.VecNorm(v2)

        if isinstance(u1, np.ndarray) or isinstance(u2, np.ndarray):
            s = self.VecDot(v1, v2) / u1 / u2
            s[np.abs(s) > 1.0] = np.sign(s[np.abs(s) > 1.0]) * 1.0
            alpha = np.arccos(s)
            if deg:
                alpha = np.degrees(alpha)
        else:
            alpha = math.acos(max(min(1., self.VecDot(v1, v2) / u1 / u2), -1.))
            if deg:
                alpha = math.degrees(alpha)

        return alpha
    def Q2Ang(self,h,k,l):
        k0 = 2*np.pi/self.wavelength
        hkl = np.array([[h,k,l]])
        q = np.array([[2*np.pi/self.a,2*np.pi/self.b,2*np.pi/self.c]])
        q = q*hkl
        angle = np.zeros((4, q.shape[0]))
        # set parameters for the calculation
        z = np.array([0,0,1])
        y = np.array([0,1,0])
        x = np.array([1,0,0])
        q[0][0],q[0][1] = -q[0][1],q[0][0]
        qa = self.VecNorm(q)
        tth = 2. * np.arcsin(qa / 2. / k0)
        om = tth / 2.

        # calculation of the sample azimuth
        # the sign depends on the phi movement direction
        phi = -1 * np.arctan2(
            self.VecDot(q, x),
            self.VecDot(q, y)) - np.pi / 2.

        chi = (self.VecAngle(q, z))

        angle[0, :] = om
        angle[1, :] = chi
        angle[2, :] = phi
        angle[3, :] = tth
        return np.degrees(angle)
        
    def calc_eta(self):
        A = np.stack((self.h,self.k,self.l),axis=1)
        B = np.array([0,0,0])
        for T in A:
            om, chi, phi, tt = self.Q2Ang(*T)
            B = np.vstack((B,[om[0], chi[0], phi[0]]))
        self.theta = B[1:,0]
        self.chi   = B[1:,1]
        self.phi   = B[1:,2]
        if self.eta_phi == False:
            self.eta   = super().calc_eta(self.theta,self.chi)
        else:
            self.eta   = super().calc_eta(self.theta,self.chi,self.phi)
    def Background_Effect(self,a,b,c,X1, Y1, X2,Y2):
        fig = plt.figure(figsize=(20,4)) 
        ax1 = fig.add_subplot(131)
        ax1.plot(self.eta,'bo')
        ax1.set_title('Only eta')
        
        ax2 = fig.add_subplot(132)
        ax2.plot(self.LorenP(self.h, self.k, self.l),'co')
        ax2.set_title('Lorentz')
        
        ax3 =fig.add_subplot(133)
        ax3.plot(self.Only_F(self.h, self.k, self.l, a, b, c,X1, Y1, X2,Y2),'ro')
        ax3.set_title('Data')
        
    # only csv format was allowed
    def import_data(self,direction):
        data = np.loadtxt(direction,delimiter=',')
        if self.I0_use == False: #normalize Experimental Data 
            self.h  = data[1:,1]
            self.k  = data[1:,2]
            self.l  = data[1:,3]
            self.ED = data[1:,4]/data[1,4] 
        else:
            self.h  = data[0:,1] 
            self.k  = data[0:,2]
            self.l  = data[0:,3]
            self.ED = data[0:,4]
        
        self.calc_eta()

        
    def Cons_ab_equal(self,t):
        return t[0] - t[1] # alpha = beta
    def Cons_x_equal(self,t):
        return t[5] - t[6] # X = X'
    def Cons_y_equal(self,t):
        return t[7] - t[8] # Y = Y'
    def Cons_xy_equal(self,t):
        return t[5] - t[7] # X = Y
    
    
    def TD(self,params): # Calculate Teorical Data
        if self.I0_use == False:
            I0 = self.intensity_ambmcp(self.h[0],self.k[0],self.l[0],self.eta[0],*params)
            return self.intensity_ambmcp(self.h,self.k,self.l,self.eta,*params)/I0
        else:
            self.I0 = 1
            SF = self.intensity_ambmcp(self.h[0],self.k[0],self.l[0],self.eta[0],*params)
            self.I0 = self.ED[1] / SF
            return self.intensity_ambmcp(self.h,self.k,self.l,self.eta,*params)
            
            self.I0 = 1
            SF = self.intensity_ambmcp(self.h[0],self.k[0],self.l[0],self.eta[0],*params)
            self.I0 = self.ED[0] / SF
            return self.intensity_ambmcp(self.h,self.k,self.l,self.eta,*params)
        
        
    def SLs_TD(self,params): # Calculate Teorical Data
        if self.I0_use == False:
            I0 = self.intensity_ambmcp_SLs(self.h[0],self.k[0],self.l[0],self.eta[0],*params)
            return self.intensity_ambmcp_SLs(self.h,self.k,self.l,self.eta,*params)/I0
        else:
            self.I0 = 1
            SF = self.intensity_ambmcp_SLs(self.h[0],self.k[0],self.l[0],self.eta[0],*params)
            self.I0 = self.ED[0] / SF
            return self.intensity_ambmcp_SLs(self.h,self.k,self.l,self.eta,*params)
    
    def Soft_l1(self,z):
        return np.sum(2 * ((1 + z)**0.5 - 1))
    def Residual(self,params):
        return np.abs((self.ED-self.TD(params)))**2
    def chi_square(self,params):
        fobs = np.array(self.ED)
        fexp = np.array(self.TD(params))
        fexp = fexp * (np.sum(fobs)/np.sum(fexp)) 
        return chisquare(fobs,f_exp = fexp)[0]
    def SLs_chi_square(self,params):
        fobs = np.array(self.ED)
        fexp = np.array(self.SLs_TD(params))
        fexp = fexp * (np.sum(fobs)/np.sum(fexp)) 
        return chisquare(fobs,f_exp = fexp)[0]
    def Residual_Soft_l1(self,params):
        return self.Soft_l1(self.Residual(params))
    def Residual_Sum(self,params):
        return np.sum(self.Residual(params))
    def fit(self):
        self.res = optimize.least_squares(self.Residual,self.Init,
                                              bounds=(self.Low_Boundary,self.High_Boundary),
                                              loss='linear',method='trf')
    def fit_minimize(self):
        bound = optimize.Bounds(self.Low_Boundary,self.High_Boundary)
        self.res = optimize.minimize(fun=self.Residual_Soft_l1,x0=self.Init,method = 'SLSQP',bounds=bound,constraints = self.cons,options={"ftol":1e-8,"maxiter":100000})
    def fit_minimize_chi(self):
        bound = optimize.Bounds(self.Low_Boundary,self.High_Boundary)
        self.res = optimize.minimize(fun=self.chi_square,x0=self.Init,method = 'SLSQP',bounds=bound,constraints = self.cons,options={"ftol":1e-8,"maxiter":100000})
    def fit_minimize_SLs_chi(self):
        bound = optimize.Bounds(self.Low_Boundary,self.High_Boundary)
        self.res = optimize.minimize(fun=self.SLs_chi_square,x0=self.Init,method = 'SLSQP',bounds=bound,constraints = self.cons,options={"ftol":1e-8,"maxiter":100000})
    def fit_shgo(self):
        bound = optimize.Bounds(self.Low_Boundary,self.High_Boundary)
        self.res = optimize.shgo(self.Residual_Sum,[(0,15),(0,15),(0,15),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],constraints = self.cons,n=128,iters=4,sampling_method='sobol', options={'disp' : True})
        
        ###### Plotting Line ######
    def plot(self):
        if self.I0_use == False:
            if self.SLs == False:
                plt.plot(np.arange(self.h.shape[0]),self.TD(self.res.x),'bo')
            else:
                plt.plot(np.arange(self.h.shape[0]),self.SLs_TD(self.res.x),'bo')
            plt.plot(np.arange(self.h.shape[0]),self.ED,'rx')
        else:
            if self.SLs == False:
                print('T_I0',self.TD(self.res.x)[0])
            else:
                print('T_I0',self.SLs_TD(self.res.x)[0])
            print('E_I0',self.ED[0])
            if self.SLs == False:
                plt.plot(np.arange(self.h.shape[0])[1:],self.TD(self.res.x)[1:],'bo')
            else:
                plt.plot(np.arange(self.h.shape[0])[1:],self.SLs_TD(self.res.x)[1:],'bo')
            plt.plot(np.arange(self.h.shape[0])[1:],self.ED[1:],'rx')
    def fit_result(self):
        print(self.res.status,self.res.success)
        print('alpha :',self.Init[0])
        print('beta  :',self.Init[1])
        print('gamma :',self.Init[2])
        print('d1    :',self.Init[3])
        print('d2    :',self.Init[4])
        print('X1    :',100*self.Init[5])
        print('Y1    :',100*self.Init[6])
        print('X2    :',100*self.Init[7])
        print('Y2    :',100*self.Init[8])
        print('fun   :',self.res.fun)
        if self.SLs == True:
            print('beta2  :',self.Init[8])
            print('Ratio  :',100*self.Init[9])
        self.plot()
    
    
    
    
    
    
    
    
    