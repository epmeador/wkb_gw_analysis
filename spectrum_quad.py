import numpy as np
import pygsl.odeiv as odeiv
import pygsl.spline as spline
from pygsl.testing import _ufuncs
import time
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
#from scipy.optimize import minimize_scalar
from scipy.special import airye
from scipy.interpolate import UnivariateSpline



#This will upload specifically the file that we need for spectrum to run: H, eps, etc. and the other slow roll variables.
from calcpath import *

knos = 1575 # total number of k-values to evaluate
kinos = 214 # total number of k-values to use for integration
k_file = "ks_eval.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks.dat" # file containing k-values for integration
k_file_tf = "k_file_tf.dat" #transfer function for transfer function
f_file_tf = "f_file_tf.dat" #frequency from transfer function
Tf_median = "Tf_median.dat" #median of transfer function
Y = 50 # Y = value of k/aH at which to initialize mode fcns
knorm = 0.05 # normalization scale
Amp = 2.0803249e-9 # scalar amplitude at knorm

VERYSMALLNUM = 1.E-18

class params:
    def __init__(self):
        self.a_init = None # initial val of the scale factor
        self.k = None # comoving wavenumber
        self.eps = None
        self.sig = None
        self.H = None
        self.xi = None

#After defining a class of necessary parameters we go ahead with the spectrum function. One can adjust several things that I can note

def spectrum(y_final, y, u_s, u_t, N, derivs1, scalarsys, tensorsys):
    i = None

    h = 0.01
    h2 = 1.e-6 # init step size for mode integration

    abserr1 = 1.e-08 # absolute error tolerance - DO NOT ADJUST THESE VALUES!
    relerr1 = 1.e-08 # relative error tolerance

    abserr2 = 1.e-9 # absolute error tolerance #modified
    relerr2 = 1.e-9 # relative error tolerance

    spec_params = params()

    # Read in k files
    k = None
    ks = np.empty(knos)
    kis = np.empty(kinos)

    try:
        ks = np.loadtxt(k_file)
    except IOError as e:
        print("Could not open file " + k_file + ", errno = " + e + ".")
        sys.exit()
        
    try:
        kis = np.loadtxt(ki_file)
    except IOError as e:
        print("Could not open file " + ki_file + ", errno = " + e + ".")
        sys.exit()

    realu_init = np.empty(2)
    imu_init = np.empty(2)

    realu_s = np.empty(kmax)
    realu_t = np.empty(kmax)

    imu_s = np.empty(kmax)
    imu_t = np.empty(kmax)

    P_s = np.empty(kinos)
    P_t = np.empty(kinos)

    j = None
    l = None
    m = None
    o = None
    status = None

    countback = 0
    count = 0

    ydoub = np.empty(NEQS)

    Ninit = None # N_obs from flow integration
    Nfinal = None # Smallest N value from flow integration

    spec_norm = None

    ru_init = None
    dru_init = None

    iu_init = None
    diu_init = None

    nu = None
    Yeff = None
    Phi = None

    # Buffers for interpolations
    Nefoldsback = np.empty(kmax)
    
    flowback = np.empty((5,kmax))

    Nordered = np.empty(kmax)
    uordered_s = np.empty(kmax)
    uordered_t = np.empty(kmax)

    """
    Initialize/allocate gsl stepper routines and variable
    step-size routines.  Define ode system.
    """
    s = odeiv.step_rk4(NEQS, derivs1)
    c = odeiv.control_y_new(s, abserr1, relerr1)
    e = odeiv.evolve(s, c, NEQS)

    """
    Set the initial value of the scale factor.  This is chosen
    so that k = aH (with k corresponding to the quadrupole) at the
    value N_obs from the path file.  The scale factor as a 
    function of N is a(N) = a_init*exp(-# of efolds).
    Units are hM_PL
    """
    Ninit = N
    spec_params.a_init = (1.73e-61/y[1]) * np.exp(Ninit)
    spec_params.k = k

    """
    To improve stability/efficiency, we first generate
    an interpolating function for H, epsilon, sigma and xi^2.  We then pass these values
    as parameters to the mode equation, rather than solving the mode equation along with
    the full set of flow equations each time.
    """

    """
    Integrate backwards from end of inflation to the earliest time needed in order to initialize the
    largest scale fluctuations in the BD limlt.
    """
    ydoub[:] = y_final[:NEQS].copy()
    N = y_final[NEQS]
    Nfinal = N

    while (kis[0]*5.41e-58) / (spec_params.a_init*np.exp(-N)*ydoub[1]) < Y:
        flowback[:, countback] = ydoub[:5].copy()

        Nefoldsback[countback] = N

        try:
            N, h2, ydoub = e.apply(N, 1000, h2, ydoub)
        except:
            status = 0
            return status
        else:
            status = 0

        countback += 1

    Nefoldsback[countback] = N

    flowback[:, countback] = ydoub[:5].copy()

    H = np.empty(countback+1)
    eps = np.empty(countback+1)
    sig = np.empty(countback+1)
    xi = np.empty(countback+1)
    Nefolds = np.empty(kmax)
    # Nefolds = np.empty(countback+1)
    phi = np.empty(countback+1)

    H[:] = flowback[1, :countback+1].copy()
    eps[:] = flowback[2, :countback+1].copy()
    sig[:] = flowback[3, :countback+1].copy()
    xi[:] = flowback[4, :countback+1].copy()
    phi[:] = flowback[0, :countback+1].copy()
    Nefolds[:countback+1] = Nefoldsback[:countback+1].copy()

    # Generate interpolating functions for H, eps, sig, xi and phi (for path gen. only)
    spline1 = spline.cspline(countback+1)
    spline1.init(Nefolds[:countback+1], H)

    spline2 = spline.cspline(countback+1)
    spline2.init(Nefolds[:countback+1], eps)

    spline3 = spline.cspline(countback+1)
    spline3.init(Nefolds[:countback+1], sig)

    spline4 = spline.cspline(countback+1)
    spline4.init(Nefolds[:countback+1], xi)

    spline0 = spline.cspline(countback+1)
    spline0.init(Nefolds[:countback+1], phi)
    
    h2 = -h2
    
    #The story begins with finding the scalar power spectrum.
 
    """
    Find scalar spectra first.
    """
    
    total_time = 0.0
    for m in range(kinos):
        print("Starting for this mode")
        start_time = time.time()
        start_scalars = time.time()


        print(m)

        k = kis[m] * 5.41e-58 # converts to Planck from hMpc^-1
        kis[m] = k
        print("k:", k)
        N = Ninit
        ydoub[1] = spline1.eval(N)
        ydoub[2] = spline2.eval(N)
        count = 0

        """
        First, check to see if the given k value is in the
        Bunch-Davies limit at the start of inflation.  This limit is
        set by the #define Y=k/aH.  If the given k value yields a
        larger Y than the BD limit, then we must integrate forward
        (to smaller N) until we reach the proper value for Y.  If it is
        smaller, we must integrate backwards (to larger N).  These
        integrators are given a fixed stepsize to ensure that we don't
        inadvertently step too far beyond Y.
        """
        
        #This will find the N and will step back or forward until the Y=k/aH condition is met
        if k/1.73e-61 > Y: # 1.73e-61 is the present Hubble radius (~3.2e-4 hMpc^-1) in Planck units
            while k / (spec_params.a_init*np.exp(-N)*ydoub[1]*(1-ydoub[2])) > Y:
                N += -0.01
                ydoub[1] = spline1.eval(N)
                ydoub[2] = spline2.eval(N)
        else:
            while k / (spec_params.a_init*np.exp(-N)*ydoub[1]*(1-ydoub[2])) < Y:
                N += 0.01
                ydoub[1] = spline1.eval(N)
                ydoub[2] = spline2.eval(N)
                
        N_start = N


        spec_params.k = k
        nu = (3-spline2.eval(N)) / (2*(1-spline2.eval(N)))
        
        #It then sets Yeff ~ Y cause it take that N that was found at Y and sets that as the starting point
        Yeff = k / (spec_params.a_init*(np.exp(-N)*(spline1.eval(N)*(1.-spline2.eval(N)))))
        
        
        def Y_of_N(N_try):
            return k / (spec_params.a_init * np.exp(-N_try) * spline1.eval(N_try) * (1.-spline2.eval(N_try)))

        N_guess = N_start
        N_final = 0
        while Y_of_N(N_guess) > 1.0 and N_guess > N_final:
            N_guess -= 0.1
        N_exit = N_guess
        print('N_exit',N_exit)
        
        print('N_start',N_start)
        print("Y_start =", Y_of_N(N_start))
        print('N_exit',N_exit)
        print("Y_exit  =", Y_of_N(N_exit))

        #According to the original code, the initial conditions are set by the value of epsilon they differ slightly
        if spline2.eval(N) < 1.:
            ru_init = realu_init[0] = 0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * _ufuncs.sf_bessel_Jnu(nu, Yeff)
            dru_init = realu_init[1] = 0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1.eval(N))) * (_ufuncs.sf_bessel_Jnu(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-_ufuncs.sf_bessel_Jnu(nu+1., Yeff)+(nu*(1.-spline2.eval(N))*_ufuncs.sf_bessel_Jnu(nu, Yeff))/(Yeff*(1.-spline2.eval(N))))))
            iu_init = imu_init[0] = 0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * _ufuncs.sf_bessel_Ynu(nu, Yeff)
            diu_init = imu_init[1] = 0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1.eval(N))) * (_ufuncs.sf_bessel_Ynu(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-_ufuncs.sf_bessel_Ynu(nu+1., Yeff)+(nu*(1.-spline2.eval(N))*_ufuncs.sf_bessel_Ynu(nu, Yeff))/(Yeff*(1.-spline2.eval(N))))))
        else:
            ru_init = realu_init[0] = -0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * _ufuncs.sf_bessel_Ynu(nu, Yeff)
            dru_init = realu_init[1] = -0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1.eval(N))) * (_ufuncs.sf_bessel_Ynu(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-_ufuncs.sf_bessel_Ynu(nu+1., Yeff)+(nu*(1.-spline2.eval(N))*_ufuncs.sf_bessel_Ynu(nu, Yeff))/(Yeff*(1.-spline2.eval(N))))))
            iu_init = imu_init[0] = 0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * _ufuncs.sf_bessel_Jnu(nu, Yeff)
            diu_init = imu_init[1] = 0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1.eval(N))) * (_ufuncs.sf_bessel_Jnu(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-_ufuncs.sf_bessel_Jnu(nu+1., Yeff)+(nu*(1.-spline2.eval(N))*_ufuncs.sf_bessel_Jnu(nu, Yeff))/(Yeff*(1.-spline2.eval(N))))))
        
        # complex BD mode at N_start, N_start is deep in horizon we expect them to match at that point
        u_init_BD = ru_init + 1j * iu_init
        

        """
        Solve for real part of u first.
        """
        s2 = odeiv.step_rkf45(2, scalarsys, args=spec_params)
        c2 = odeiv.control_y_new(s2, abserr2, relerr2)

        #If we want to see the mode to just a little after horizon crossing we can change the bounds such that we extend to N_exit and not N_final. Assume N_final extends down to N=0 (at zero is the end of inflation).
        while N > Nfinal:
#        while N > N_exit:

            realu_s[count] = realu_init[0] * realu_init[0]
            Nefolds[count] = N

            spec_params.H = spline1.eval(N)
            spec_params.eps = spline2.eval(N)
            spec_params.sig = spline3.eval(N)
            spec_params.xi = spline4.eval(N)
            Phi = spline0.eval(N)

            e2 = odeiv.evolve(s2, c2, 2) # mode eqs
            
            try:
                N, h2, realu_init = e2.apply(N, 0, h2, realu_init)
            except:
                status = 0
                return status
            else:
                status = 0

            count += 1
            
            if count == kmax:
                status = 0
                return status

        realu_s[count] = realu_init[0] * realu_init[0]
        Nefolds[count] = N

        for j in range(count+1):
            Nordered[j] = Nefolds[count-j]
            uordered_s[j] = realu_s[count-j]

        """
        Generate interpolating function for realu(N)
        """
        spline5 = spline.cspline(count+1)
        spline5.init(Nordered[:count+1], uordered_s[:count+1])

        """
        Imaginary part
        """
        count = 0
        N = Nefolds[0]

        s2 = odeiv.step_rkf45(2, scalarsys, args=spec_params)
        c2 = odeiv.control_y_new(s2, abserr2, relerr2)
        e2 = odeiv.evolve(s2, c2, 2) # mode eqs

        while N > Nfinal:
#        while N > N_exit:

            imu_s[count] = imu_init[0] * imu_init[0]
            Nefolds[count] = N

            spec_params.H = spline1.eval(N)
            spec_params.eps = spline2.eval(N)
            spec_params.sig = spline3.eval(N)
            spec_params.xi = spline4.eval(N)
            
            try:
                N, h2, imu_init = e2.apply(N, 0, h2, imu_init)
            except:
                status = 0
                return status
            else:
                status = 0

            count += 1

            if count == kmax:
                status = 0
                return status

        imu_s[count] = imu_init[0] * imu_init[0]
        Nefolds[count] = N
        count -= 1

        P_s[m] = (k**3./(2.*(np.pi**2.))) * (spline5.eval(Nefolds[count])+imu_s[count]) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])*spline2.eval(Nefolds[count]))/(4*np.pi))
        end_scalars = time.time()
        
        print(f"[DEBUG] scalar mode {m} took {end_scalars - start_scalars:.3f} s")
        
    

        #We continue the story with the tensor power spectrum, which is not based on the numerical code anymore. Instead we use a WKB approximation to find this.

        """
        Tensor spectra
        """
        start_tensors = time.time()
        print('Stats from actual mode evaluation part:')


        #Define Ngrid that is the same as the original code, this is something that will be used to evaluate our power spectrum.
        Ngrid_ode = Nefolds[:count+1]
        
        #WKB adjusted
        ut_N_norm_complex, Ngrid, spline_ut_N_norm_complex, N_star  = tensorsys(N_start, N_exit, Nfinal, Ngrid_ode, spec_params, spline1, spline2, k)

        print(f"[CHECK] ODE-based N_exit = {N_exit}")
        print(f"[CHECK] WKB-based N_star = {N_star}")
        

        #I can evaluate variables over the numerical N using a spline created in tensorsys for ut
        u_t_ode_length = spline_ut_N_norm_complex(Ngrid_ode)
        u_t_real = np.real(u_t_ode_length) #This is our real part
        u_t_imag = np.imag(u_t_ode_length) #This is our imaginary part

        #We proceed to make a spline for Re and Im
        spline_real = CubicSpline(Nordered[:count+1], u_t_real[::-1])
        spline_imag = CubicSpline(Nordered[:count+1], u_t_imag[::-1])
   
        #As diagnostics we can also take a look at what u looks like over the full range:
        u_t_all_sq =  u_t_real**2 +  u_t_imag**2 #u^2
        amp_u_t_all = np.sqrt(u_t_all_sq)
        u_t_all_real = u_t_real #Re(u)
        u_t_all_imag = u_t_imag #Im(u)
        a_ode = spec_params.a_init * np.exp(-Ngrid_ode)


        #Take a look at phi = u/a
        phi_wkb_test = amp_u_t_all / a_ode
        plt.figure(figsize=(6,4))
        plt.plot(Ngrid_ode, phi_wkb_test, label=r"$\phi = |u|/a$ (WKB)", color="purple")
        plt.axvline(N_star, color="gray", linestyle="--", label="Horizon exit")
        plt.title(r"Freezeout check: $\phi(N) = |u(N)|/a(N)$")
        plt.xlabel(r"N")
        plt.ylabel(r"$\phi$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        print("phi near end of inflation:", phi_wkb_test[-10:])
        
        
        #Plot u^2/a^2 since this is what is fed into P_t
        plt.figure(figsize=(5,3))
        plt.plot(Ngrid_ode, u_t_all_sq/np.abs(a_ode**2), label=r"$\phi^2 =|u_t^2|/a^2$ ", color='r')
        plt.axvline(N_exit, color='gray', linestyle='--', label='N_exit')
        plt.xlabel(r"N")
        plt.ylabel(r"$\phi = u_t(N)^2/a^2$")
        plt.title(f"Examining Mode Behavior: Mode k = {k:.2e}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        
        #To define specifically the point that we evaluate the tensor power spectrum, we can choose the N value we are interested in measuring this at for every mode.
        N_freeze = Nfinal + 0.65
        u_freeze_wkb_squared =  (spline_real(N_freeze))**2 + (spline_imag(N_freeze))**2 #u^2
        a_freeze = spec_params.a_init * np.exp(-N_freeze)
        print("N_freeze from WKB",N_freeze)
        
        #We define the tensor power spectrum as the following:
        P_t[m] = 64*np.pi * (k**3/(2*np.pi**2)) * ((u_freeze_wkb_squared) / (a_freeze**2))
        
        
        end_tensors= time.time()
        print(f"[DEBUG] tensor mode {m} took {end_tensors - start_tensors:.3f} s")

       
        #For saving and comparing purposes
        u_wkb = amp_u_t_all #gets |u|
        u_wkb_real = u_t_all_real.copy()
        u_wkb_imag = u_t_all_imag.copy()
        u_wkb_sq = u_t_all_sq.copy()
        u_wkb_sq_over_a2 = u_wkb_sq / (a_ode**2)

        
        #I can mark horizon crossing like so:
        #get a*H
        H_all = np.array([spline1.eval(Ni) for Ni in Ngrid_ode])
        aH_length = a_ode * H_all
        #Find index where k-aH=tiny
        hc_idx = np.argmin(np.abs(k - aH_length))
        N_hc = Ngrid_ode[hc_idx]
        
        N_exit_val = N_hc

        outfile = f"wkb_diag_mode_{m:04d}.dat"

        #Below I changed N_exit_val to N_exit see if that is more accurate since that is what I used in WKB approx loosely
        np.savetxt(
            outfile,
            np.column_stack([
                Ngrid_ode,
                u_wkb,
                u_wkb_real,
                u_wkb_imag,
                u_wkb_sq,
                u_wkb_sq_over_a2,
                np.ones_like(Ngrid_ode) * N_exit,
                a_ode
            ]),
            header="N   |u_wkb(N)|  u_wkb_real(N)  u_wkb_imag(N)   u_wkb^2(N)   |u_wkb|^2/a^2    N_exit  a(N)"
        )
        
        
    
        print(f"[WKB] saved diagnostics → {outfile}")
    

        #Counts time taken
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        print(f"Mode {m+1}/{kinos} (k = {kis[m]:.3e}) took {elapsed:.2f} s")
        print("Finished for this mode")
        print("")



        if kis[m] == knorm * 5.41e-58: # normalize here
            spec_norm = Amp / (P_s[m]+P_t[m])
            """
            This is a little different from the C code,
            because the y[1] change is outside the if statement
            """
            y[1] = np.sqrt(spec_norm) # normalize H for later recon
    

    """
    Now that we have finished calculating the spectra, interpolate each spectrum and evaluate at k-values of interest
    """
    spline8 = spline.cspline(kinos)
    spline8.init(kis, P_t)

    spline6 = spline.cspline(kinos)
    spline6.init(kis, P_s)
    
    
    plt.figure()
    plt.loglog(kis, P_s, 'o-')
    plt.xlabel("log k")
    plt.ylabel("log P_s (normalized raw)")
    plt.show()
    
    plt.figure()
    plt.loglog(kis, P_t, 'o-')
    plt.xlabel("log k")
    plt.ylabel("log P_t (normalized raw)")
    plt.show()
     
     
    #If we wanted to discuss more about applying a transfer function to this:
    
#    tf_spline = spline.cspline(len(tf_k)) #length of k
#    tf_spline.init(tf_k, tf_median) #T(k)
#    #Tf_our_k = tf_spline.eval(ks)
##    Tf_our_k = np.array([tf_spline.eval(k) for (k/ 5.41e-58) in kis])
#    Tf_our_k = np.array([tf_spline.eval(k / 5.41e-58) for k in kis])
#
#    print('Shape of Tf our k', Tf_our_k.shape)
#    print(len(kis), len(P_s))
#
#    plt.figure(figsize=(6,4))
#    plt.plot(kis,Tf_our_k*P_s, label='T(k)*P_s')
#    plt.xlabel('k comoving wavenumber')
#    plt.ylabel('T(k)*P_s')
#    plt.title('Test Plot T_Median(k)*P_s(k)')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.legend()
#    plt.show()
    
    

    for i in range(knos):
        u_s[0, i] = ks[i]
        u_s[1, i] = spec_norm * spline6.eval(ks[i]*5.41e-58)

        u_t[0, i] = ks[i]
        u_t[1, i] = spec_norm * spline8.eval(ks[i]*5.41e-58)

    return status

def derivs1(t, y, dydN):
    dydN = np.zeros(NEQS, dtype=float, order='C')
    
    if y[2] > VERYSMALLNUM:
        dydN[0]= - np.sqrt(y[2]/(4*np.pi))
    else:
        dydN[0] = 0.

    dydN[1] = y[1] * y[2]
    dydN[2] = y[2] * (y[3]+2.*y[2])
    dydN[3] = 2.*y[4] - 5.*y[2]*y[3] - 12.*y[2]*y[2]
    
    for i in range(4, NEQS-1):
         dydN[i] = (0.5*(i-3)*y[3]+(i-4)*y[2])*y[i] + y[i+1]

    dydN[NEQS-1] = (0.5*(NEQS-4)*y[3]+(NEQS-5)*y[2]) * y[NEQS-1]

    return dydN
    

def scalarsys(t, y, parameters):
    dydN = np.empty(2)

    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-2.*(1.-2.*(p.eps)-0.75*(p.sig) - (p.eps)*(p.eps) + 0.125*(p.sig)*(p.sig) + 0.5*(p.xi)))*y[0]

    return dydN
 
 
def tensorsys(N_start, N_exit, Nfinal, Ngrid_ode, spec_params, spline1, spline2, k):

    print('Stats from tensorsys:')


#We define the range of e-folds over which to evaluate the mode:
    Ngrid = np.linspace(N_start, Nfinal, 4500) #This one comes directly from te Nstart and Nfinal value and offers better resolution for WKB
#    Ngrid = Ngrid_ode.copy() #This one is identical to that which is used for the original numerical
    print("Last 20 Ngrid values:", Ngrid[-20:])


#First I want to make sure I am accurately finding N_star (basically our turning point), and I will do this using e-folds as my variable not tau. Tau in this computer language makes everything very complicated. So our variables that get fed into the WKB expression are already listed in terms of N.

    a_of_Ngrid = spec_params.a_init * np.exp(-Ngrid) #uses a_init
    H_of_Ngrid = np.array([spline1.eval(float(N)) for N in Ngrid])
    eps_of_Ngrid = np.array([spline2.eval(float(N)) for N in Ngrid])
    
    Q_vals_N = (a_of_Ngrid**2) * (H_of_Ngrid**2) * (2 - eps_of_Ngrid) - k**2

    #A parameter I will care about later as a checking point. When k/aH = 1 this is at horizon crossing
    k_over_aH = k / (a_of_Ngrid * H_of_Ngrid)
    k_over_aH_with_eps = k / (a_of_Ngrid * H_of_Ngrid * (np.sqrt(2-eps_of_Ngrid)))


#If I use N variables:

    a_of_N = lambda N: spec_params.a_init * np.exp(-N)
    H_of_N = lambda N: spline1.eval(N)
    eps_of_N = lambda N: spline2.eval(N)

##Method to find N_star
    #define Q of N first, cause we need to be able to get N_star so I can define w
    def Q_of_N(N_here):
        a_here = spec_params.a_init * np.exp(-N_here) #find a
        H_here = spline1.eval(N_here) #find H
        eps_here = spline2.eval(N_here) #find eps
        return a_here*a_here * H_here*H_here * (2 - eps_here) - k*k #Find Q


    #Use the N grid, the Q_vals, and Q_of_N func
    def find_turning_point(Ngrid, Q_vals, Q_of_N):
        #first need to find where there is a sign change
        for i in range(len(Q_vals)-1): #in length of 0 to Qvals-1
            if Q_vals[i] * Q_vals[i+1] < 0: #finds points where Qi is less than 0 and Qi is greater than 0 and put them in N_L or N_R
                N_L = Ngrid[i]
                N_R = Ngrid[i+1]
                break
        else:
            raise RuntimeError("No sign change in Q(N)! No turning point found.")

        #N star should be between these two points of ti and ti+1
        #brentq takes a function that changes sign and the intervals where Q<0 and Q>0
        N_star = brentq(Q_of_N, N_L, N_R)
        return N_star
        
        
    #Our turning point is as follows:
    N_star_from_N = find_turning_point(Ngrid, Q_vals_N, Q_of_N)
    print("N_star from N",N_star_from_N)
    
    #Q is shifted such that Q(0)=0 to follow math in this paper: https://arxiv.org/pdf/astro-ph/9805173
    Q_spline = UnivariateSpline(Ngrid[::-1], Q_vals_N[::-1], s=0.0)
    Q_shift = Q_spline(N_star_from_N)
    Q_vals_N = Q_vals_N - Q_shift
    
    def Qtilde_of_N(N):
        return Q_of_N(N) - Q_shift

    # Q_vals_N from the same shifted function so its consistent
    Q_vals_N = np.array([Qtilde_of_N(N) for N in Ngrid])

    
    #We can also confirm that behavior here:
    print("")
    print("From N: N⋆-dN, N⋆, N⋆+dN")
    for Ncheck in [N_star_from_N-1e-3, N_star_from_N, N_star_from_N+1e-3]:
        print(Ncheck, Q_of_N(Ncheck))
    # Find index of N_star
    idx_star_N = np.argmin(np.abs(Ngrid - N_star_from_N))
    print("Q(N_star) =", Q_of_N(N_star_from_N))
        
    
    #Let's go ahead and check that turning point with our expression that describes the horizon size at crossing
    plt.plot(Ngrid,k_over_aH, color='forestgreen', label = r"$k/aH=1$")
    plt.plot(Ngrid,k_over_aH_with_eps, color='maroon', label = r"$k/aH \sqrt{2-\epsilon}=1$", linestyle="--")
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.axhline(y=1, color='gray', linestyle='--', label='k/aH=1')
    plt.xlabel("N"); plt.ylabel("k/aH(N)")
    plt.legend()
    plt.show()
    
  
    #We can also check the behavior of Q in the limits that we expect, which are basically where eitehr k^2 is dominant or a^2 H^2 (2-eps) is dominant. Of course we can only check that for Q in the correct limits where Q < 0 or Q > 0

    #Q limits
    mask_Q_osc = Q_vals_N < 0
    mask_Q_exp = Q_vals_N > 0

    #Expected results in either regime
    k2 = k**2
    bg_term = (a_of_Ngrid**2) * (H_of_Ngrid**2) * (2 - eps_of_Ngrid)
    
    plt.plot(Ngrid, Q_vals_N,label="Computed Q(N)", color='b')
    plt.plot(Ngrid[mask_Q_osc],-k2 * np.ones_like(Ngrid[mask_Q_osc]),label=r"$-k^2$ (osc)", color="red",linestyle="--")
    plt.plot(Ngrid[mask_Q_exp],bg_term[mask_Q_exp],label=r"$a^2H^2(2-\epsilon)$ (exp)",color="green",linestyle="--")
#    plt.xlim(61.7,62)
#    plt.ylim(-3e-126,3e-126)
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(N_star_from_N, color="gray", linestyle=":")
    plt.xlim(N_star_from_N-3,N_star_from_N+3)
    plt.ylim(-1e-124,1e-123)
    plt.xlabel("N")
    plt.ylabel("Q from Exponential and Oscillatory Expectation")
    plt.title("Decomposition of Q(N) with Explicit Forms")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.show()
    
    plt.plot(Ngrid, Q_vals_N,label="Computed Q(N)", color='b')
    plt.plot(Ngrid[mask_Q_osc], Q_vals_N[mask_Q_osc],label="osc Q", color='r', linestyle="--")
    plt.plot(Ngrid[mask_Q_exp], Q_vals_N[mask_Q_exp],label="exp Q", color='g', linestyle="--")
    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.ylim(-1e-126,1e-126)
    plt.axhline(0, color='k')
    plt.xlabel("N"); plt.ylabel("Q(N)")
    plt.axvline(N_star_from_N, color='gray')
    plt.title("Q(N)")
    plt.legend()
    plt.show()


##To find S0:


    S0 = np.zeros_like(Ngrid)

    # Define distance-from-turning-point variable
    x = N_star_from_N - Ngrid   # x=0 at turning point which occurs at N_star

    # Masks
    mask_exp = x > 0    # N < N*  → superhorizon (exponential)
    mask_osc = x < 0    # N > N*  → subhorizon (oscillatory)

    # Exponential side (x > 0, S0 > 0)
    x_exp = x[mask_exp]
    f_exp = np.sqrt(Q_vals_N[mask_exp]) / (a_of_Ngrid[mask_exp] * H_of_Ngrid[mask_exp])


    # We want to integrate from x=0 to x
    x_exp_inc = x_exp #already increasing without having to flip
    f_exp_inc = f_exp #already increasing without having to flip

    S0_exp = cumulative_trapezoid(
        f_exp_inc,
        x_exp_inc,
        initial=0.0
    )

    # Thus S0 in the exponential region looks like:
    S0[mask_exp] = S0_exp


    # Oscillatory side (x < 0, S0 < 0)
    x_osc = x[mask_osc]
    f_osc = np.sqrt(-Q_vals_N[mask_osc]) / (a_of_Ngrid[mask_osc] * H_of_Ngrid[mask_osc])

    # Integrate over |x| from 0 → |x|
    x_osc_dist = -x_osc              # make distance positive
    x_osc_inc  = x_osc_dist[::-1]
    f_osc_inc  = f_osc[::-1]

    S0_osc = cumulative_trapezoid(
        f_osc_inc,
        x_osc_inc,
        initial=0.0
    )
    
#    print('S0_osc',S0_osc[:5])
#    print('S0_exp',S0_exp[:5])


    # Flip back and apply physical minus sign
    S0[mask_osc] = -S0_osc[::-1]
    
    #We save this S0 so far as the S0 for either side and call it numeric. After this step I will focus on the shelf that accors at S0(0)
    S0_numeric = S0.copy()

#Now, I will try and use quad and not some equation to see about matching the kink
    def integrand_tp(N):
        Q = Qtilde_of_N(N)  # or Q_of_N if you choose not to shift
        return np.sqrt(np.abs(Q)) / (a_of_N(N) * H_of_N(N))

    def S0_quad_at(N):
        if N == N_star_from_N:
            return 0.0
        val, err = quad(integrand_tp, N_star_from_N, N,
                        epsabs=1e-11, epsrel=1e-11, limit=200)
        # enforce your sign convention: S0>0 for N < N*, S0<0 for N > N*
        return np.sign(N_star_from_N - N) * np.abs(val)
        
    delta_quad = 0.013
#    mask_q = np.abs(Ngrid - N_star_from_N) < delta_quad
    mask_q = np.abs(x) < delta_quad


    S0 = S0_numeric.copy()
    S0[mask_q] = np.array([S0_quad_at(N) for N in Ngrid[mask_q]])
    
    
    print("Qtilde(N_star) =", Qtilde_of_N(N_star_from_N))
    print("min |Qtilde| on grid =", np.min(np.abs(Q_vals_N)))
    
    f_grid = np.sqrt(np.abs(Q_vals_N)) / (a_of_Ngrid * H_of_Ngrid)
    mask_zoom = np.abs(Ngrid - N_star_from_N) < 0.2
    plt.plot(Ngrid[mask_zoom], f_grid[mask_zoom])
    plt.axvline(N_star_from_N, ls="--", c="k")
    plt.title("f(N) = sqrt(|Q|)/(aH) near turning point")
    plt.show()
    
    mask_zoom = np.abs(Ngrid - N_star_from_N) < 0.2
    plt.plot(Ngrid[mask_zoom], (S0 - S0_numeric)[mask_zoom])
    plt.axvline(N_star_from_N, ls="--", c="k")
    plt.title("S0_patch - S0_numeric near turning point")
    plt.show()
    
#    # correction relative to baseline
#    dS0 = S0 - S0_numeric
#
#    # smooth in a tiny region around turning point in x-space
#    eps_x = 0.0065   # start ~0.5*delta_quad; tune 0.004–0.010
#
#    mask_s = np.abs(x) < eps_x
#
#    # if we have enough points
#    if np.sum(mask_s) >= 3:
#        # because Ngrid is ordered, x[mask_s] is also ordered (monotonic),
#        # so first/last are valid endpoints without sorting
#        dS0_interp = np.interp(
#            x[mask_s],
#            [x[mask_s][0], x[mask_s][-1]],
#            [dS0[mask_s][0], dS0[mask_s][-1]]
#        )
#        dS0[mask_s] = dS0_interp
#
#    # reconstruct
#    S0 = S0_numeric + dS0


#    #To try and fix the shelf at S0(0)=0, this was an issue because we also need derivative to be equal to zero there and were not seeing that before.
#    dN = 1e-6 #define dN for the derivative parameter
#    dQdN_star = (Q_of_N(N_star_from_N + dN) - Q_of_N(N_star_from_N - dN)) / (2 * dN) #this should be the derivative of Q or the change in Q over a small range near Nstar
#    
#    #Around the turning point we know that S0 actually reduces to the following expression which is some coeff *
#    def S0_turning_point(N):
#        coeff = (2/3) * np.sqrt(np.abs(dQdN_star)) / (a_of_N(N_star_from_N) * H_of_N(N_star_from_N))
#        return np.sign(N_star_from_N - N) * coeff * np.abs(N - N_star_from_N)**(3/2)
#
#    #This delta will define the range over which to assume S0 takes this form of this mask near the turning point
#    delta = 0.03
#    S0 = S0_numeric.copy()
#
#    mask_near = np.abs(Ngrid - N_star_from_N) < delta #mask for everything within delta near the turning point
#    mask_far  = ~mask_near #tilde is that negation operator so basically everything not at the mask_near set of points
#
#    # Replace only the near-turning-point region
#    S0[mask_near] = S0_turning_point(Ngrid[mask_near])




    #Let's check this out now with a plot
    plt.figure(figsize=(5,3))
    plt.plot(Ngrid, S0, label='S0', color='b')
    plt.title("S0(N)")
    plt.axvline(0, color='gray', linestyle='--', label='Zero Point')
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.xlabel("N"); plt.ylabel("S0")
    plt.legend()
    plt.show()
    
    #Checks that S behavior is following the expected trend in each region
    plt.figure(figsize=(5,3))
    plt.plot(Ngrid, S0, label='S0', color='b')
    plt.plot(Ngrid[mask_Q_osc], S0[mask_Q_osc], label='S0 osc', color='r', linestyle="--")
    plt.plot(Ngrid[mask_Q_exp], S0[mask_Q_exp], label='S0 exp', color='g', linestyle="--")
    plt.title("S0(N)")
    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.ylim(-0.02,0.02)
    plt.axhline(0, color='gray', linestyle='--', label='Zero Point')
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.xlabel("N"); plt.ylabel("S0")
    plt.legend()
    plt.show()
    
    
    plt.title("S0(N) near turning point")
    mask_zoom = np.abs(Ngrid - N_star_from_N) < 0.1   # SUPER close
    plt.plot(Ngrid[mask_zoom], S0[mask_zoom], color='b')
    plt.axhline(y=0, color='gray', linestyle='--', label='Zero Point')
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.xlabel("N"); plt.ylabel("S0")
    plt.show()
    
    
    #We define zeta using S0
    zeta = np.sign(S0) * (1.5 * np.abs(S0))**(2/3) #keep in mind the odd way python does things to the fractional power
    
#    #This is shifted to make sure that zeta(0)=0 as well
#    zeta_spline = UnivariateSpline(Ngrid[::-1], zeta[::-1], s=0.0)
#    zeta_shift = zeta_spline(N_star_from_N)
#    zeta = zeta - zeta_shift
#    
    #We can check out zeta and make sure you see oscillatory and exponential behavior
    plt.figure(figsize=(5,3))
    plt.plot(Ngrid, zeta, label='zeta', color='b')
    plt.plot(Ngrid[mask_Q_osc], zeta[mask_Q_osc], label='zeta osc', color='r', linestyle="--")
    plt.plot(Ngrid[mask_Q_exp], zeta[mask_Q_exp], label='zeta exp', color='g', linestyle="--")
    plt.title("zeta(N)")
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.axhline(y=0, color='gray', linestyle='--', label='Zero Point')
    plt.axvline(0, color='gray', linestyle='--', label='y=0')
    plt.xlabel("N"); plt.ylabel("zeta")
    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.ylim(-0.005,0.005)
    plt.legend()
    plt.show()
    
    
    #Zoom in on zeta around the turning point
    plt.title("zeta(N) near turning point")
    plt.plot(Ngrid[mask_zoom], zeta[mask_zoom])
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.axhline(0, color='gray', linestyle='--', label='Zero Point')
    plt.xlabel("N"); plt.ylabel("zeta")
    plt.show()
    
    #Comparing zeta and S0
    plt.title("zeta(N) and S0(N) near turning point")
    plt.plot(Ngrid[mask_zoom], zeta[mask_zoom], color='pink', label="zeta(N)")
    plt.plot(Ngrid[mask_zoom], S0[mask_zoom], color='turquoise', label="S0(N)")
    plt.axvline(x=N_star_from_N, color='k', linestyle='--', label='Horizon Exit')
    plt.axhline(0, color='gray', linestyle='--', label='Zero Point')
    plt.xlabel("N"); plt.ylabel("zeta or S0")
    plt.show()
    
    # Numerical derivatives
    dS0_dN = np.gradient(S0, Ngrid)
    dzeta_dN = np.gradient(zeta, Ngrid)


    print("At turning point over Ngrid:")
    print("  S0(N*) =", S0[idx_star_N])
    print("  dS0/dN(N*) =", dS0_dN[idx_star_N])
    print("  zeta(N*) =", zeta[idx_star_N])
    print("  dzeta/dN(N*) =", dzeta_dN[idx_star_N])
    
    plt.figure(figsize=(6,4))
    plt.plot(Ngrid, dS0_dN, lw=1)
    plt.axvline(N_star_from_N, color='k', ls='--', label=r'$N_*$')
    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.ylim(-0.3,0.1)
    plt.axhline(0, color='k', ls=':')
    plt.xlabel(r"$N$")
    plt.ylabel(r"$dS_0/dN$")
    plt.title(r"$dS_0/dN$ vs $N$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
#    dS0_expected = np.sign(Ngrid - N_star_from_N) * np.sqrt(np.abs(Q_vals_N))
#    dS0_expected = np.sign(N_star_from_N - Ngrid) * np.sqrt(np.abs(Q_vals_N)) / (a_of_Ngrid * H_of_Ngrid)


    plt.figure(figsize=(6,4))
    plt.plot(Ngrid, dS0_dN, label="numeric ∇S0", lw=1)
#    plt.plot(Ngrid, dS0_expected, '--', label=r"$\pm\sqrt{|Q_{\rm eff}|}$")
    plt.axvline(N_star_from_N, color='k', ls='--')
    plt.axhline(0, color='k', ls=':')
    plt.xlim(N_star_from_N-0.5, N_star_from_N+0.5)
    plt.ylim(-0.3,0.1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

        

    #Below we make a super big check so that I am getting the correct S0 as expected from the behavior of zeta
    #Have to write this special cause zeta is signed. Apparently the fractional power only acts on positive quantity. zeta is already signed so all that is negative
    expectedS0neg = -(2/3*np.abs(zeta[mask_Q_osc]))**(3/2) #should represent (-2/3*S0)^3/2
    expectedS0pos = (2/3*np.abs(zeta[mask_Q_exp]))**(3/2) #should represent (+2/3*S0)^3/2
    N_zoom = Ngrid > 5

    plt.figure(figsize=(8,5))
    plt.plot(Ngrid, S0, label="S0", color='g')
    plt.plot(Ngrid[mask_Q_osc], expectedS0neg, label=r"S0-: $-(2/3*|\zeta|)^{3/2}$", linestyle='--', color = "brown")
    plt.plot(Ngrid[mask_Q_exp], expectedS0pos, label=r"S0+: $(2/3*|\zeta|)^{3/2}$", linestyle='--', color = "magenta")
    plt.title(r'Expected $S0(N)$ from $\zeta(N)$', fontsize=16, weight='bold', color='k')
    plt.xlabel(r'$N$', fontsize=14, color='#333333')
    plt.ylabel(r'$S0$ (from $\zeta$)', fontsize=14, color='#333333')
    plt.axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='$Horizon Exit$')
    plt.axhline(0, linestyle='--', color='k', linewidth=1.2, label='$Zero Point$')
#    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.grid(color='lightgrey', linestyle=':', linewidth=0.8)
    plt.legend(fontsize=12, loc='upper right', frameon=False)
    plt.tight_layout(pad=2)
    plt.show()
    
    
    #A place that we can sometimes get a kink is when we take a look at zeta/Q which appears in the prefactor
    #The plot below makes sure I am seeing correct exponential and oscillatory behavior
    plt.figure(figsize=(8,5))
    plt.plot(Ngrid, zeta/Q_vals_N , label=r'$\zeta(N)/Q(N)$', color='b', linewidth=1.5)
    plt.plot(Ngrid[mask_Q_osc], zeta[mask_Q_osc]/Q_vals_N[mask_Q_osc] , label=r'$Osc: \zeta(N)/Q(N)$', color='r', linestyle='--', linewidth=1.5)
    plt.plot(Ngrid[mask_Q_exp], zeta[mask_Q_exp]/Q_vals_N[mask_Q_exp] , label=r'$Exp: \zeta(N)/Q(N)$', color='g', linestyle='--', linewidth=1.5)
    plt.title(r'$\zeta(N)/Q(N)$', fontsize=16, weight='bold', color='k')
    plt.xlabel(r'$N$', fontsize=14, color='#333333')
    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.ylim(-0.2e126,0.2e126)
    plt.ylabel(r'$\zeta(N)/Q(N)$', fontsize=14, color='#333333')
    plt.axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='$Horizon Exit$')
    plt.grid(color='lightgrey', linestyle=':', linewidth=0.8)
    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.tight_layout(pad=2)
    plt.show()
    
    #This zeta/Q plot checks for the behavior of absolute power
    plt.figure(figsize=(8,5))
#    plt.plot(Ngrid, zeta/Q_vals_N , label=r'$\zeta(N)/Q(N)$', color='b', linewidth=1.5)
#    plt.plot(Ngrid, zeta/np.abs(Q_vals_N) , label=r'$\zeta(N)/|Q(N)|$', color='brown', linewidth=1.5,linestyle='-')
    plt.plot(Ngrid, np.abs(zeta/Q_vals_N), label=r'$|\zeta(N)/Q(N)|$', color='magenta', linewidth=1.5,linestyle='--')
#    plt.plot(Ngrid, np.abs(zeta)/Q_vals_N, label=r'$|\zeta(N)|/Q(N)$', color='turquoise', linewidth=1.5,linestyle='--')
    plt.title(r'$|\zeta(N)/Q(N)|$', fontsize=16, weight='bold', color='k')
    plt.xlabel(r'$N$', fontsize=14, color='#333333')
    plt.xlim(N_star_from_N-0.5,N_star_from_N+0.5)
    plt.ylim(-0.2e126,0.2e126)
    plt.ylabel(r'$|\zeta(N)/Q(N)|$', fontsize=14, color='#333333')
    plt.axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='$Horizon Exit$')
    plt.grid(color='lightgrey', linestyle=':', linewidth=0.8)
    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.tight_layout(pad=2)
    plt.show()
    

    #For airy analysis
    Ai_vals, Aip_vals, Bi_vals, Bip_vals = airy(zeta)
    airy_vals = -(Ai_vals + 1j * Bi_vals)
    #I applied a rotation here! It does not do anything but it matches very well with the numerical with this rotation

    #If we want to take a look at just the airy functions both the real and imaginary parts we can do that here
    plt.figure(figsize=(8,5))
    plt.plot(Ngrid[N_zoom], np.real(airy_vals)[N_zoom], label="Re(Ai + i Bi)", linestyle='--', color = "g")
    plt.plot(Ngrid[N_zoom], np.imag(airy_vals)[N_zoom], label="Im(Ai + i Bi)", linestyle='--', color = "r")
    plt.title(r'$Airy(\zeta(N))$', fontsize=16, weight='bold', color='k')
    plt.ylim(-5,5)
    plt.xlabel(r'$N$', fontsize=14, color='#333333')
    plt.ylabel(r'$Airy(\zeta(N))$', fontsize=14, color='#333333')
    plt.axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='$Horizon Exit$')
    plt.grid(color='lightgrey', linestyle=':', linewidth=0.8)
    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.tight_layout(pad=2)
    plt.show()
#    
#    #Following that we can also even test the real and imaginary parts over a to make sure the imaginary part survives
#    plt.figure(figsize=(8,5))
#    plt.title(r'$Airy(\zeta(N))$ Over a(N)', fontsize=16, weight='bold', color='k')
#    plt.xlabel(r'$N$', fontsize=14, color='#333333')
#    plt.ylabel(r'$Airy(\zeta(N))/a(N)$', fontsize=14, color='#333333')
#    plt.ylim(-2e63,4e63)
#    plt.grid(color='lightgrey', linestyle=':', linewidth=0.8)
#    plt.plot(Ngrid, Ai_vals/a_of_Ngrid, label="Ai(zeta)/a(N)", linestyle='--', color = "purple")
#    plt.plot(Ngrid, Bi_vals/a_of_Ngrid, label="Bi(zeta)/a(N)",linestyle='--', color = "teal")
#    plt.axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='$Horizon Exit$')
#    plt.tight_layout(pad=2)
#    plt.legend(fontsize=12, loc='best', frameon=False)
#    plt.show()

    
    #Finally, our final wkb approximation has a solution of the following form
    prefactor = np.abs((zeta/Q_vals_N))**(1/4)
    ut_N = np.sqrt(np.pi/2) * prefactor * (airy_vals)
    
    #Let's also go ahead and confirm these stats
    idx_S0_min   = np.argmin(np.abs(S0))
    idx_zeta_min = np.argmin(np.abs(zeta))
    print("True turning point:", N_star_from_N)
    print("S0 min at         :", Ngrid[idx_S0_min])
    print("zeta min at       :", Ngrid[idx_zeta_min])
    
    #Here we are indeed close BUT we add another part here that might seem a little silly but we are trying to match the numerical code
    #In the numerical code it assumes that shortly after horizon exit the value that is frozen and it uses that. What it is doing in this moment is applying a boundary condition where we expect phi -> constant. In other words, dphi/dN = d/dN u/a = 0. We can also check that boundary condition and if it is not there we can apply it ourselves. The numerical code assumes and forces the decaying mode to completely go away by doing this but we have surving decaying mode when we check dphi/dN.
    
    # Define an N at which to check where phi is truly going to a constant
    N_proj = N_star_from_N - 1.0   # ~1 efolds after exit. We can adjust this number a bit.
    idx = np.argmin(np.abs(Ngrid - N_proj)) #it would occur at this index
    phi = ut_N / a_of_Ngrid
    dphi_dN = np.gradient(phi, Ngrid) #build dphi/dN
    
    print("Before projection:")
    print("  max |dphi/dN| =", np.max(np.abs(dphi_dN)))
    print("  dphi/dN at late times:", dphi_dN[-5:])
    log_deriv = dphi_dN / phi
    print("log derivative at projection point:", log_deriv[idx])
    
    #This is not zero so it cannot match the numerical code one bit.
    #Find the index at the projection point which is after horizon exit
    idx = np.argmin(np.abs(Ngrid - N_proj))
    C_grow = phi[idx] #This tells us the amplitude of phi at that point
    ut_proj = C_grow * a_of_Ngrid #We expect u to be proportional to a after horizon crossing so we find that amplitude
    ut_final = ut_N.copy()
    ut_final[idx:] = ut_proj[idx:] #Starting from that index in ut_final we say that from that index we can assume the frozen amplitude just like the full numerical code
    phi_proj = ut_final / a_of_Ngrid #this now becomes our new phi
    dphi_dN_proj = np.gradient(phi_proj, Ngrid) #We can define again the derivative phi over the new phi and check the value to confirm that it indeed goes to 0
        
    print("\nAfter projection:")
    print("  max |dphi/dN| =", np.max(np.abs(dphi_dN_proj[idx:])))
    print("  dphi/dN at late times:", dphi_dN_proj[-5:])
    
    #Our final tensor mode appears then like so:
    ut_N = ut_final


    # Plot the Real parts of u_t(N)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Ngrid, np.real(ut_N), label=r'$u_{real}$', color='brown', linewidth=1.5)
    axes[0].axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[0].set_ylim(-5e31,5e31)
#    axes[0].set_xlim(N_star_from_N-5,N_star_from_N+3)
    axes[0].set_xlim(N_star_from_N-2,N_star_from_N+1)
    axes[0].set_title('Real Part of $u_t(N)$', fontsize=14)
    axes[0].set_xlabel(r'$N$', fontsize=12)
    axes[0].set_ylabel(r'$u_{real}$', fontsize=12)
    axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=10, frameon=False)


    # Plot the Imaginary parts of u_t(N)
    axes[1].plot(Ngrid, np.imag(ut_N), label=r'$u_{imag}$', color='darkkhaki', linewidth=1.5)
    axes[1].axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[1].set_ylim(-5e31,5e31)
#    axes[1].set_xlim(N_star_from_N-5,N_star_from_N+3)
    axes[1].set_xlim(N_star_from_N-2,N_star_from_N+1)
    axes[1].set_title('Imaginary Part of $u_t(N)$', fontsize=14)
    axes[1].set_xlabel(r'$N$', fontsize=12)
    axes[1].set_ylabel(r'$u_{imag}$', fontsize=12)
    axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[1].legend(fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.show()


    #Save the Re(u) and Im(u)
    ut_N_real =  np.real(ut_N)
    ut_N_imag =  np.imag(ut_N)


    #Plot phi behavior!
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Plot 1: phi=|u|/a
    wkb_color = '#DB7093'  # Vibrant rose for WKB approximation
    axes[0].plot(Ngrid, np.abs(ut_N)/a_of_Ngrid, label=r'$\phi = |u|/a$', color=wkb_color, linewidth=1.5)
    axes[0].axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[0].set_title('Mode Amplitude $\phi=|u_t(N)|/a$', fontsize=14)
    axes[0].set_xlabel(r'$N$', fontsize=12)
    axes[0].set_ylabel(r'$\phi=|u|/a$', fontsize=12)
    axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=10, frameon=False)


    # Plot 2: u^2/a^2
    amp_color = color='#333333'
    axes[1].plot(Ngrid, np.abs(ut_N)**2/a_of_Ngrid**2, label=r'$|u_t^2|/a^2$', color=amp_color, linewidth=1.5)
    axes[1].axvline(x=N_star_from_N, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[1].set_title('Amplitude Squared $\phi^2=|u_t(N)^2|/a^2$', fontsize=14)
    axes[1].set_xlabel(r'$N$', fontsize=12)
    axes[1].set_ylabel(r'$\phi^2=|u_t^2|/a^2$', fontsize=12)
    axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[1].legend(fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.show()

    #Now it is time to save what we need to save here for our mode and apply it to the tensor power spectrum. We get the real and imaginary parts and create a complex spline to use on a N range outside the tensorsys function.
    spline_ut_N_norm_real = CubicSpline(Ngrid[::-1], ut_N_real[::-1])
    spline_ut_N_norm_imag = CubicSpline(Ngrid[::-1], ut_N_imag[::-1])

        
    def spline_ut_N_norm_complex(N):
        return spline_ut_N_norm_real(N) + 1j * spline_ut_N_norm_imag(N)


    return ut_N, Ngrid, spline_ut_N_norm_complex, N_star_from_N

