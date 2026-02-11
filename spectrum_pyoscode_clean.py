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
import pyoscode




#This will upload specifically the file that we need for spectrum to run: H, eps, etc. and the other slow roll variables.
from calcpath import *

#I am going to extend to LISA scales
#Extend to nanograv scale
# Keep existing minimum
k_min = 7.791356504680618e-06   # h/Mpc
## Extend to NANOGrav-ish max (covers ~1e-9 to 1e-8 Hz band)
#k_max = 6.47e6                # h/Mpc
#Extend to LISA-ish gets to like 1 Hz and still covers the 0.1mHz
k_max = 6.47e14


# Keep current point counts
knos  = 1575   # number of k values to evaluate spectrum at  -> ks_eval.dat
kinos = 214    # number of k values to integrate modes on     -> ks.dat


# Log-spaced grids
ks_eval_extend = np.logspace(np.log10(k_min), np.log10(k_max), knos)
#[7.79135650e-06 7.92840248e-06 8.06785902e-06 ... 6.24825948e+06, 6.35816317e+06 6.47000000e+06]
ks_int_extend  = np.logspace(np.log10(k_min), np.log10(k_max), kinos)

# Write with scientific notation similar to current file
np.savetxt("ks_eval_extend.dat", ks_eval_extend, fmt="%.15E")
np.savetxt("ks_int_extend.dat",     ks_int_extend,  fmt="%.15E")

print("ks_eval_extend:", ks_eval_extend[0], "to", ks_eval_extend[-1], "N =", len(ks_eval_extend))
print("ks_int_extend:     ", ks_int_extend[0],  "to", ks_int_extend[-1],  "N =", len(ks_int_extend))



#What worked before is below
#knos = 1575 # total number of k-values to evaluate
#kinos = 214 # total number of k-values to use for integration
k_file = "ks_eval_extend.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks_int_extend.dat" # file containing k-values for integration
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
        
    
#=========================================================================================================================

        #We continue the story with the tensor power spectrum, which is not based on the numerical code anymore. Instead we use a WKB approximation to find this.

        """
        Tensor spectra
        """
        
#=========================================================================================================================

        start_tensors = time.time()
        print('Stats from actual mode evaluation part:')


    ## 1) DEFINE NGRID ON WHICH TO EVALUATE TENSOR POWER SPECTRUM
        
        Ngrid_ode = Nefolds[:count+1]
        Nfinal_num = Ngrid_ode[-1]

    ## 2) CALL THE TENSOR SYSTEM FUNCTION
        #WKB adjusted tensorsys
        ut_N_array, Ngrid_use, N_star, phi_squared_tensor  = tensorsys(N_start, N_exit, Nfinal_num, Ngrid_ode, spec_params, spline1, spline2, k,ru_init, iu_init, dru_init, diu_init)
        
    ## 3) GENERATE A SPLINE OF PHI THAT GETS EVALUATED AT AN N OF YOUR CHOICE
        phi_squared_spl = CubicSpline(Ngrid_use[::-1], phi_squared_tensor[::-1])


        #Use an N_freeze value close to the end but not exactly at the end
        N_eval = Nfinal_num + 5*(Ngrid_use[-2]-Ngrid_use[-1])
               # Ensure N_freeze is within the WKB grid
        if not (Ngrid_ode[-1] <= N_eval <= Ngrid_ode[0]):
            raise RuntimeError(f"N_eval={N_eval} outside WKB grid "
                               f"[{Ngrid_wkb[-1]}, {Ngrid_wkb[0]}]")
        print("N_eval",N_eval)
        
        #Evaluate at N_eval
        phi_squared_eval = float(phi_squared_spl(N_eval))

    ## EVALUATE TENSOR POWER SPECTRUM
    
        P_t[m] = 64*np.pi * (k**3/(2*np.pi**2)) * phi_squared_eval
        
        end_tensors= time.time()
        print(f"[DEBUG] tensor mode {m} took {end_tensors - start_tensors:.3f} s")

#=========================================================================================================================

        #DIAGNOSTICS FOR COMPARISON PURPOSES
        
        #I can evaluate variables over the numerical N using a spline created in tensorsys for ut
        ut_N_spline = CubicSpline(Ngrid_use[::-1], ut_N_array[::-1])
        u_t_ode_length = ut_N_spline(Ngrid_ode) #Make sure it is the same length as the original numerical one
        u_wkb_real = np.real(u_t_ode_length) #This is our real part
        u_wkb_imag = np.imag(u_t_ode_length) #This is our imag part
        
        u_wkb_squared = (u_wkb_real*u_wkb_real) +  (u_wkb_imag*u_wkb_imag) #u^2
        u_wkb = np.abs(u_wkb_real+1j*u_wkb_imag) #gets |u|
#        u_wkb = np.emath.sqrt(u_wkb_squared) #gets |u|
        a_ode = spec_params.a_init * np.exp(-Ngrid_ode) #a(N)
        u_wkb_squared_over_a2 = u_wkb_squared / (a_ode*a_ode)


        #I can mark horizon crossing like so:
        H_all = np.array([spline1.eval(Ni) for Ni in Ngrid_ode]) #H(N) for the whole Ngrid
        aH_length = a_ode * H_all
        
        #Find index where k-aH=tiny which is basically horizon crossing
        hc_idx = np.argmin(np.abs(k - aH_length))
        N_hc = Ngrid_ode[hc_idx]
        
        outfile = f"wkb_diag_mode_{m:04d}.dat"

        #Below I changed N_exit_val to N_exit see if that is more accurate since that is what I used in WKB approx loosely
        np.savetxt(
            outfile,
            np.column_stack([
                Ngrid_ode,
                u_wkb,
                u_wkb_real,
                u_wkb_imag,
                u_wkb_squared,
                u_wkb_squared_over_a2,
                np.ones_like(Ngrid_ode) * N_hc, #horizon crossing k=aH
                np.ones_like(Ngrid_ode) * N_star, #turning point Q=0
                a_ode
            ]),
            header="N   |u_wkb(N)|  u_wkb_real(N)  u_wkb_imag(N)   u_wkb^2(N)   |u_wkb|^2/a^2    N_hc  N_star  a(N)"
        )
        
        print(f"[WKB] saved diagnostics → {outfile}")
    

        #Counts time taken
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        print(f"Mode {m+1}/{kinos} (k = {kis[m]:.3e}) took {elapsed:.2f} s")
        print("Finished for this mode")
        print("")

#=========================================================================================================================

    #Normalize according to the value of k at the characteristic length scale
        k_pivot = knorm * 5.41e-58
        pivot_index = np.argmin(np.abs(kis - k_pivot)) #can also make it an integer to be explicit

        spec_norm = Amp / (P_s[pivot_index] + P_t[pivot_index])
        y[1] = np.sqrt(spec_norm)  # kept this behavior from original code (normalize H for later recon)

#=========================================================================================================================

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
    
    
    for i in range(knos):
        u_s[0, i] = ks[i]
        u_s[1, i] = spec_norm * spline6.eval(ks[i]*5.41e-58)

        u_t[0, i] = ks[i]
        u_t[1, i] = spec_norm * spline8.eval(ks[i]*5.41e-58)

    return status

#=========================================================================================================================

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
    
#=========================================================================================================================

def scalarsys(t, y, parameters):
    dydN = np.empty(2)

    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-2.*(1.-2.*(p.eps)-0.75*(p.sig) - (p.eps)*(p.eps) + 0.125*(p.sig)*(p.sig) + 0.5*(p.xi)))*y[0]

    return dydN
 
#=========================================================================================================================


def tensorsys(N_start, N_exit, Nfinal, Ngrid_ode, spec_params, spline1, spline2, k, ru_init, iu_init, dru_init, diu_init):

    print('Stats from tensorsys:')
   

   ## FIRST WE DEFINE A TENSOR SYSTEM THAT USES CONFORMAL TIME (THIS IS DONE BECAUSE PYOSCODE PREFERS THAT VARIABLE IT SEEMS--SMOOTHER FUNCTION TO EVALUATE)
    """
        Truly we must first define our numerical integrator in conformal time space instead of N. 
        It seems to struggle more with N as an independent variable. 

        We create a function that takes in:
        - an N range
        - spec_params
        - splines for H(N) and eps(N)
        - current k value
        - n_oscode (resolution of N grid)
        - n_plot (resolution for pyoscode)
        - tolerance
        - order
        - optional phase toggle
    """
           
#=========================================================================================================================


    def tensorsys_tau_solve(
        Ngrid_to_use, spec_params, splineH, splineeps, k,
        n_oscode, n_plot, rtol=1e-4, order=3,phase=True):

    # 1) Build an N grid (we have in descending: for first mode ex. early ~65 -> late ~0)
        N_hi = float(Ngrid_to_use[0])
        N_lo = float(Ngrid_to_use[-1])
        Ngrid = np.linspace(N_hi, N_lo, n_oscode)  # descending if N_hi>N_lo

        # Define background functions in convention a(N)=a_init e^{-N}
        aN = spec_params.a_init * np.exp(-Ngrid) #a(N)
        HN = np.array([splineH.eval(float(N)) for N in Ngrid]) #H(N)
        epsN = np.array([splineeps.eval(float(N)) for N in Ngrid]) #eps(N)

    # 2) Build conformal time tau(N): d tau / dN = -1/(aH) for N = e-folds remaining
        integrand = 1.0 / (aN * HN)  # positive
        # integrate from late to early so tau is negative at early times
        tau = cumulative_trapezoid(-integrand[::-1], Ngrid[::-1], initial=0.0)[::-1] #should go from very negative to 0 if inflation
        ## IMP: Defines our spline below of tau(N)
        tau_to_N = PchipInterpolator(tau, Ngrid) #gives N = N(tau)



    # 3) Build omega^2(tau) = k^2 - a^2 H^2 (2-eps)
    
        N_of_tau = tau_to_N(tau)
        a_tau = spec_params.a_init * np.exp(-N_of_tau)
        H_tau = np.array([splineH.eval(float(N)) for N in N_of_tau])
        eps_tau = np.array([splineeps.eval(float(N)) for N in N_of_tau])
        w2_tau = k*k - (a_tau*a_tau)*(H_tau*H_tau)*(2.0 - eps_tau)

        def w2_of_tau(tau_val):
            N_here = float(tau_to_N(tau_val))
            a_here = spec_params.a_init * np.exp(-N_here)
            H_here = splineH.eval(N_here)
            eps_here = splineeps.eval(N_here)
            return k*k - a_here*a_here * H_here*H_here * (2 - eps_here)
        
        # Slower to do this way:
        #w2_tau = np.array([w2_of_tau(t) for t in tau])

        print("w2(tau[0]) =", w2_tau[0])
        print("w2(tau[-1]) =", w2_tau[-1])
        print("fraction positive =", np.mean(w2_tau > 0))

    # 4) Find turning point tau_star (where w2 crosses 0)

        #Before I had been just finding the index this occured at but now I can find the root
        #Use the tau grid, the w2, and w2 func where w2 = w^2 in your DE
#        def find_turning_point(tau_grid, w2_tau, w2_of_tau):
#            #first need to find where there is a sign change
#            for i in range(len(w2_tau)-1): #in length of 0 to Qvals-1
#                if w2_tau[i] * w2_tau[i+1] < 0: #finds points where Qi is less than 0 and Qi is greater than 0 and put them in tau_L or tau_R
#                    tau_L = tau_grid[i]
#                    tau_R = tau_grid[i+1]
#                    break
#            else:
#                raise RuntimeError("No sign change in Q(tau)! No turning point found.")
#
#            #tau star should be between these two points of ti and ti+1
#            #brentq takes a function that changes sign and the intervals where Q<0 and Q>0
#            tau_star = brentq(w2_of_tau, tau_L, tau_R)
#
#            return tau_star

#        tau_star = find_turning_point(tau, w2_tau, w2_of_tau)
#        print("w^2(τ⋆) = ", w2_of_tau(tau_star))
#        N_star = float(tau_to_N(tau_star))
#        print("N_star",N_star)
            
        #Try this way for speed enhancement, but it should be doing the same thing
        # bracket (vectorized)
        s = np.sign(w2_tau)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        if len(idx) == 0:
            raise RuntimeError("No sign change in w2(tau)! No turning point found.")
        i = idx[0]
        tau_L, tau_R = tau[i], tau[i+1]

        # root using interpolated w2(tau)
        w2_interp = PchipInterpolator(tau, w2_tau)
        tau_star = brentq(w2_interp, tau_L, tau_R)

        print("w^2(tau_star) =", w2_interp(tau_star))
        N_star = float(tau_to_N(tau_star))
        print("N_star", N_star)


    #THIS IS THE MEAT MAN
    # Remember this is being done in CONFORMAL TIME here! We are solving basically, u'' + w^2 u = 0
    # 5) Prepare pyoscode inputs on ascending grid. Pyoscode requires t_eval to be in ascending order
        ts = tau  # ascending
        ws = np.emath.sqrt(w2_tau.astype(np.complex128))  # complex allowed
        gs = np.zeros_like(ts)  # g=0 in conformal time equation

        # 6) BD initial conditions at earliest time (most negative tau)
        # u = exp(-ik tau)/sqrt(2k), u' = -ik u
        tau_i = tau[0]
        if phase:
            u0 = (1.0 / np.emath.sqrt(2.0 * k)) * np.exp(1j * k * tau_i)
        else:
            u0 = 1.0 / np.emath.sqrt(2.0 * k)

        du0_dtau = -1j * k * u0
        

        # 7) Solve with pyoscode
        # Downsample output for plotting speed
        if n_plot is None or n_plot >= len(ts):
            t_eval = ts
        else:
            t_eval = np.linspace(ts[0], ts[-1], n_plot)

        sol = pyoscode.solve(
            ts=ts, ws=ws, gs=gs,
            ti=ts[0], tf=ts[-1],
            x0=u0, dx0=du0_dtau,
            t_eval=t_eval,
            order=order,
            rtol=rtol,
            check_grid=False)
     

        u_eval = np.array(sol["x_eval"]) #gives u(tau)

        return t_eval, u_eval, tau_star, tau_to_N, Ngrid, N_star
        #End of tensorsys_tau_solve function
        
#=========================================================================================================================

## RUN THE TENSOR SYS FOR WKB BASED SOLUTION IN CONFORMAL TIME
    tau_plot, u_plot, tau_star, tau_to_N_spline, Ngrid, N_star = tensorsys_tau_solve(
        Ngrid_to_use=Ngrid_ode,
        spec_params=spec_params,
        splineH=spline1,
        splineeps=spline2,
        k=k,
        n_oscode=8000,
        n_plot=15000,
        rtol=1e-6,
        order=3,
        phase=True)

#=========================================================================================================================


## PLOT:
    ## Build a(tau)
    N_of_tauplot = tau_to_N_spline(tau_plot)
    a_of_tauplot = spec_params.a_init * np.exp(-N_of_tauplot)
    
        #If Needed can also get H(tau(N)) and eps(tau(N))
    #    H_of_tauplot = np.array([spline1.eval(float(N)) for N in N_of_tauplot])
    #    eps_of_tauplot = np.array([spline2.eval(float(N)) for N in N_of_tauplot])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Top panel: u(tau)
    axes[0].plot(tau_plot, np.real(u_plot), label="Re(u)")
    axes[0].plot(tau_plot, np.imag(u_plot), "--", label="Im(u)")
    axes[0].plot(tau_plot, np.abs(u_plot), label="|u|", color="k", linestyle="--")
    axes[0].axvline(0, color="gray", linestyle=":")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$u_k$")
    axes[0].set_ylim(-1e32, 1e32)
    axes[0].set_title(r"Mode Function $u_k(\tau)$")
    axes[0].legend()

    # Bottom panel: phi = u/a
    phi_tau = u_plot / a_of_tauplot

    axes[1].plot(tau_plot, np.real(phi_tau), label="Re(u/a)")
    axes[1].plot(tau_plot, np.imag(phi_tau), "--", label="Im(u/a)")
    axes[1].plot(tau_plot, np.abs(phi_tau), label="|u/a|", color="k", linestyle="--")
    axes[1].axvline(0, color="gray", linestyle=":")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$\phi = u_k/a$")
    axes[1].set_title(r"Physical Mode $\phi(\tau)$")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

#=========================================================================================================================

## SWITCH VARIABLES NOW TO E-FOLDS (N)
    
    # a(N) from N(tau) and a(tau)
    spline_a_N = PchipInterpolator(N_of_tauplot[::-1], a_of_tauplot[::-1])
    a_N = spline_a_N(Ngrid)
    
    # tau(Ngrid)
    tau_of_N = PchipInterpolator(N_of_tauplot[::-1], tau_plot[::-1])  # N decreasing -> reverse to increasing
    tau_on_Ngrid = tau_of_N(Ngrid)  # tau(Ngrid)
    
    # Now interpolate u(tau) on tau grid (monotone, safe)
    u_re_spline = PchipInterpolator(tau_plot, np.real(u_plot))
    u_im_spline = PchipInterpolator(tau_plot, np.imag(u_plot))

    # To get u(N) I take Re(u(tau(N))) + Im(u(tau(N)))
    ut_N = u_re_spline(tau_on_Ngrid) + 1j*u_im_spline(tau_on_Ngrid)
        
#=========================================================================================================================

## PLOT:
    #Real and imaginary parts of ut_N
    
    #Save the Re(u) and Im(u)
    ut_N_real =  np.real(ut_N)
    ut_N_imag =  np.imag(ut_N)
    
    # Plot the Real parts of u_t(N)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(Ngrid, ut_N_real, label=r'$u_{real}$', color='brown', linewidth=1.5)
    axes[0].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[0].set_ylim(-5e31,5e31)
    axes[0].set_xlim(N_star-5,N_star+2)
    axes[0].set_title('Real Part of $u_t(N)$', fontsize=14)
    axes[0].set_xlabel(r'$N$', fontsize=12)
    axes[0].set_ylabel(r'$u_{real}$', fontsize=12)
    axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=10, frameon=False)


    # Plot the Imaginary parts of u_t(N)
    axes[1].plot(Ngrid, ut_N_imag, label=r'$u_{imag}$', color='darkkhaki', linewidth=1.5)
    axes[1].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[1].set_ylim(-5e31,5e31)
    axes[1].set_xlim(N_star-5,N_star+2)
    axes[1].set_title('Imaginary Part of $u_t(N)$', fontsize=14)
    axes[1].set_xlabel(r'$N$', fontsize=12)
    axes[1].set_ylabel(r'$u_{imag}$', fontsize=12)
    axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[1].legend(fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.show()
    
#=========================================================================================================================

## PLOT:
    #Mode amplitude and amplitude squared
    
    #Plot phi behavior!
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # Plot 1: phi=|u|/a
    wkb_color = '#DB7093'  # Vibrant rose for WKB approximation
    axes[0].plot(Ngrid, np.abs(ut_N_real+1j*ut_N_imag)/a_N, label=r'$\phi = |u|/a$', color=wkb_color, linewidth=1.5)
    axes[0].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[0].set_title(r'Mode Amplitude $\phi=|u_t(N)|/a$', fontsize=14)
    axes[0].set_xlabel(r'$N$', fontsize=12)
    axes[0].set_ylabel(r'$\phi=|u|/a$', fontsize=12)
    axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=10, frameon=False)


    # Plot 2: u^2/a^2
    amp_color = '#333333'
    axes[1].plot(Ngrid, (np.abs(ut_N_real+1j*ut_N_imag)*np.abs(ut_N_real+1j*ut_N_imag))/(a_N*a_N), label=r'$|u_t^2|/a^2$', color=amp_color, linewidth=1.5)
    axes[1].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[1].set_title(r'Amplitude Squared $\phi^2=|u_t(N)^2|/a^2$', fontsize=14)
    axes[1].set_xlabel(r'$N$', fontsize=12)
    axes[1].set_ylabel(r'$\phi^2=|u_t^2|/a^2$', fontsize=12)
    axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[1].legend(fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.show()
    
#=========================================================================================================================

## DEFINE PHI AND PHI SQUARED TO BE USED IN

#    #Now it is time to save what we need to save here for our mode and apply it to the tensor power spectrum. We get the real and imaginary parts and square them

    u_squared = (ut_N_real*ut_N_real) + (ut_N_imag*ut_N_imag)
    phi_squared = u_squared/(a_N*a_N)
        
    return ut_N, Ngrid, N_star, phi_squared

