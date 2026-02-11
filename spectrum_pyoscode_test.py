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
        
        print("DEBUG: y_final[NEQS] =", y_final[NEQS])
        print("DEBUG: Nfinal        =", Nfinal)

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
        Nfinal_num = Ngrid_ode[-1]
        
        print("DEBUG just before WKB call:")
        print("  Nfinal =", Nfinal)
        print("  current count =", count)
        print("  Nefolds[0], Nefolds[count] =", Nefolds[0], Nefolds[count])
        print("  min N in Nefolds[:count+1] =", np.min(Nefolds[:count+1]))


        #WKB adjusted
        ut_N_array, Ngrid_use, N_star, phi2_tensor  = tensorsys(N_start, N_exit, Nfinal_num, Ngrid_ode, spec_params, spline1, spline2, k,ru_init, iu_init, dru_init, diu_init)
        

##Without Ngrid_ode for phi squared, from here:
#        N_freeze = 0.5

                           
                               
        # ---- DIAGNOSTIC BLOCK ----
        print("--------------------------------------------------------------------------------------")
        x = Ngrid_use[::-1]
        y = ut_N_array[::-1]

        print("x finite?", np.all(np.isfinite(x)), "  y finite?", np.all(np.isfinite(y)))
        print("x dtype:", x.dtype, "y dtype:", y.dtype)
        print("x min/max:", np.min(x), np.max(x))
        print("y min/max:", np.min(y), np.max(y))

        # monotonic check (CubicSpline requires strict increase)
        dx = np.diff(x)
        print("min dx:", dx.min(), "max dx:", dx.max())
        print("any dx <= 0 ?", np.any(dx <= 0))

        # duplicate check
        print("unique x:", np.unique(x).size, "len x:", len(x))

        print("--------------------------------------------------------------------------------------")

        #just trying a diff N_freeze
        N_freeze = Nfinal_num + 5*(Ngrid_use[-2]-Ngrid_use[-1])
               # Ensure N_freeze is within the WKB grid
        if not (Ngrid_ode[-1] <= N_freeze <= Ngrid_ode[0]):
            raise RuntimeError(f"N_freeze={N_freeze} outside WKB grid "
                               f"[{Ngrid_wkb[-1]}, {Ngrid_wkb[0]}]")
          

        phi2_spl = CubicSpline(Ngrid_use[::-1], phi2_tensor[::-1])
        phi2_freeze = float(phi2_spl(N_freeze))

        P_t[m] = 64*np.pi * (k**3/(2*np.pi**2)) * phi2_freeze

        print(f"[Pt] WKB native grid: N_freeze={N_freeze}, phi2={phi2_freeze:.6e}")
        
        end_tensors= time.time()
        print(f"[DEBUG] tensor mode {m} took {end_tensors - start_tensors:.3f} s")

       
        #For saving and comparing purposes
        #I can evaluate variables over the numerical N using a spline created in tensorsys for ut
        ut_N_spline = CubicSpline(Ngrid_use[::-1], ut_N_array[::-1])
        u_t_ode_length = ut_N_spline(Ngrid_ode)
        u_wkb_real = np.real(u_t_ode_length) #This is our real part
        u_wkb_imag = np.imag(u_t_ode_length) #This is our imag part

        #We proceed to make a spline for Re and Im
#        spline_real = CubicSpline(Nordered[:count+1], u_wkb_real[::-1])
#        spline_imag = CubicSpline(Nordered[:count+1], u_wkb_imag[::-1])
        #same as
#        spline_real = CubicSpline(Ngrid_ode[::-1], u_wkb_real[::-1])
#        spline_imag = CubicSpline(Ngrid_ode[::-1], u_wkb_imag[::-1])
#   
   
        u_wkb_sq = u_wkb_real**2 +  u_wkb_imag**2 #u^2
        u_wkb = np.sqrt(u_wkb_sq) #gets |u|
        a_ode = spec_params.a_init * np.exp(-Ngrid_ode)
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



#        if kis[m] == knorm * 5.41e-58: # normalize here
#            spec_norm = Amp / (P_s[m]+P_t[m])
#            """
#            This is a little different from the C code,
#            because the y[1] change is outside the if statement
#            """
#            y[1] = np.sqrt(spec_norm) # normalize H for later recon
        k_pivot = knorm * 5.41e-58
        pivot_index = np.argmin(np.abs(kis - k_pivot)) #can also make it an integer to be explicit

        spec_norm = Amp / (P_s[pivot_index] + P_t[pivot_index])
        y[1] = np.sqrt(spec_norm)  # kept this behavior from original code (normalize H for later recon)

        print(f"[NORM] pivot_index={pivot_index}, "
              f"k_pivot_target={k_pivot:.3e}, k_used={kis[pivot_index]:.3e}, "
              f"spec_norm={spec_norm:.3e}")
        print("[DEBUG] kis range (Planck):", kis.min(), "→", kis.max())
        print("[DEBUG] k_pivot (Planck):", k_pivot)
        
            

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
 
 
def tensorsys(N_start, N_exit, Nfinal, Ngrid_ode, spec_params, spline1, spline2, k, ru_init, iu_init, dru_init, diu_init):

    
    print("")
    print('Stats from tensorsys:')
    print("u_init", ru_init + iu_init)
    print("du_init", dru_init + diu_init)


##DEFINE INITIAL NGRID
#
##We define the range of e-folds over which to evaluate the mode:
#    Ngrid_original = Ngrid_ode.copy() #This one is identical to that which is used for the original numerical
#
#    # Build an even uniform grid specifically for oscode
#    N_hi = Ngrid_original[0]
#    N_lo = Ngrid_original[-1]
#    n_oscode = 5000
#    Ngrid = np.linspace(N_hi, N_lo, n_oscode)
#    
##DEFINE TAU GRID
#    a_of_N = lambda N: spec_params.a_init * np.exp(-N)
#    H_of_N = lambda N: spline1.eval(N)
#    eps_of_N = lambda N: spline2.eval(N)
#    integrand_tau = np.array([1.0 / (a_of_N(N) * H_of_N(N)) for N in Ngrid])
#    
#    #Define tau from the integral
#    tau_range = cumulative_trapezoid(-integrand_tau[::-1], Ngrid[::-1], initial=0)[::-1]
#    tau_to_N = PchipInterpolator(tau_range, Ngrid)
#    tau_grid = tau_range.copy()
#    N_of_taugrid = tau_to_N(tau_grid) #Gives N(tau)
#    
#    #Apply to a(tau)=a(N(tau)), H(tau)=H(N(tau)), eps(tau)=eps(N(tau))
#    a_of_taugrid = spec_params.a_init * np.exp(-N_of_taugrid)
#    H_of_taugrid = np.array([spline1.eval(float(N)) for N in N_of_taugrid])
#    eps_of_taugrid = np.array([spline2.eval(float(N)) for N in N_of_taugrid])
#
### Define Q(tau)
##    Qvals = ((k*k)/((a_of_Ngrid*a_of_Ngrid) * (H_of_Ngrid*H_of_Ngrid))) - (2 - eps_of_Ngrid)
#    Qvals = k**2 - (a_of_taugrid**2) * (H_of_taugrid**2) * (2 - eps_of_taugrid)
#    
#    ##Method to find tau_star
#    #define Q of tau first, cause we need to be able to get tau_star so I can define w
#    def Q_of_tau(tau):
#        N_here = float(tau_to_N(tau)) #converts tau to N
#        a_here = spec_params.a_init * np.exp(-N_here) #find a
#        H_here = spline1.eval(N_here) #find H
#        eps_here = spline2.eval(N_here) #find eps
#        return a_here*a_here * H_here*H_here * (2 - eps_here) - k*k


#    #Before I had been just finding the index this occured at but now I can find the root
#    #Use the tau grid, the Q_vals, and Q_of_tau func
#    def find_turning_point(tau_grid, Q_vals, Q_of_tau):
#        #first need to find where there is a sign change
#        for i in range(len(Q_vals)-1): #in length of 0 to Qvals-1
#            if Q_vals[i] * Q_vals[i+1] < 0: #finds points where Qi is less than 0 and Qi is greater than 0 and put them in tau_L or tau_R
#                tau_L = tau_grid[i]
#                tau_R = tau_grid[i+1]
#                break
#        else:
#            raise RuntimeError("No sign change in Q(tau)! No turning point found.")
#
#        #tau star should be between these two points of ti and ti+1
#        #brentq takes a function that changes sign and the intervals where Q<0 and Q>0
#        tau_star = brentq(Q_of_tau, tau_L, tau_R)
#
#        return tau_star


#    tau_star = find_turning_point(tau_grid, Qvals, Q_of_tau)
#    print("Q(τ⋆) = ", Q_of_tau(tau_star))
#    N_star = float(tau_to_N(tau_star))
#    print("N_star",N_star)


    def tensorsys_tau_solve(
        Ngrid_ode, spec_params, splineH, splineeps, k,
        n_oscode, n_plot, rtol=1e-4, order=3,
        use_shifted_w=True, phase=True
    ):

        # 1) Build an N grid (descending: early ~65 -> late ~0)
        N_hi = float(Ngrid_ode[0])
        N_lo = float(Ngrid_ode[-1])
        Ngrid = np.linspace(N_hi, N_lo, n_oscode)  # descending if N_hi>N_lo

        # background functions in your convention a(N)=a_init e^{-N}
        aN = spec_params.a_init * np.exp(-Ngrid)
        HN = np.array([splineH.eval(float(N)) for N in Ngrid])
        epsN = np.array([splineeps.eval(float(N)) for N in Ngrid])

        # 2) Build conformal time tau(N): d tau / dN = -1/(aH) for N = e-folds remaining
        integrand = 1.0 / (aN * HN)  # positive
        # integrate from late to early so tau is negative at early times
        tau = cumulative_trapezoid(-integrand[::-1], Ngrid[::-1], initial=0.0)[::-1]
#        print("tau",tau)


        # 3) Build omega^2(tau) = k^2 - a^2 H^2 (2-eps)
        w2_tau = (k**2) - ((aN**2) * (HN**2) * (2.0 - epsN))
        print("w2(tau[0]) =", w2_tau[0])
        print("w2(tau[-1]) =", w2_tau[-1])
        print("fraction positive =", np.mean(w2_tau > 0))


        # 4) Find turning point tau_star (where w2 crosses 0)
#        def Q_of_tau(tau):
#            N_here = float(tau_to_N(tau)) #converts tau to N
#            a_here = spec_params.a_init * np.exp(-N_here) #find a
#            H_here = spline1.eval(N_here) #find H
#            eps_here = spline2.eval(N_here) #find eps
##            H_here = splineH.eval(N_here)
##            eps_here = splineeps.eval(N_here)
#            return a_here*a_here * H_here*H_here * (2 - eps_here) - k*k

        tau_to_N = PchipInterpolator(tau, Ngrid) #gives N = N(tau)

            
        def w2_of_tau(tau_val):
            N_here = float(tau_to_N(tau_val))
            a_here = spec_params.a_init * np.exp(-N_here)
            H_here = splineH.eval(N_here)
            eps_here = splineeps.eval(N_here)
            return k*k - a_here*a_here * H_here*H_here * (2 - eps_here)
        

    #Before I had been just finding the index this occured at but now I can find the root
    #Use the tau grid, the Q_vals, and Q_of_tau func
        def find_turning_point(tau_grid, Q_vals, w2_of_tau):
            #first need to find where there is a sign change
            for i in range(len(Q_vals)-1): #in length of 0 to Qvals-1
                if Q_vals[i] * Q_vals[i+1] < 0: #finds points where Qi is less than 0 and Qi is greater than 0 and put them in tau_L or tau_R
                    tau_L = tau_grid[i]
                    tau_R = tau_grid[i+1]
                    break
            else:
                raise RuntimeError("No sign change in Q(tau)! No turning point found.")

            #tau star should be between these two points of ti and ti+1
            #brentq takes a function that changes sign and the intervals where Q<0 and Q>0
            tau_star = brentq(w2_of_tau, tau_L, tau_R)

            return tau_star
    

        tau_star = find_turning_point(tau, w2_tau, w2_of_tau)
        print("Q(τ⋆) = ", w2_of_tau(tau_star))
        N_star = float(tau_to_N(tau_star))
        print("N_star",N_star)
        
        

        # 5) Prepare pyoscode inputs on ascending grid
        ts = tau  # ascending
        ws = np.emath.sqrt(w2_tau.astype(np.complex128))  # complex allowed
        gs = np.zeros_like(ts)  # g=0 in conformal time equation

        # 6) BD initial conditions at earliest time (most negative tau)
        # u = exp(-ik tau)/sqrt(2k), u' = -ik u
        tau_i = tau[0]
        if phase:
            u0 = (1.0 / np.emath.sqrt(2.0 * k)) * np.exp(-1j * k * tau_i)
        else:
            u0 = 1.0 / np.emath.sqrt(2.0 * k)

        du0_dtau = -1j * k * u0
        print("u0",u0)
        print("du0_dtau",du0_dtau)

        # If we shifted time: w = tau - tau_star, then d/dw = d/dtau, so same derivative.
        dx0 = du0_dtau

        # 7) Solve with pyoscode
        # Downsample output for plotting speed
        if n_plot is None or n_plot >= len(ts):
            t_eval = ts
        else:
            t_eval = np.linspace(ts[0], ts[-1], n_plot)

        sol = pyoscode.solve(
            ts=ts, ws=ws, gs=gs,
            ti=ts[0], tf=ts[-1],
            x0=u0, dx0=dx0,
            t_eval=t_eval,
            order=order,
            rtol=rtol,
            check_grid=False)
        
#        sol = pyoscode.solve(
#            ts=ts, ws=ws, gs=gs,
#            ti=ts[0], tf=ts[-1],
#            x0=ru_init+imu_init, dx0=dru_init+diu_init,
#            t_eval=t_eval,
#            order=order,
#            rtol=rtol,
#            check_grid=False)

        u_eval = np.array(sol["x_eval"])

        return t_eval, u_eval, tau_star, tau_to_N, Ngrid, N_star


    t_plot, u_plot, tau_star, tau_to_N_spline, Ngrid, N_star = tensorsys_tau_solve(
        Ngrid_ode=Ngrid_ode,
        spec_params=spec_params,
        splineH=spline1,
        splineeps=spline2,
        k=k,
        n_oscode=8000,
        n_plot=15000,
        rtol=1e-6,
        order=3,
        use_shifted_w=True,
        phase=True)
    
    print("t_plot",t_plot)
    print("u_plot[0]",u_plot[0])
    print("u_plot[-1]",u_plot[-1])

    

    plt.figure(figsize=(10,6))
    plt.plot(t_plot, np.real(u_plot), label="Re(u)")
    plt.plot(t_plot, np.imag(u_plot), "--", label="Im(u)")
    plt.plot(t_plot, np.abs(u_plot), label="|u|", color="k", linestyle="--")
    plt.axvline(0, color="gray", linestyle=":")
    plt.grid(True, alpha=0.3)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$u_k$")
    plt.ylim(-1e32,1e32)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    N_of_tauplot = tau_to_N_spline(t_plot) #N of tau
    a_of_tauplot = spec_params.a_init * np.exp(-N_of_tauplot)
    H_of_tauplot = np.array([spline1.eval(float(N)) for N in N_of_tauplot])
    eps_of_tauplot = np.array([spline2.eval(float(N)) for N in N_of_tauplot])

    
    plt.figure(figsize=(10,6))
    plt.plot(t_plot, np.real(u_plot/a_of_tauplot), label="Re(u/a)")
    plt.plot(t_plot, np.imag(u_plot/a_of_tauplot), "--", label="Im(u/a)")
    plt.plot(t_plot, np.abs(u_plot/a_of_tauplot), label="|u/a|", color="k", linestyle="--")
    plt.axvline(0, color="gray", linestyle=":")
    plt.grid(True, alpha=0.3)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\phi=u_k/a$")
#    plt.ylim(-1e32,1e32)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #Okay now that we have everything in tau variables, it is time to take a look in N variables to make sure this looks the way I would expect.
    
#    N_of_tau = N_of_tauplot
#    print("N_of_tau",N_of_tau)
#    u_of_tau = u_plot
#    spline_ut_N = CubicSpline(N_of_tau[::-1], u_of_tau[::-1])
#    ut_N = spline_ut_N(Ngrid)
    spline_a_N = PchipInterpolator(N_of_tauplot[::-1], a_of_tauplot[::-1])
    a_N = spline_a_N(Ngrid)
    
    tau_of_N = PchipInterpolator(N_of_tauplot[::-1], t_plot[::-1])  # N decreasing -> reverse to increasing

    tau_on_Ngrid = tau_of_N(Ngrid)  # tau(Ngrid)
    
    print("tau range from solver:", t_plot.min(), t_plot.max())
    print("tau range from tau_on_Ngrid:", tau_on_Ngrid.min(), tau_on_Ngrid.max())


    # Now interpolate u(τ) on tau grid (monotone, safe)
    u_re_spline = PchipInterpolator(t_plot, np.real(u_plot))
    u_im_spline = PchipInterpolator(t_plot, np.imag(u_plot))

    ut_N = u_re_spline(tau_on_Ngrid) + 1j*u_im_spline(tau_on_Ngrid)
        
    

#================================================================================================================================
    
    #Save the Re(u) and Im(u)
    ut_N_real =  np.real(ut_N)
    ut_N_imag =  np.imag(ut_N)
    
    # Plot the Real parts of u_t(N)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(Ngrid, np.real(ut_N), label=r'$u_{real}$', color='brown', linewidth=1.5)
    axes[0].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[0].set_ylim(-5e31,5e31)
    axes[0].set_xlim(N_star-5,N_star+2)
    axes[0].set_title('Real Part of $u_t(N)$', fontsize=14)
    axes[0].set_xlabel(r'$N$', fontsize=12)
    axes[0].set_ylabel(r'$u_{real}$', fontsize=12)
    axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=10, frameon=False)


    # Plot the Imaginary parts of u_t(N)
    axes[1].plot(Ngrid, np.imag(ut_N), label=r'$u_{imag}$', color='darkkhaki', linewidth=1.5)
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
    

    #Plot phi behavior!
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # Plot 1: phi=|u|/a
    wkb_color = '#DB7093'  # Vibrant rose for WKB approximation
    axes[0].plot(Ngrid, np.abs(ut_N)/a_N, label=r'$\phi = |u|/a$', color=wkb_color, linewidth=1.5)
    axes[0].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[0].set_title(r'Mode Amplitude $\phi=|u_t(N)|/a$', fontsize=14)
    axes[0].set_xlabel(r'$N$', fontsize=12)
    axes[0].set_ylabel(r'$\phi=|u|/a$', fontsize=12)
    axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[0].legend(fontsize=10, frameon=False)


    # Plot 2: u^2/a^2
    amp_color = '#333333'
    axes[1].plot(Ngrid, np.abs(ut_N)**2/a_N**2, label=r'$|u_t^2|/a^2$', color=amp_color, linewidth=1.5)
    axes[1].axvline(x=N_star, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
    axes[1].set_title(r'Amplitude Squared $\phi^2=|u_t(N)^2|/a^2$', fontsize=14)
    axes[1].set_xlabel(r'$N$', fontsize=12)
    axes[1].set_ylabel(r'$\phi^2=|u_t^2|/a^2$', fontsize=12)
    axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
    axes[1].legend(fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.show()

#    #Now it is time to save what we need to save here for our mode and apply it to the tensor power spectrum. We get the real and imaginary parts and create a complex spline to use on a N range outside the tensorsys function.
#    spline_ut_N_norm_real = CubicSpline(Ngrid[::-1], ut_N_real[::-1])
#    spline_ut_N_norm_imag = CubicSpline(Ngrid[::-1], ut_N_imag[::-1])
#
##        
#    def spline_ut_N_norm_complex(N):
#        return spline_ut_N_norm_real(N) + 1j * spline_ut_N_norm_imag(N)
#    

    u_squared = ut_N_real**2 + ut_N_imag**2
    phi_squared = u_squared/(a_N*a_N)
#    spline_phi_squared = CubicSpline(Ngrid[::-1],phi_squared[::-1])
        
    return ut_N, Ngrid, N_star, phi_squared

