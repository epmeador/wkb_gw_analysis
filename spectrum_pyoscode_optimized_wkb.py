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
    Ngrid_original = Nefolds[:countback+1].copy()[::-1]

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
       
        #
        print('N_start',N_start)
        print("Y_start =", Y_of_N(N_start))
        print('N_exit',N_exit)
        print("Y_exit  =", Y_of_N(N_exit))
    
      
  #=========================================================================================================================
        """
        Define global variables first to try and help optimize the code
        """
        
        # 1) DEFINE NGRID
        #One of the most important variables to define is the range of e-folds. We want to make sure this is not uniform. The following Ngrid is based entirely off of three region. The middle region (where the expected turning point lives in the WKB approximation) we expect there to need more points so we define a fine grid, and everywhere else we define a rather coarse grid.
        
        def make_3region_Ngrid(N_hi, N_lo, N_center, delta,
                               dN_coarse=0.05374459791000419,
                               dN_fine=0.001):
            # clamp center and compute region boundaries
            N_center = float(np.clip(N_center, N_lo, N_hi))
            N_top = min(N_hi, N_center + delta)
            N_bot = max(N_lo, N_center - delta)

            def arange_desc(start, stop, step):
                # descending arange that *includes* stop (approximately)
                arr = np.arange(start, stop, -abs(step))
                if arr.size == 0 or arr[-1] > stop:
                    arr = np.append(arr, stop)
                return arr

            # Region A: N_hi -> N_top (coarse)
            A = arange_desc(N_hi, N_top, dN_coarse)

            # Region B: N_top -> N_bot (fine)
            B = arange_desc(N_top, N_bot, dN_fine)

            # Region C: N_bot -> N_lo (coarse)
            C = arange_desc(N_bot, N_lo, dN_coarse)

            # stitch, avoiding duplicated boundary points
            Ngrid = np.concatenate([A, B[1:], C[1:]])

            # enforce strict decreasing (kill any duplicates from float roundoff)
            d = np.diff(Ngrid)
            keep = np.ones_like(Ngrid, dtype=bool)
            keep[1:] = d < 0
            Ngrid = Ngrid[keep]

            return Ngrid
            
            
        N_hi, N_lo = N_start, 0.0496408597 #N_lo is defined from the previous numerical code
        Ngrid_ode = make_3region_Ngrid(N_hi=N_hi, N_lo=N_lo,N_center=N_exit, #could use Nstar
        delta=1.0, # try 1–3 efolds
        dN_coarse=0.05374459791000419,
        dN_fine=0.01)
        Nfinal_num = Ngrid_ode[-1]
        
        
        print("NGRID DIAGNOSTICS")
        print("Ngrid_ode start to end:", Ngrid_ode[0], Ngrid_ode[-1], "len", len(Ngrid_ode))
        print("strictly decreasing?", np.all(np.diff(Ngrid_ode) < 0))
        print("Ngrid new start to end",Ngrid_ode[0], Ngrid_ode[-1])
        print("")
        
        # 2) DEFINE N AT WHICH TO EVALUATE PS
        print("CHOSEN N TO EVALUATE POWER SPECTRUM")
#        N_eval = Nfinal_num + 5*(Ngrid_ode[-2]-Ngrid_ode[-1])
#               # Ensure N_freeze is within the WKB grid
#        if not (Ngrid_ode[-1] <= N_eval <= Ngrid_ode[0]):
#            raise RuntimeError(f"N_eval={N_eval} outside WKB grid "
#                               f"[{Ngrid_wkb[-1]}, {Ngrid_wkb[0]}]")
#        N_eval_tensor = 0.0496408597
#        N_eval_tensor = Nfinal_num + 5*(Ngrid_ode[-2]-Ngrid_ode[-1])
        
#        print("N_eval:",N_eval)
        
        # 3) DEFINE a(N), H(N), and eps(N) of length n_oscode. We really want a large set of values in n_oscode, we can play with this.
        
        n_oscode = 8000
        Ngrid = np.linspace(N_hi, N_lo, n_oscode)  # descending if N_hi>N_lo
        
#        N_eval_tensor = Ngrid[-1] + 5*(Ngrid[-2] - Ngrid[-1])
        N_eval_tensor = 0.0496408597
               # Ensure N_freeze is within the WKB grid
#        if not (Ngrid_ode[-1] <= N_eval <= Ngrid_ode[0]):
#            raise RuntimeError(f"N_eval={N_eval} outside WKB grid "
#                               f"[{Ngrid_wkb[-1]}, {Ngrid_wkb[0]}]")
        N_eval_scalar = 0.010524043109012227
        
        # Define background functions in convention a(N)=a_init e^{-N}
        aN = spec_params.a_init * np.exp(-Ngrid) #a(N)
        HN = np.array([spline1.eval(float(N)) for N in Ngrid]) #H(N)
        epsN = np.array([spline2.eval(float(N)) for N in Ngrid]) #eps(N)
        
        # 4) BUILD CONFORMAL TIME FROM THIS NGRID
        #d tau / dN = -1/(aH) for N = e-folds remaining
        integrand = 1.0 / (aN * HN)  # positive
        # integrate from late to early so tau is negative at early times
        tau = cumulative_trapezoid(-integrand[::-1], Ngrid[::-1], initial=0.0)[::-1] #should go from very negative to 0 if inflation
        ## IMP: Defines our spline below of tau(N)
        tau_to_N = PchipInterpolator(tau, Ngrid) #gives N = N(tau)
        
        print("tau[0], tau[-1] =", tau[0], tau[-1])
        print("tau strictly increasing?", np.all(np.diff(tau) > 0))
        print("min dtau =", np.min(np.diff(tau)))
                
        # 5) EVALUATE BACKGROUND VARIABLES IN CONFORMAL TIME
        
        #N(tau)
        N_of_tau = tau_to_N(tau)
        #a(tau)
        a_tau = spec_params.a_init * np.exp(-N_of_tau)
        #Define in increasing direction to create spline
        N_inc   = Ngrid[::-1]      # increasing
        H_inc   = HN[::-1]
        eps_inc = epsN[::-1]
        a_inc = aN[::-1]
        
        H_of_N   = PchipInterpolator(N_inc, H_inc)
        eps_of_N = PchipInterpolator(N_inc, eps_inc)
#        a_of_N = PchipInterpolator(N_inc, a_inc)
        
        H_tau   = H_of_N(N_of_tau)
        eps_tau = eps_of_N(N_of_tau)
#        a_tau = a_of_N(N_of_tau)
        
        
        # 6) DEFINE W2 FOR SCALAR AND TENSOR MODES
        #With the background variables define we can now define two separate functions, one for scalar modes and a separate for tensor modes
        
        #For scalar modes
        aH_tau = a_tau * H_tau
        app_over_a = (a_tau*a_tau)*(H_tau*H_tau)*(2.0 - eps_tau)
#        L_tau = np.log(eps_tau)
#        L_spl = PchipInterpolator(tau, L_tau)
#        L1 = L_spl.derivative(1)(tau)
#        L2 = L_spl.derivative(2)(tau)
#        zpp_over_z = app_over_a + 0.5*L2 + aH_tau*L1 + 0.25*(L1*L1)
#        w2_scalar = k*k - zpp_over_z
#        
        
        
        
        # Also interpolate sigma(N), xi(N) onto tau
        sigN = np.array([spline3.eval(float(N)) for N in Ngrid])  # sigma on Ngrid
        xiN  = np.array([spline4.eval(float(N)) for N in Ngrid])  # xi on Ngrid

        sig_inc = sigN[::-1]
        xi_inc  = xiN[::-1]

        sig_of_N = PchipInterpolator(N_inc, sig_inc)
        xi_of_N  = PchipInterpolator(N_inc, xi_inc)

        sig_tau = sig_of_N(N_of_tau)
        xi_tau  = xi_of_N(N_of_tau)

        aH_tau = a_tau * H_tau

        # Tensor (unchanged)
        ztt_over_ztt = (aH_tau**2) * (2.0 - eps_tau)
        w2_tensor = k*k - ztt_over_ztt
                
        #For tensor modes
        w2_tensor = k*k - (a_tau*a_tau)*(H_tau*H_tau)*(2.0 - eps_tau)
        
        print("")
        print("w2 for scalar modes and tensor modes")
        print("w2_scalar from index 0 to -1:",w2_scalar[0],w2_scalar[-1])
        print("w2_tensor from index 0 to -1:",w2_tensor[0],w2_tensor[-1])
        
        
        # 7) DEFINE FUNCTIONS FOR SCALARS AND TENSORS WHICH WILL USE PYOSCODE
#        def scalarsys(tau_ascending, w2_for_scalars, rtol=1e-4, order=3, phase=True):
#
#            #Define variables needed for pyoscode
#            ts = tau_ascending  #ascending
#            ws = np.emath.sqrt(w2_for_scalars.astype(np.complex128))  # complex allowed
#            gs = np.zeros_like(ts)  # g=0 in conformal time equation
#
#            #BD initial conditions at earliest time (most negative tau)
#            tau_i = ts[0]
#            
#            if phase:
#                u0 = (1.0 / np.emath.sqrt(2.0 * k)) * np.exp(1j * k * tau_i)
#            else:
#                u0 = 1.0 / np.emath.sqrt(2.0 * k)
#
#            du0_dtau = -1j * k * u0
#            
#            #Define t_eval range
#            tau_scalar_evaluated = np.linspace(ts[0], ts[-1], n_oscode)
#            t_eval = tau_scalar_evaluated
#            
#            # Solve with pyoscode
#
#            sol = pyoscode.solve(
#                ts=ts, ws=ws, gs=gs,
#                ti=ts[0], tf=ts[-1],
#                x0=u0, dx0=du0_dtau,
#                t_eval=t_eval,
#                order=order,
#                rtol=rtol,
#                check_grid=False)
#         
#
#            u_scalar_evaluated = np.array(sol["x_eval"]) #gives u(tau) for tensor modes
#
#            return tau_scalar_evaluated, u_scalar_evaluated



        
#        def tensorsys(tau_ascending, w2_for_tensors, rtol=1e-4, order=3, phase=True):
#        
#            #Define variables needed for pyoscode
#            ts = tau_ascending  #ascending
#            ws = np.emath.sqrt(w2_for_tensors.astype(np.complex128))  # complex allowed
#            gs = np.zeros_like(ts)  # g=0 in conformal time equation
#
#            #BD initial conditions at earliest time (most negative tau)
#            tau_i = ts[0]
#            
#            if phase:
#                u0 = (1.0 / np.emath.sqrt(2.0 * k)) * np.exp(1j * k * tau_i)
#            else:
#                u0 = 1.0 / np.emath.sqrt(2.0 * k)
#
#            du0_dtau = -1j * k * u0
#            
#            #Define t_eval range
#            tau_tensor_evaluated = np.linspace(ts[0], ts[-1], n_oscode)
#            t_eval = tau_tensor_evaluated
#            
#            # Solve with pyoscode
#
#            sol = pyoscode.solve(
#                ts=ts, ws=ws, gs=gs,
#                ti=ts[0], tf=ts[-1],
#                x0=u0, dx0=du0_dtau,
#                t_eval=t_eval,
#                order=order,
#                rtol=rtol,
#                check_grid=False)
#         
#
#            u_tensor_evaluated = np.array(sol["x_eval"]) #gives u(tau) for tensor modes
#
#            return tau_tensor_evaluated, u_tensor_evaluated

        



        """
        Solve for real part of u first.
        """
        start_time = time.time()
        start_scalars = time.time()

        # 1) MODE SOLN IN CONFORMAL TIME
        tau_scalar, v_tau = scalarsys(k, n_oscode, tau, w2_scalar, rtol=1e-4, order=3, phase=True)
        
        #N(tau)
        N_of_tau_scalars = tau_to_N(tau_scalar)
        #a(tau)
        a_of_tau_scalars = spec_params.a_init * np.exp(-N_of_tau_scalars)
        #eps(tau)
        eps_of_tau_scalars = eps_of_N(N_of_tau_scalars)
#        #Using reduced epsilon
#        eps_of_tau_scalars = eps_of_tau_scalars / (8*np.pi)
        
        #z(tau)
#        z_of_tau = a_of_tau_scalars * np.sqrt((2*eps_of_tau_scalars))
        z_of_tau = a_of_tau_scalars * np.sqrt(eps_of_tau_scalars/(4*np.pi))

                
        # 2) SWITCH VARIABLES TO E-FOLDS
        #a(N)
        spline_a_N = PchipInterpolator(N_of_tau_scalars[::-1], a_of_tau_scalars[::-1])
        a_N = spline_a_N(Ngrid)
        
        #z(N)
        spline_z_N = PchipInterpolator(N_of_tau_scalars[::-1], z_of_tau[::-1])
        z_N = spline_z_N(Ngrid)
        
    
        # tau(Ngrid)
        tau_of_N_scalars = PchipInterpolator(N_of_tau_scalars[::-1], tau_scalar[::-1])  # N decreasing -> reverse to increasing
        tau_on_Ngrid_scalars = tau_of_N_scalars(Ngrid)  # tau(Ngrid)
        
        
        # Now interpolate u(tau) on tau grid (monotone, safe)
        vs_re_spline = PchipInterpolator(tau_scalar, np.real(v_tau))
        vs_im_spline = PchipInterpolator(tau_scalar, np.imag(v_tau))
        
        vs_N = vs_re_spline(tau_on_Ngrid_scalars) + 1j*vs_im_spline(tau_on_Ngrid_scalars)
        
        
        
#=========================================================================================================================

        #Real and imaginary parts of ut_N
        
        #Save the Re(u) and Im(u)
        vs_N_real =  np.real(vs_N)
        vs_N_imag =  np.imag(vs_N)
        
        # Plot the Real parts of u_t(N)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].plot(Ngrid, vs_N_real, label=r'$v_{real}$', color='brown', linewidth=1.5)
        axes[0].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[0].set_ylim(-5e31,5e31)
        axes[0].set_xlim(N_exit-5,N_exit+2)
        axes[0].set_title('Real Part of $v_s(N)$', fontsize=14)
        axes[0].set_xlabel(r'$N$', fontsize=12)
        axes[0].set_ylabel(r'$v_{real}$', fontsize=12)
        axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[0].legend(fontsize=10, frameon=False)
        
        # Plot the Imaginary parts of u_t(N)
        axes[1].plot(Ngrid, vs_N_imag, label=r'$v_{imag}$', color='darkkhaki', linewidth=1.5)
        axes[1].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[1].set_ylim(-5e31,5e31)
        axes[1].set_xlim(N_exit-5,N_exit+2)
        axes[1].set_title('Imaginary Part of $v_s(N)$', fontsize=14)
        axes[1].set_xlabel(r'$N$', fontsize=12)
        axes[1].set_ylabel(r'$v_{imag}$', fontsize=12)
        axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[1].legend(fontsize=10, frameon=False)
        
        plt.tight_layout()
        plt.show()


    #Mode amplitude and amplitude squared
    
        #Plot phi behavior!
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        # Plot 1: R=|v|/z
        wkb_color = '#DB7093'  # Vibrant rose for WKB approximation
        axes[0].plot(Ngrid, np.abs(vs_N_real+1j*vs_N_imag)/z_N, label=r'R = |v|/z', color=wkb_color, linewidth=1.5)
        axes[0].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[0].set_title(r'Mode Amplitude $R=|v_s(N)|/z$', fontsize=14)
        axes[0].set_xlabel(r'$N$', fontsize=12)
        axes[0].set_ylabel(r'R=|v|/z', fontsize=12)
        axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[0].legend(fontsize=10, frameon=False)

        # Plot 2: v^2/z^2
        amp_color = '#333333'
        axes[1].plot(Ngrid, (np.abs(vs_N_real+1j*vs_N_imag)*np.abs(vs_N_real+1j*vs_N_imag))/(z_N*z_N), label=r'$|{v_s}^2|/z^2$', color=amp_color, linewidth=1.5)
        axes[1].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[1].set_title(r'Amplitude Squared $R^2=|v_s(N)^2|/z^2$', fontsize=14)
        axes[1].set_xlabel(r'$N$', fontsize=12)
        axes[1].set_ylabel(r'$R^2=|v_s^2|/z^2$', fontsize=12)
        axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[1].legend(fontsize=10, frameon=False)
        
        plt.tight_layout()
        plt.show()


#=========================================================================================================================

        #EVALUATE SCALAR POWER SPECTRUM
        # Define v(N) and R(N)
        vs_squared = (vs_N_real*vs_N_real) + (vs_N_imag*vs_N_imag)
        R_squared = vs_squared/(z_N*z_N)
        
        #Define spline for R(N) over Ngrid_ode
        R_squared_spl = CubicSpline(Ngrid[::-1], R_squared[::-1])
        
        #Evaluate at N_eval
        R_squared_eval = float(R_squared_spl(N_eval_scalar))
        
        ## EVALUATE P_s
        P_s[m] = (k**3/(2*np.pi**2)) * (R_squared_eval)

        
        end_scalars = time.time()
        
        print(f"[DEBUG] scalar mode {m} took {end_scalars - start_scalars:.3f} s")
        
    
#=========================================================================================================================

        #We continue the story with the tensor power spectrum, which is not based on the numerical code anymore. Instead we use a WKB approximation to find this.

        
#=========================================================================================================================

        """
        Tensor spectra
        """

        start_tensors = time.time()
        print('Stats from actual mode evaluation part:')

        # 1) MODE SOLN IN CONFORMAL TIME
#        tau_tensor, u_tau = tensorsys(k,n_oscode, tau, w2_tensor, rtol=1e-4, order=3, phase=True)
        
        #Test
        ut_N, Ngrid_check, phi_squared_check = tensorsys(Ngrid_ode, spec_params, spline1, spline2, k, n_oscode=8000, n_plot=15000, rtol=1e-4, order=3,phase=True)
 
#        #N(tau)
#        N_of_tau_tensors = tau_to_N(tau_tensor)
#        #a(tau)
#        a_of_tau_tensors = spec_params.a_init * np.exp(-N_of_tau_tensors)
#        
#        # 2) SWITCH VARIABLES TO E-FOLDS
#        spline_a_N = PchipInterpolator(N_of_tau_tensors[::-1], a_of_tau_tensors[::-1])
#        a_N = spline_a_N(Ngrid)
#
#        # tau(Ngrid)
#        tau_of_N_tensors = PchipInterpolator(N_of_tau_tensors[::-1], tau_tensor[::-1])  # N decreasing -> reverse to increasing
#        tau_on_Ngrid_tensors = tau_of_N_tensors(Ngrid)  # tau(Ngrid)
#        
#        # Now interpolate u(tau) on tau grid (monotone, safe)
#        ut_re_spline = PchipInterpolator(tau_tensor, np.real(u_tau))
#        ut_im_spline = PchipInterpolator(tau_tensor, np.imag(u_tau))
#        
#        ut_N = ut_re_spline(tau_on_Ngrid_tensors) + 1j*ut_im_spline(tau_on_Ngrid_tensors)
#        
#    

        
#=========================================================================================================================


        ## PLOT:
        #Real and imaginary parts of ut_N
        
        #Save the Re(u) and Im(u)
        ut_N_real =  np.real(ut_N)
        ut_N_imag =  np.imag(ut_N)
        
        # Plot the Real parts of u_t(N)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].plot(Ngrid_check, ut_N_real, label=r'$u_{real}$', color='brown', linewidth=1.5)
        axes[0].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[0].set_ylim(-5e31,5e31)
        axes[0].set_xlim(N_exit-5,N_exit+2)
        axes[0].set_title('Real Part of $u_t(N)$', fontsize=14)
        axes[0].set_xlabel(r'$N$', fontsize=12)
        axes[0].set_ylabel(r'$u_{real}$', fontsize=12)
        axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[0].legend(fontsize=10, frameon=False)


        # Plot the Imaginary parts of u_t(N)
        axes[1].plot(Ngrid_check, ut_N_imag, label=r'$u_{imag}$', color='darkkhaki', linewidth=1.5)
        axes[1].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[1].set_ylim(-5e31,5e31)
        axes[1].set_xlim(N_exit-5,N_exit+2)
        axes[1].set_title('Imaginary Part of $u_t(N)$', fontsize=14)
        axes[1].set_xlabel(r'$N$', fontsize=12)
        axes[1].set_ylabel(r'$u_{imag}$', fontsize=12)
        axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[1].legend(fontsize=10, frameon=False)
        
        plt.tight_layout()
        plt.show()
        
        #Mode amplitude and amplitude squared
    
        #Plot phi behavior!
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        # Plot 1: phi=|u|/a
        wkb_color = '#DB7093'  # Vibrant rose for WKB approximation
        axes[0].plot(Ngrid_check, np.abs(ut_N_real+1j*ut_N_imag)/a_N, label=r'$\phi = |u|/a$', color=wkb_color, linewidth=1.5)
        axes[0].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[0].set_title(r'Mode Amplitude $\phi=|u_t(N)|/a$', fontsize=14)
        axes[0].set_xlabel(r'$N$', fontsize=12)
        axes[0].set_ylabel(r'$\phi=|u|/a$', fontsize=12)
        axes[0].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[0].legend(fontsize=10, frameon=False)


        # Plot 2: u^2/a^2
        amp_color = '#333333'
        axes[1].plot(Ngrid_check, (np.abs(ut_N_real+1j*ut_N_imag)*np.abs(ut_N_real+1j*ut_N_imag))/(a_N*a_N), label=r'$|u_t^2|/a^2$', color=amp_color, linewidth=1.5)
        axes[1].axvline(x=N_exit, linestyle='--', color='gray', linewidth=1.2, label='Horizon Exit')
        axes[1].set_title(r'Amplitude Squared $\phi^2=|u_t(N)^2|/a^2$', fontsize=14)
        axes[1].set_xlabel(r'$N$', fontsize=12)
        axes[1].set_ylabel(r'$\phi^2=|u_t^2|/a^2$', fontsize=12)
        axes[1].grid(color='lightgrey', linestyle=':', linewidth=0.8)
        axes[1].legend(fontsize=10, frameon=False)
        
        plt.tight_layout()
        plt.show()
        
    
#=========================================================================================================================

        #EVALUATE TENSOR POWER SPECTRUM
        # Define u(N) and phi(N)
        ut_squared = (ut_N_real*ut_N_real) + (ut_N_imag*ut_N_imag)
        phi_squared = ut_squared/(a_N*a_N)
        
        #Define spline for phi(N) over Ngrid_ode
#        phi_squared_spl = CubicSpline(Ngrid[::-1], phi_squared[::-1])
        
        #Test
        phi_squared_spl = CubicSpline(Ngrid_check[::-1], phi_squared_check[::-1])

    
        #Evaluate at N_eval
        phi_squared_eval = float(phi_squared_spl(N_eval_tensor))
        
        print("a_eval =", float(spec_params.a_init*np.exp(-N_eval_tensor)))
        print("phi2_eval =", float(phi_squared_eval))
            
        ## EVALUATE P_t
        P_t[m] = 64*np.pi * (k**3/(2*np.pi**2)) * phi_squared_eval
        
        print("-------------------------------------------------------------------")
        print("tau monotone:", np.all(np.diff(tau) > 0),
      "min dtau:", np.min(np.diff(tau)),
      "num dtau<=0:", np.sum(np.diff(tau) <= 0))


        #Counts time taken
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        print(f"Mode {m+1}/{kinos} (k = {kis[m]:.3e}) took {elapsed:.2f} s")
        print("Finished for this mode")
        print("")


#=========================================================================================================================

        #DIAGNOSTICS FOR COMPARISON PURPOSES
        
        #I can evaluate variables over the numerical N using a spline created in tensorsys for ut
        ut_N_spline = CubicSpline(Ngrid_check[::-1], ut_N[::-1])

        u_t_ode_length = ut_N_spline(Ngrid_ode) #Make sure it is the same length as the original numerical one
        u_wkb_real = np.real(u_t_ode_length)
        u_wkb_imag = np.imag(u_t_ode_length)
        u_wkb_squared = (u_wkb_real*u_wkb_real) +  (u_wkb_imag*u_wkb_imag) #u^2
        u_wkb = np.abs(u_wkb_real+1j*u_wkb_imag)
        a_ode = spec_params.a_init * np.exp(-Ngrid_ode) #a(N)
        u_wkb_squared_over_a2 = u_wkb_squared / (a_ode*a_ode)
        
        vs_N_spline = CubicSpline(Ngrid[::-1], vs_N[::-1])
        v_s_ode_length = vs_N_spline(Ngrid_ode)
        v_wkb_real = np.real(v_s_ode_length)
        v_wkb_imag = np.imag(v_s_ode_length)
        z_N_spline = CubicSpline(Ngrid[::-1], z_N[::-1])
        z_ode_length = z_N_spline(Ngrid_ode)


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
                a_ode,
                v_wkb_real,
                v_wkb_imag,
                z_ode_length
            ]),
            header="N   |u_wkb(N)|  u_wkb_real(N)  u_wkb_imag(N)   u_wkb^2(N)   |u_wkb|^2/a^2    N_hc  N_exit  a(N)  v_wkb_real(N)  v_wkb_imag(N)  z(N)"
        )
        
        print(f"[WKB] saved diagnostics → {outfile}")
    

#=========================================================================================================================

    #Normalize according to the value of k at the characteristic length scale
        k_pivot = knorm * 5.41e-58
        pivot_index = np.argmin(np.abs(kis - k_pivot)) #can also make it an integer to be explicit

        spec_norm = Amp / (P_s[pivot_index] + P_t[pivot_index])
        y[1] = np.sqrt(spec_norm)  # kept this behavior from original code (normalize H for later recon)
        
        print("RAW pivot Ps:", P_s[pivot_index])
        print("RAW pivot Pt:", P_t[pivot_index])


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
    
    
#    
#def tensorsys(k, n_size, tau_ascending, w2_for_tensors, rtol=1e-4, order=3, phase=True):
#
#    #Define variables needed for pyoscode
#    ts = tau_ascending  #ascending
#    ws = np.emath.sqrt(w2_for_tensors.astype(np.complex128))  # complex allowed
#    gs = np.zeros_like(ts)  # g=0 in conformal time equation
#
#    #BD initial conditions at earliest time (most negative tau)
#    tau_i = ts[0]
#    
#    if phase:
#        u0 = (1.0 / np.emath.sqrt(2.0 * k)) * np.exp(1j * k * tau_i)
#    else:
#        u0 = 1.0 / np.emath.sqrt(2.0 * k)
#
#    du0_dtau = -1j * k * u0
#    
#    #Define t_eval range
##    tau_tensor_evaluated = np.linspace(ts[0], ts[-1], n_size)
#    t_eval = ts
#    
#    # Solve with pyoscode
#
#    sol = pyoscode.solve(
#        ts=ts, ws=ws, gs=gs,
#        ti=ts[0], tf=ts[-1],
#        x0=u0, dx0=du0_dtau,
#        t_eval=t_eval,
#        order=order,
#        rtol=rtol,
#        check_grid=False)
# 
#
#    u_tensor_evaluated = np.array(sol["x_eval"]) #gives u(tau) for tensor modes
#
#    return t_eval, u_tensor_evaluated
#        
#        
def scalarsys(k,n_size, tau_ascending, w2_for_scalars, rtol=1e-4, order=3, phase=True):

    #Define variables needed for pyoscode
    ts = tau_ascending  #ascending
    ws = np.emath.sqrt(w2_for_scalars.astype(np.complex128))  # complex allowed
    gs = np.zeros_like(ts)  # g=0 in conformal time equation

    #BD initial conditions at earliest time (most negative tau)
    tau_i = ts[0]
    
    if phase:
        u0 = (1.0 / np.emath.sqrt(2.0 * k)) * np.exp(1j * k * tau_i)
    else:
        u0 = 1.0 / np.emath.sqrt(2.0 * k)

    du0_dtau = -1j * k * u0
    
    #Define t_eval range
    t_eval = ts
    
    # Solve with pyoscode

    sol = pyoscode.solve(
        ts=ts, ws=ws, gs=gs,
        ti=ts[0], tf=ts[-1],
        x0=u0, dx0=du0_dtau,
        t_eval=t_eval,
        order=order,
        rtol=rtol,
        check_grid=False)
 

    u_scalar_evaluated = np.array(sol["x_eval"]) #gives u(tau) for tensor modes

    return t_eval, u_scalar_evaluated
    
def tensorsys(
        Ngrid_to_use, spec_params, splineH, splineeps, k,
        n_oscode, n_plot, rtol=1e-4, order=3,phase=True):
        
    def tensorsys_tau_solve(Ngrid_to_use, spec_params, splineH, splineeps, k, n_oscode, n_plot, rtol=1e-4, order=3,phase=True):

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
    #        H_tau = np.array([splineH.eval(float(N)) for N in N_of_tau])
    #        eps_tau = np.array([splineeps.eval(float(N)) for N in N_of_tau])
    #        w2_tau = k*k - (a_tau*a_tau)*(H_tau*H_tau)*(2.0 - eps_tau)
        
        print("tau[-1] =", tau[-1])
        print("N_of_tau[-1] =", N_of_tau[-1])
        print("Ngrid min/max =", Ngrid.min(), Ngrid.max())

        print("a_tau[-1] finite?", np.isfinite(a_tau[-1]), a_tau[-1])
        
        
        N_inc   = Ngrid[::-1]      # increasing
        H_inc   = HN[::-1]
        eps_inc = epsN[::-1]

        H_of_N   = PchipInterpolator(N_inc, H_inc)
        eps_of_N = PchipInterpolator(N_inc, eps_inc)

        H_tau   = H_of_N(N_of_tau)
        eps_tau = eps_of_N(N_of_tau)
        print("H_tau[-1] finite?", np.isfinite(H_tau[-1]), H_tau[-1])
        print("eps_tau[-1] finite?", np.isfinite(eps_tau[-1]), eps_tau[-1])
        
        w2_tau = k*k - (a_tau*a_tau)*(H_tau*H_tau)*(2.0 - eps_tau)



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

        return t_eval, u_eval, tau_to_N, Ngrid
        #End of tensorsys_tau_solve function
        
    #=========================================================================================================================

    ## RUN THE TENSOR SYS FOR WKB BASED SOLUTION IN CONFORMAL TIME
    tau_plot, u_plot, tau_to_N_spline, Ngrid = tensorsys_tau_solve(
        Ngrid_to_use=Ngrid_to_use,
        spec_params=spec_params,
        splineH=splineH,
        splineeps=splineeps,
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

    ## DEFINE PHI AND PHI SQUARED TO BE USED IN

    #    #Now it is time to save what we need to save here for our mode and apply it to the tensor power spectrum. We get the real and imaginary parts and square them
    ut_N_real =  np.real(ut_N)
    ut_N_imag =  np.imag(ut_N)
    u_squared = (ut_N_real*ut_N_real) + (ut_N_imag*ut_N_imag)
    phi_squared = u_squared/(a_N*a_N)
        
    return ut_N, Ngrid, phi_squared
