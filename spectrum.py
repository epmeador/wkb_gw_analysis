import numpy as np
import pygsl.odeiv as odeiv
import pygsl.spline as spline
from pygsl.testing import _ufuncs
import time
import matplotlib.pyplot as plt

from calcpath import *

knos = 1575 # total number of k-values to evaluate
kinos = 214 # total number of k-values to use for integration
k_file = "ks_eval.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks.dat" # file containing k-values for integration
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
    
    

    # Diagnostic: background quantities
#    plt.figure(figsize=(10,6))
#    plt.subplot(211)
#    plt.plot(Nefolds[:countback+1], H, label="H(N)")
#    plt.ylabel("H")
#    plt.legend()
#
#    plt.subplot(212)
#    plt.plot(Nefolds[:countback+1], eps, label="ε(N)")
#    plt.ylabel("epsilon")
#    plt.legend()
#
#    plt.tight_layout()
#    plt.show()
    
    
    H_reduced = H * np.sqrt(8 * np.pi)

    plt.figure(figsize=(10,6))

    plt.subplot(211)
    plt.plot(Nefolds[:countback+1], H_reduced, label=r"$H(N)$ (reduced units)")
    plt.ylabel(r"$H$  [$M_{\rm Pl,red}=1$]")
    plt.legend()

    plt.subplot(212)
    plt.plot(Nefolds[:countback+1], eps, label=r"$\epsilon(N)$")
    plt.ylabel(r"$\epsilon$")
    plt.legend()

    plt.tight_layout()
    plt.show()


    """
    Find scalar spectra first.
    """
    
    total_time = 0.0
#    N_cross_list = []
#    k_list = []
    for m in range(kinos):
        print("Starting for this mode")
        start_time = time.time()
        start_scalars = time.time()


        print(m)

        k = kis[m] * 5.41e-58 # converts to Planck from hMpc^-1
        kis[m] = k
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

        spec_params.k = k
        nu = (3-spline2.eval(N)) / (2*(1-spline2.eval(N)))
        # print(nu)
        Yeff = k / (spec_params.a_init*(np.exp(-N)*(spline1.eval(N)*(1.-spline2.eval(N)))))
        # print(Yeff)

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

        """
        Solve for real part of u first.
        """
        s2 = odeiv.step_rkf45(2, scalarsys, args=spec_params)
        c2 = odeiv.control_y_new(s2, abserr2, relerr2)

        while N > Nfinal:
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

        P_s[m] = (k**3./(2.*(np.pi**2.))) * ((spline5.eval(Nefolds[count]))+(imu_s[count])) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])*spline2.eval(Nefolds[count]))/(4*np.pi))
        end_scalars = time.time()
        print(f"[DEBUG] scalar mode {m} took {end_scalars - start_scalars:.3f} s")


        """
        Tensor spectra
        """
        start_tensors = time.time()

        count = 0
        
        N = Nefolds[0]
        realu_init[0] = ru_init
        realu_init[1] = dru_init

        s2 = odeiv.step_rkf45(2, tensorsys, args=spec_params)
        c2 = odeiv.control_y_new(s2, abserr2, relerr2)
        
#        cross_found = False
#        N_cross_mode = None
#        

        while N > Nfinal:
#            realu_t[count] = realu_init[0]
            #original
            realu_t[count] = realu_init[0] * realu_init[0]
            Nefolds[count] = N

            spec_params.H = spline1.eval(N)
            spec_params.eps = spline2.eval(N)
            
#            a_here = spec_params.a_init * np.exp(-N)
#            ratio_here = k / (a_here * spec_params.H)
#
#            if (not cross_found) and (ratio_here >= 1.0):
#                cross_found = True
#                N_cross_mode = N
            
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
#        realu_t[count] = realu_init[0]
        #original
        realu_t[count] = realu_init[0] * realu_init[0]
        Nefolds[count] = N
        
#        if N_cross_mode is not None:
#            k_list.append(k)
#            N_cross_list.append(N_cross_mode)


        for j in range(count+1):
            Nordered[j] = Nefolds[count-j]
            uordered_t[j] = realu_t[count-j]

        spline7 = spline.cspline(count+1)
        spline7.init(Nordered[:count+1], uordered_t[:count+1])

        """
        Imaginary part
        """
        count = 0

        N = Nefolds[0]
        imu_init[0] = iu_init
        imu_init[1] = diu_init

        s2 = odeiv.step_rkf45(2, tensorsys, args=spec_params)
        c2 = odeiv.control_y_new(s2, abserr2, relerr2)

        while N > Nfinal:
#            imu_t[count] = imu_init[0] #* imu_init[0]
            #original
            imu_t[count] = imu_init[0] * imu_init[0]
            Nefolds[count] = N

            spec_params.H = spline1.eval(N)
            spec_params.eps = spline2.eval(N)

            e2 = odeiv.evolve(s2, c2, 2) # mode eqs
            
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
        
#        imu_t[count] = imu_init[0]
        #original below
        imu_t[count] = imu_init[0] * imu_init[0]
        Nefolds[count] = N
        count -= 1

        P_t[m] = 64. * np.pi * (k**3./(2.*np.pi**2.)) * ((spline7.eval(Nefolds[count]))+(imu_t[count])) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])))
        
        end_tensors= time.time()
        print(f"[DEBUG] tensor mode {m} took {end_tensors - start_tensors:.3f} s")

        
        
#        # combine real and imaginary u_t for this mode
#        u_t_complex = realu_t[:count+1] + 1j * imu_t[:count+1]
        N_t = Nefolds[:count+1]
        u_real = realu_t[:count+1]
        u_imag = imu_t[:count+1]
                
        print("u_real min/max:", u_real.min(), u_real.max())
        print("u_imag min/max:", u_imag.min(), u_imag.max())
        #This is technically u_real^2 +u_imag^2
        u_t_all = u_real + u_imag
        amp_u_t_all = np.abs(u_t_all)
        a_t = spec_params.a_init * np.exp(-N_t)


#        # make an array with columns: mode index, k, N, Re(u_t), Im(u_t)
#        u_save = np.column_stack([
#            np.full_like(N_t, m, dtype=int),      # mode index
#            np.full_like(N_t, kis[m]),            # k value (same for all rows of this mode)
#            N_t,
#            np.real(u_t_complex),
#            np.imag(u_t_complex)
#        ])
#
#        # append instead of overwrite
#        header = "m k N Re(u_t) Im(u_t)" if m == 0 else ""
#        with open("ut_N_original_N_real_img.dat", "a") as f:
#            np.savetxt(f, u_save, header=header if m == 0 else "", comments='')
#        
        plt.figure(figsize=(5,3))
        plt.plot(N_t, amp_u_t_all, label='|u_t^2|', color='maroon')
#        plt.plot(N_t, u_imag, label='Im(u_t)^2', alpha=0.7)
#        plt.plot(Ngrid, ut_real, label='Re(u_t)')
#        plt.plot(Ngrid, ut_imag, label='Im(u_t)', alpha=0.7)
#        plt.axvline(N_t, color='gray', linestyle='--', label='N_eval (freeze-out)')
        plt.xlabel("N")
        plt.ylabel(r"$|u_t^2 = u_{real}^2 + u_{imag}^2|(N)$")
        plt.title(f"Mode k = {k:.2e}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    
        
        plt.figure(figsize=(5,3))
        plt.plot(N_t, amp_u_t_all/(a_t**2), label='|u_t^2|/a^2 (from tensorsys)', color='slateblue')
#        plt.axvline(N_t[-1], color='gray', linestyle='--', label='N_t (freeze-out)')
        plt.xlabel("N")
        plt.ylabel(r"$u_t^2(N)/a^2$")
        plt.title(f"Mode k = {k:.2e}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        #do a zoom in on the time steps so i can maybe see some oscillatory behavior
        Re_slice = u_real.copy()
        Im_slice = u_imag.copy()
        amp_u_t_all_slice = amp_u_t_all.copy()

        print("Re min/max:", Re_slice.min(), Re_slice.max())
        print("Im min/max:", Im_slice.min(), Im_slice.max())
        #copies the real data

        #only the first ~2000 time steps for a quick zoom so i can zoom in a see the oscillatory behavior. this is the first 2000 points so this isnt event showing me
        zoom_len = min(1225, len(N_t[:count+1]))

#        #zoomed in on first 2000 steps for real and imag parts
#        plt.figure(figsize=(6,3))
#        plt.plot(N_t[:zoom_len], Re_slice[:zoom_len], 'r.-', label='Re u_t (early)')
#        plt.plot(N_t[:zoom_len], Im_slice[:zoom_len], 'b.-', label='Im u_t (early)')
##        plt.xlim(50,65)
##        plt.axvline(N_cross_mode, ls=':', label="horizon exit k=aH")
#        plt.xlabel("N")
#        plt.ylabel("u_t raw")
#        plt.title(f"Zoomed oscillations for first 2000 steps (k={k:.2e})")
#        plt.grid(True, ls='--', alpha=0.5)
#        plt.legend()
#        plt.show()
        
        #zoomed in on first 1225 steps of u/a
        plt.figure(figsize=(5,3))
        plt.plot(N_t[:zoom_len], amp_u_t_all_slice[:zoom_len]/((a_t[:zoom_len])**2), 'r.-', label='u_t^2/a^2 ')
#        plt.xlim(50,65)
#        plt.axvline(N_cross_mode, ls=':', label="horizon exit k=aH")
        plt.xlabel("N")
        plt.ylabel("u_t raw^2/a^2")
        plt.title(f"Zoomed oscillations for first 1225 steps (k={k:.2e})")
        plt.grid(True, ls='--', alpha=0.5)
        plt.legend()
        plt.show()
        
        #zoomed in on first 1225 steps of u
        plt.figure(figsize=(5,3))
        plt.plot(N_t[:zoom_len], amp_u_t_all_slice[:zoom_len], linestyle='-.', color='maroon', label='u_t^2 ')
#        plt.xlim(50,65)
#        plt.axvline(N_cross_mode, ls=':', label="horizon exit k=aH")
        plt.xlabel("N")
        plt.ylabel("u_t raw^2")
        plt.title(f"Zoomed oscillations for first 1225 steps (k={k:.2e})")
        plt.grid(True, ls='--', alpha=0.5)
        plt.legend()
        plt.show()

    
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        print(f"Mode {m+1}/{kinos} (k = {kis[m]:.3e}) took {elapsed:.2f} s")
        print("Finished for this mode")
        print("")



#        if kis[m] == knorm * 5.41e-58: # normalize here
#            pivot_index = np.where(np.isclose(kis, knorm * 5.41e-58, rtol=1e-12))
#            print("pivot_index:", pivot_index)
#            spec_norm = Amp / (P_s[m]+P_t[m])

        #u_t_complex = realu_t[:count+1] + 1j * imu_t[:count+1]
#        N_t = Nefolds[:count+1]
#        np.savetxt("ut_N_original_N_real_img{m:04d}.dat", np.column_stack([Nefolds[count], spline7.eval(Nefolds[count]), imu_t[count]]))
#
#        if np.isclose(kis[m], knorm * 5.41e-58, rtol=0, atol=1e-63):
#            print(f"Normalization triggered at index m={m}, k={kis[m]:.3e}")
#            spec_norm = Amp / (P_s[m]+P_t[m])
            
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

def tensorsys(t, y, parameters):
    dydN = np.empty(2)

    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-(2.-p.eps))*y[0]

    return dydN
