import numpy as np
import pygsl.odeiv as odeiv
import pygsl.spline as spline
from pygsl.testing import _ufuncs
from calcpath import *


knos = 1575 # total number of k-values to evaluate
kinos = 214 # total number of k-values to use for integration
k_file = "ks_eval.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks.dat" # file containing k-values for integration
Y = 50 # Y = value of k/aH at which to initialize mode fcns
knorm = 0.05 # normalization scale aka the pivot
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

    abserr1 = 1.e-8 # absolute error tolerance - DO NOT ADJUST THESE VALUES!
    relerr1 = 1.e-8 # relative error tolerance

    abserr2 = 1e-08 #1e-10 # absolute error tolerance
    relerr2 = 1e-08 #1e-10 # relative error tolerance

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

    """
    Find scalar spectra first.
    """

    for m in range(kinos):
        #print(m)
        #print(f"Starting spectrum for mode {m}/{kinos-1}")

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
        
        
        if k/1.73e-61 > Y: # 1.73e-61 is the present inverse Hubble radius (~3.2e-4 hMpc^-1) in Planck units
            while k / (spec_params.a_init*np.exp(-N)*ydoub[1]*(1-ydoub[2])) > Y:
                N += -0.01
                ydoub[1] = spline1.eval(N)
                ydoub[2] = spline2.eval(N)
        #print(f"Finished spectrum for mode {m}/{kinos-1}")
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

            e2 = odeiv.evolve(s2, c2, 2) # mode eqs here?
            
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

        #print(f"About to compute P_s at mode {m}")


        P_s[m] = (k**3./(2.*(np.pi**2.))) * (spline5.eval(Nefolds[count])+imu_s[count]) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])*spline2.eval(Nefolds[count]))/(4*np.pi))
     
        
        #print(f"Finished computing P_s at mode {m}")

        """
        Tensor spectra
        """

        count = 0
        
        N = Nefolds[0]
        realu_init[0] = ru_init
        realu_init[1] = dru_init

        s2 = odeiv.step_rkf45(2, tensorsys, args=spec_params)
        c2 = odeiv.control_y_new(s2, abserr2, relerr2)

        while N > Nfinal:
            realu_t[count] = realu_init[0] * realu_init[0]
            Nefolds[count] = N

            spec_params.H = spline1.eval(N)
            spec_params.eps = spline2.eval(N)

            e2 = odeiv.evolve(s2, c2, 2) #maybe mode eq
            
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
        
        realu_t[count] = realu_init[0] * realu_init[0]
        Nefolds[count] = N

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

        imu_t[count] = imu_init[0] * imu_init[0]
        Nefolds[count] = N
        count -= 1

        #print(f"About to compute P_t at mode {m}")


        P_t[m] = 64. * np.pi * (k**3./(2.*np.pi**2.)) * (spline7.eval(Nefolds[count])+imu_t[count]) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])))

        #print(f"Finished computing P_t at mode {m}")

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

    for i in range(knos):
        u_s[0, i] = ks[i]
        u_s[1, i] = spec_norm * spline6.eval(ks[i]*5.41e-58)

        u_t[0, i] = ks[i]
        u_t[1, i] = spec_norm * spline8.eval(ks[i]*5.41e-58)
    print("Exiting spectrum evaluation normally")
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



      
