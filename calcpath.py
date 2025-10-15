import numpy as np

from MacroDefinitions import *
from int_de import *

NEQS = 6
kmax = 20000

SMALLNUM = 0.0001
VERYSMALLNUM = 1e-18
LOTSOFEFOLDS = 1000.0

c = 0.0814514 # = 4 (ln(2)+\gamma)-5, \gamma = 0.5772156649


def calcpath(Nefolds, y, path, N, calc):
    retval = "internal_error"
    i = None
    j = None
    k = None
    z = None
    kount = None
    Hnorm = None
    
    #print("NEQS =", NEQS, "| len(y) =", len(y))
    #print("Initial y =", y)

    # Check to make sure we are calculating to sufficient order.
    if NEQS < 6:
        raise Exception("calcpath(): NEQS must be at least 6\n")
        sys.exit()
    
    # Allocate buffers for integration.
    # dydN = derivatives of flow functions wrt N
    # yp = intermediate values for y
    # xp = intermediate values for N
    yp = np.zeros((NEQS, kmax), dtype=float, order='C')
    xp = np.zeros(kmax, dtype=float, order='C')
    
    # First find the end of inflation, when epsilon crosses through unity.
    Nstart = LOTSOFEFOLDS
    Nend = 0.
    
    z, kount = int_de(y, Nstart, Nend, kount, kmax, yp, xp, NEQS, derivs)

    if z:
        retval = "internal_error"
        z = 0
    else:
        # Find when epsilon passes through unity
        i = check_convergence(yp, kount)

        if i == 0:
            #if np.isclose(y[2], 0.5, atol=1e-2) and np.isclose(y[3], -1.0, atol=1e-2):
                #retval = "powerlaw"
            # We never found an end to inflation, so we must be at a late-time attractor
            if y[2] > SMALLNUM or y[3] < 0.:
                # The system did not evolve to a known asymptote
                retval = "noconverge"
            else:
                retval = "asymptote"
        else: # if check_convergence: we have found an end to inflation
            # We found an end to inflation: integrate backwards Nefolds e-folds from that point

            Nstart = xp[i-2] - xp[i-1]
            Nend = Nefolds

            y[:] = yp[:, i-2].copy()

            yp = np.zeros((NEQS, kmax), dtype=float, order='C')
            xp = np.zeros(kmax, dtype=float, order='C')

            z, kount = int_de(y, Nstart, Nend, kount, kmax, yp, xp, NEQS, derivs)

            if z:
                retval = "internal_error"
                z = 0
            elif check_convergence(yp, kount):
                # Not enough inflation.
                retval = "insuff"
            else:
                retval = "nontrivial"

    # Normalize H to give the correct CMB amplitude.  If we are not interested in generating power
    # spectra, normalizing H to give CMB amplitude of 10^-5 at horizon crossing (N = Nefolds) is
    # sufficient

#    if retval == "nontrivial":
#        Hnorm = 0.00001 * 2 * np.pi * np.sqrt(y[2]) / y[1]
#        y[1] = Hnorm * y[1] #this is H
#
#        yp[1, :] = Hnorm * yp[1, :]
#        yp[0, :] = yp[0, :] - y[0] # recenter phi so end of inflation is at 0
#        
 
 #The one that got H(N)
#    if retval == "nontrivial" and SPECTRUM:
#        #end-of-inflation values for rhese variables
#        eps_end = yp[2, kount - 1]   # ε at final step
#        H_end   = yp[1, kount - 1]   # Hubble at final step
#        phi_end = yp[0, kount - 1]   # φ at final step
#
#        # define normalization here
#        # 1e-5 * 2π * sqrt(ε_end) / H_end   → reduced Planck units
#        Hnorm = 5.70227e-6 #5.70343e-6 #0.00001 * 2 * np.pi * np.sqrt(eps_end) / H_end
#
#        # apply the norm here
#        yp[1, :] *= Hnorm #yp[1, :] * normalization_factor_for_H
#
#        # recenteing so phi end is at N=0
#        yp[0, :] -= phi_end #yp[0, :] - phi_at_end_of_inflation
#
#        # keep consistent
#        y[1] *= Hnorm # y[1] * normalization_factor_for_H
#        y[0] -= phi_end #y[0] - phi_at_end_of_inflation
#            

#Trying to get phi(N) and V(N)
    if retval == "nontrivial" and SPECTRUM:
        eps_end = yp[2, kount - 1]
        H_end   = yp[1, kount - 1]
        phi_end = yp[0, kount - 1]

#        # Use the tuned constant that matches Mathematica
        Hnorm = 5.70324e-6

        # Apply consistently
        yp[1, :] *= Hnorm
        y[1]     *= Hnorm

#        Hnorm = 0.00001 * 2 * np.pi * np.sqrt(y[2]) / y[1]
#        y[1] = Hnorm * y[1]
#
#        yp[1, :] = Hnorm * yp[1, :]
#        yp[0, :] = yp[0, :] - y[0]


#        # (meh) recenter φ so φ_end = 0
#        yp[0, :] -= phi_end
#        y[0]     -= phi_end
        
    # Fill in return buffers with path info. Note that the calling
    # function is responsible for freeing these buffers! The
    # buffers are only filled in if non-null pointers are provided.

    if (path is not None) and (N is not None) and (retval != "internal_error") and kount > 1:
#        N.resize(kount, refcheck=False)
#        path.resize(NEQS, kount, refcheck=False)
        N.resize((kount,), refcheck=False)
        path.resize((NEQS, kount), refcheck=False)

        for j in range(kount):
            N[j] = xp[j]

            for i in range(NEQS):
                path[i, j] = yp[i, j] #gets filled with y at time step j

        count = kount
    else:
        count = 0

    calc.npoints = count

    if NEQS > 6:
            print("Final λ₄ max:", np.max(np.abs(path[6])))

    return retval

def derivs(t, y, dydN): #these are flow eq
    dydN = np.zeros(NEQS, dtype=float, order='C')
    
    if y[2] >= 1.0: #this is epsilon
        dydN = np.zeros(NEQS , dtype=float , order='C')
    else:
        if y[2] > VERYSMALLNUM:
            #dydN[0] = - np.sqrt(y[2] / (4 * np.pi))
            #dydN[0] = - np.sqrt(y[2]) #in reduced now i think?
            dydN[0] = -np.sqrt(2.0 * y[2])
        else:
            dydN[0] = 0.0
            
        #print("ratio check:", (dydN[0]**2) / (2.0*y[2]))
        
        dydN[1] = y[1] * y[2] #H*epsilon
        dydN[2] = y[2] * (y[3] + 2.0 * y[2]) #depsilon/dN
        dydN[3] = 2. * y[4] - 5. * y[2] * y[3] - 12. * y[2] * y[2] #dsigma/dN
        
        for i in range(4, NEQS-1):
            dydN[i] = ( 0.5 * (i-3) * y[3] + (i-4) * y[2] ) * y[i] + y[i+1]
            
        dydN[NEQS-1] = ( 0.5 * (NEQS-4) * y[3] + (NEQS-5) * y[2] ) * y[NEQS-1]
#        
#        LAMBDA_MAX = 3  # e.g., evolve only up to λ₃
#        for i in range(4, NEQS):
#            if i - 4 < LAMBDA_MAX:
#                if i+1 < NEQS:
#                    dydN[i] = (0.5 * (i - 3) * y[3] + (i - 4) * y[2]) * y[i] + y[i + 1]
#                else:
#                    dydN[i] = (0.5 * (i - 3) * y[3] + (i - 4) * y[2]) * y[i]
#            else:
#                dydN[i] = 0.0  # freeze higher-order λₗ

#    print("λ₃ =", y[5], "λ₄ =", y[6] if NEQS > 6 else "N/A", "dλ₃ =", dydN[5])
#    if NEQS > 6:
#        print("dλ₄ =", dydN[6])

    return dydN

#def derivs(t, y, dydN):
#    dydN = np.zeros(NEQS, dtype=float, order='C')
#
#    if y[2] >= 1.0:  # ε ≥ 1 → end of inflation
#        return dydN  # no evolution beyond this point
#
#    # φ' = -sqrt(ε / 4π)
#    if y[2] > VERYSMALLNUM:
#        dydN[0] = -np.sqrt(y[2] / (4 * np.pi))
#    else:
#        dydN[0] = 0.0
#
#    # H' = H * ε
#    dydN[1] = y[1] * y[2]
#
#    # ε' = ε (σ + 2ε)
#    dydN[2] = y[2] * (y[3] + 2.0 * y[2])
#
#    # σ' = 2λ₂ - 5εσ - 12ε²
#    dydN[3] = 2. * y[4] - 5. * y[2] * y[3] - 12. * y[2] * y[2]
#
#    # Automatically determine highest λ_H^ℓ to evolve
#    # y[4] = λ₂, y[5] = λ₃, y[6] = λ₄, ...
#    # So: LAMBDA_MAX = NEQS - 4
#    lambda_max_index = NEQS - 1
#
#    # Recurrence: evolve λ_ℓ for ℓ = 2 up to (but not including) λ_{LAMBDA_MAX}
#    for i in range(4, lambda_max_index):
#        dydN[i] = (0.5 * (i - 3) * y[3] + (i - 4) * y[2]) * y[i] + y[i + 1]
#
#    # Last λ_ℓ (highest one): truncate with no λ_{ℓ+1} term
#    if lambda_max_index >= 4:
#        dydN[lambda_max_index] = (
#            0.5 * (lambda_max_index - 3) * y[3]
#            + (lambda_max_index - 4) * y[2]
#        ) * y[lambda_max_index]
#
#    return dydN


def check_convergence(yy, kount):
    for i in range(kount):
        if np.abs(yy[2, i]) >= 1.:
            print("ε crosses 1 at step", i)
            return i
        
    return 0
        
def tsratio(y):
    tsratio = 16 * y[2] * (1.-c*(y[3]+2.*y[2]))

    return tsratio

def specindex(y):
    if SECONDORDER is True:
        specindex = 1. + y[3] - (5.-3.*c)*y[2]*y[2] - 0.25*(3.-5.*c)*y[2]*y[3] + 0.5*(3.-c)*y[4]
    else:
        specindex = (1.0 + y[3]
            - 4.75564*y[2]*y[2]
            - 0.64815*y[2]*y[3]
            + 1.45927*y[4]
            + 7.55258*y[2]*y[2]*y[2]
            + 12.0176*y[2]*y[2]*y[3]
            + 3.12145*y[2]*y[3]*y[3]
            + 0.0725242*y[3]*y[3]*y[3]
            + 5.92913*y[2]*y[4]
            + 0.085369*y[3]*y[4]
            + 0.290072*y[5])

    return specindex

def dspecindex(y):
    ydoub = y.copy()

    dydN = np.zeros(NEQS)
    dydN = derivs(0, ydoub, dydN)

    y = ydoub.copy()

    if SECONDORDER is True:
        dspecindex = - (1./(1 - y[2])*
                     (dydN[3] - 2.*(5.-3.*c)*y[2]*dydN[2]
                     - 0.25 * (3.-5.*c)*(y[2]*dydN[3]+y[3]*dydN[2])
                     + 0.5 * (3.0 - c)*dydN[4]))
    else:
        dspecindex =  - (1./(1 - y[2])*
                    (dydN[3]
                    - 2.0*4.75564*y[2]*dydN[2]
                    - 0.64815*(y[2]*dydN[3] + dydN[2]*y[3])
                    + 1.45927*dydN[4]
                    + 3.0*7.55258*y[2]*y[2]*dydN[2]
                    + 12.0176*(y[2]*y[2]*dydN[3]+2.0*y[2]*dydN[2]*y[3])
                    + 3.12145*(2.0*y[2]*y[3]*dydN[3]+dydN[2]*y[3]*y[3])
                    + 3.0*0.0725242*y[3]*y[3]*dydN[3]
                    + 5.92913*(y[2]*dydN[4]+dydN[2]*y[4])
                    + 0.085369*(y[3]*dydN[4]+dydN[3]*y[4])
                    + 0.290072*dydN[5]))

    return dspecindex






















