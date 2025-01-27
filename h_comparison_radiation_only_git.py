#!/usr/bin/env python
# coding: utf-8

# # Comparing h in Radiation Only Case 
# 
# (MATTER CASE WILL BE CONSIDERED FURTHER IN LATER NOTEBOOK)
# 
# The goal of this notebook is to be able to confirm the WKB approximation method. The way we can do that in this notebook is to go ahead and apply the WKB approximation method that I have developed to a radiation and matter only case.
# 
# According to Pritchard and Kaminiokowski (https://arxiv.org/pdf/astro-ph/0412581), 
# 
#     h_rad(tau) = h(0) * j0(k*tau)/k*tau = sin(k*tau)/k*tau
#     
#     h_matter(tau) = 3 * h(0) * (j1(k*tau)/(k*tau))
#     
#     
# In order to have an accurate comparison I need to be able to make sure the units I have are in fact accurate. To solve the gravitational wave, in conformal time I have the following:
# 
#     u'' + (k^2 - a''/a)u = 0
#     Q = a''/a - k^2
#     
# Keep in mind that, 
# 
#     a''/a = (adot)^2 + addot * a ->
#     = (4 * pi * G/3)* a^2(t) * (rho - 3 * P)
# 
# 
# In this attempt I want to first define my spline and it's first and second derivatives. I want to solve for u(x) using on the splines that I know are correct.

# In[1]:


import numpy as np
from numpy import sqrt, sin, cos, pi
import matplotlib.pyplot as plt
from scipy import special
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
from scipy.misc import derivative
from scipy.special import airy, spherical_jn
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline, splrep, splev
from scipy.integrate import cumtrapz
from scipy.optimize import root_scalar
from scipy.optimize import newton


# Cosmological parameters
H0 = 70.0  # Hubble constant in km/s/Mpc
H0_s = H0 / (3.086e19)  # Convert H0 to s^-1
H0_s_m = H0_s * (1/(3e8)) #H0(1/s)*1/(s/m) to be m^-1

omega_m = 0.3          # Matter density parameter
omega_lambda = 0.7     # Cosmological constant density parameter
omega_r = 2.9e-4       # Radiation density parameter
omega_all = 1 #omega_m + omega_lambda + omega_r
omega_k = 1 - omega_all  # Curvature density parameter

a_min = 1e-2
a_mr_eq = 2.9*1e-4 #a at matter-radiation equality
a_ml_eq = 0.77
a_max = 1


# # PHASE I : GETTING THE RIGHT SPLINE (S)
# 
# a(t)
# 
# a($\tau$)
# 
# a($k\tau$)
# 
# 
# ## Define a(t) Spline
# 
# Consider using updated: make_splrep for spline creation instead of UnivariateSpline

# In[2]:


# Define the Friedmann equation for radiation-dominated era
def H_radiation(a):
    """Hubble parameter during the radiation-dominated era."""
    return H0_s * np.sqrt((omega_r / a**4) + (omega_k / a**2))

# Define scale factor range
a_min = 1e-5
a_max = 1  # Up to a = 1
a_range_rad = np.logspace(np.log10(a_min), np.log10(a_max), num=500)

# Compute t(a) for the radiation-dominated era
t_a_rad = np.array([
    quad(lambda a: 1 / H_radiation(a), a_min, a, limit=1000)[0]
    for a in a_range_rad
])

# Create a spline for a(t)
spline_a_t_rad = UnivariateSpline(t_a_rad, a_range_rad, s=0)

# Define time range for evaluating the spline
t_range_rad = np.linspace(t_a_rad.min(), t_a_rad.max(), len(a_range_rad))
a_t_rad = spline_a_t_rad(t_range_rad)

# Plot: Original Data Points and Spline-Evaluated Curve
plt.figure(figsize=(12, 6))
expected_a_t = np.sqrt(t_range_rad / t_range_rad.max()) * a_max  # Normalize to match the range of a_max
plt.plot(t_range_rad, expected_a_t, label="Expected $a(t) \propto \sqrt{t}$", linestyle="--", color="green")
plt.scatter(t_a_rad, a_range_rad, label="Original Data Points", color="blue", s=10, alpha=0.5)
plt.plot(t_range_rad, a_t_rad, label="Spline-Evaluated Curve", color="red", linestyle="-", linewidth=2)
plt.title("Spline Interpolation of $a(t)$ in Radiation-Dominated Era")
plt.xlabel("Cosmic Time $t$ (s)")
plt.ylabel("Scale Factor $a$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()

# Validation: Compare Original vs Spline-Evaluated a(t)
print("Validation: Original vs Spline-Evaluated Values")
test_t_indices = [10, 100, 300]  # Select indices for testing
for idx in test_t_indices:
    t = t_a_rad[idx]
    original_a = a_range_rad[idx]
    spline_a = spline_a_t_rad(t)
    print(f"t = {t:.2e} s: Original a = {original_a:.5e}, Spline a = {spline_a:.5e}")

# Optional: Logarithmic scaling for the plot
plt.figure(figsize=(12, 6))
plt.scatter(t_a_rad, a_range_rad, label="Original Data Points", color="blue", s=10, alpha=0.5)
plt.plot(t_range_rad, a_t_rad, label="Spline-Evaluated Curve", color="red", linestyle="-", linewidth=2)
plt.title("Log-Log Plot of $a(t)$ in Radiation-Dominated Era")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Cosmic Time $t$ (s)")
plt.ylabel("Scale Factor $a$")
plt.legend(loc="best")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()


# In[3]:


# First derivative of the spline: da/dt
spline_a_t_rad_derivative_1 = spline_a_t_rad.derivative(n=1)
a_t_rad_derivative_1 = spline_a_t_rad_derivative_1(t_range_rad)

# Second derivative of the spline: d^2a/dt^2
spline_a_t_rad_derivative_2 = spline_a_t_rad.derivative(n=2)
a_t_rad_derivative_2 = spline_a_t_rad_derivative_2(t_range_rad)

# Plotting the derivatives
plt.figure(figsize=(12, 6))
plt.plot(t_range_rad, a_t_rad, label="$a(t)$ (Spline)", color="red", linewidth=2)
plt.plot(t_range_rad, a_t_rad_derivative_1, label="$\\frac{da}{dt}$ (First Derivative)", color="blue", linestyle="--")
plt.plot(t_range_rad, a_t_rad_derivative_2, label="$\\frac{d^2a}{dt^2}$ (Second Derivative)", color="green", linestyle=":")
plt.title("Scale Factor and its Derivatives in Radiation-Dominated Era")
plt.xlabel("Cosmic Time $t$ (s)")
plt.ylabel("Scale Factor / Derivatives")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()


# Compute the cosmological term a(tau)''/a(tau)
adprimea = (a_t_rad_derivative_1)**2 + a_t_rad_derivative_2 * a_t_rad

# Plot the term a(tau)''/a(tau)
plt.figure(figsize=(12, 6))
plt.plot(t_range_rad, adprimea, label="$a(\\tau)''/a(\\tau)$", color="purple", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Expected Zero")
plt.title("Cosmological Term $a(\\tau)''/a(\\tau)$ in the Radiation-Dominated Era")
plt.xlabel("Cosmic Time $t$ (s)")
plt.ylabel("$a(\\tau)''/a(\\tau)$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()

# Verify if the term approaches zero
max_deviation = np.max(np.abs(adprimea))
print(f"Maximum deviation of $a(\\tau)''/a(\\tau)$ from zero: {max_deviation:.2e}")


# ## Define a($\tau$) Spline

# In[4]:


# Define the integral to compute conformal time tau
def compute_tau(spline_a_t, t_min, t_max):
    """Compute conformal time tau for a given spline a(t)."""
    result, _ = quad(lambda t: 1 / spline_a_t(t), t_min, t_max, limit=1000)
    return result

# Create a spline for a(t)
#spline_a_t_rad = UnivariateSpline(t_a_rad, a_range_rad, s=0)

# # Define time range for evaluating the spline
# t_range_rad = np.linspace(t_a_rad.min(), t_a_rad.max(), len(a_range_rad))
# a_t_rad = spline_a_t_rad(t_range_rad)

# Compute tau for the full radiation-dominated era
tau_t_rad = compute_tau(spline_a_t_rad, t_a_rad.min(), t_a_rad.max()) #computes one at a time
print(f"Conformal time τ (radiation era): {tau_t_rad:.5e} s")

# Generate sampled tau values for t_range
tau_t_array_rad = np.array([compute_tau(spline_a_t_rad, t_a_rad.min(), t) for t in t_range_rad])

# Create a spline for a(tau)
spline_a_tau = UnivariateSpline(tau_t_array_rad, a_t_rad, s=0)

# Define a range of tau values for evaluation
tau_range = np.linspace(tau_t_array_rad.min(), tau_t_array_rad.max(), 500)
a_tau = spline_a_tau(tau_range)

# Validation and Visualization
print(f"Min a(τ): {np.amin(a_tau):.5e}, Max a(τ): {np.amax(a_tau):.5e}")

plt.figure(figsize=(12, 6))
plt.plot(tau_range, a_tau, label="$a(\\tau)$ (Spline)", color="red", linewidth=2)
plt.title("Scale Factor $a(\\tau)$ in Radiation-Dominated Era")
plt.xlabel("Conformal Time $\\tau$ (s)")
plt.ylabel("Scale Factor $a$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(tau_range, a_t_rad, label="$a(t)$ (Spline)", color="blue", linewidth=2)
plt.plot(tau_range, a_tau, label="$a(\\tau)$ (Spline)", color="red", linestyle="--", linewidth=2)
#plt.plot(tau_range, a_tau, label="$a(\\tau)$ (Spline)", color="red", linestyle="--", linewidth=2)
plt.title("Comparison of $a(t)$ and $a(\\tau)$ in Radiation-Dominated Era")
plt.xlabel("Conformal Time (s)")
plt.ylabel("Scale Factor $a$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(t_range_rad, a_t_rad, label="$a(t)$ (Spline)", color="blue", linewidth=2)
plt.plot(t_range_rad, a_tau, label="$a(\\tau)$ (Spline)", color="red", linestyle="--", linewidth=2)
#plt.plot(tau_range, a_tau, label="$a(\\tau)$ (Spline)", color="red", linestyle="--", linewidth=2)
plt.title("Comparison of $a(t)$ and $a(\\tau)$ in Radiation-Dominated Era")
plt.xlabel("Time (s)")
plt.ylabel("Scale Factor $a$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()


# ### Compare the Derivatives of a($\tau$)

# In[5]:


# First derivative of the spline: da/dτ
spline_a_tau_derivative_1 = spline_a_tau.derivative(n=1)
a_tau_derivative_1 = spline_a_tau_derivative_1(tau_range)

# Second derivative of the spline: d^2a/dτ^2
spline_a_tau_derivative_2 = spline_a_tau.derivative(n=2)
a_tau_derivative_2 = spline_a_tau_derivative_2(tau_range)

# Plotting the scale factor and its derivatives
plt.figure(figsize=(12, 6))

# Plot the scale factor a(τ)
plt.plot(tau_range, a_tau, label="$a(\\tau)$ (Scale Factor)", color="red", linewidth=2)

# Plot the first derivative da/dτ
plt.plot(tau_range, a_tau_derivative_1, label="$\\frac{da}{d\\tau}$ (First Derivative)", color="blue", linestyle="--")

# Plot the second derivative d^2a/dτ^2
plt.plot(tau_range, a_tau_derivative_2, label="$\\frac{d^2a}{d\\tau^2}$ (Second Derivative)", color="green", linestyle=":")

# Add labels, title, and grid
plt.title("Scale Factor and its Derivatives wrt Conformal Time $\\tau$")
plt.xlabel("Conformal Time $\\tau$")
plt.ylabel("Scale Factor / Derivatives")
plt.legend(loc="best")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()


# ## Define a($k\tau$) Spline (Incorporating k)

# In[6]:


# Define a function to compute extended a(k * tau)
def compute_extended_a_k_tau(tau_array, spline_a_tau, k, num_points=500):
    """
    Compute a(k * tau) by extending the range of tau for a specific k.

    tau_array: Original conformal time array.
    spline_a_tau: Spline for a(tau).
    k: Scaling factor for tau.
    num_points: Resolution for the extended tau range.
    """
    # Extend tau range for the specific k
    tau_min_k = tau_array.min() * k
    tau_max_k = tau_array.max() * k
    ktau_range = np.linspace(tau_min_k, tau_max_k, num_points)

    # Create a spline for a(k * tau)
    spline_a_ktau = UnivariateSpline(k * tau_array, spline_a_tau(tau_array), s=0)

    # Evaluate a(k * tau) over the extended range
    a_k_tau = spline_a_ktau(ktau_range)

    return ktau_range, a_k_tau, spline_a_ktau

# Compute a(k * tau) for k = 1
k = 1
ktau_range, a_k_tau, spline_a_ktau = compute_extended_a_k_tau(tau_t_array_rad, spline_a_tau, k)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(ktau_range, a_k_tau, label="$a(k\\tau)$ if $k = 1$", color="blue", linewidth=2)
plt.plot(tau_range, a_tau, label="Original (no k or k=1) $a(\\tau)$ ", color="red", linestyle="--", linewidth=2)
plt.title("$a(k\\tau)$ for $k = 1$")
plt.xlabel("Conformal Time $k\\tau$")
plt.ylabel("Scale Factor $a$")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# In[7]:


fig, axs = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

k01tau_range, a_k01_tau, spline_a_k01tau = compute_extended_a_k_tau(tau_t_array_rad, spline_a_tau, 0.1)
k1tau_range, a_k1_tau, spline_a_k1tau = compute_extended_a_k_tau(tau_t_array_rad, spline_a_tau, 1)
k1000tau_range, a_k1000_tau, spline_a_k1000tau = compute_extended_a_k_tau(tau_t_array_rad, spline_a_tau, 1000)

print(k1tau_range.shape)

# Subplot 1: k = 0.1
axs[0].plot(tau_range, a_tau, label="Original (no k or k=1) $a(\\tau)$ ", color="red", linestyle="--", linewidth=2)
axs[0].plot(k01tau_range, a_k01_tau, label="$k = 0.1$", color="g", linestyle='dashed', linewidth=2)
axs[0].set_title("$a(\\tau)$ for $k = 0.1$")
axs[0].set_xlabel("Conformal Time $\\tau$")
axs[0].set_ylabel("Scale Factor $a$")
axs[0].grid(which="both", linestyle="--", linewidth=0.5)
axs[0].legend()

# Subplot 2: k = 1
axs[1].plot(tau_range, a_tau, label="Original (no k or k=1) $a(\\tau)$ ", color="red", linestyle='solid', linewidth=2)
axs[1].plot(k1tau_range, a_k1_tau, label="$k = 1$", color="purple", linestyle='--', linewidth=2)
axs[1].set_title("$a(k\\tau)$ for $k = 1$")
axs[1].set_xlabel("Conformal Time $k\\tau$")
axs[1].grid(which="both", linestyle="--", linewidth=0.5)
axs[1].legend()

# Subplot 2: k = 1000
axs[2].plot(tau_range, a_tau, label="Original (no k or k=1) $a(\\tau)$ ", color="red", linestyle="--", linewidth=2)
axs[2].plot(k1000tau_range, a_k1000_tau, label="$k = 1000$", color="orange", linestyle='dashed', linewidth=2)
axs[2].set_title("$a(k\\tau)$ for $k = 1000$")
axs[2].set_xlabel("Conformal Time $k\\tau$")
axs[2].grid(which="both", linestyle="--", linewidth=0.5)
axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.show()


# ### Compare the Derivatives of a($k\tau$)

# In[8]:


#Assume k =1,

# First derivative of the spline: da/dτ
spline_a_k1tau_derivative_1 = spline_a_k1tau.derivative(n=1)
a_k1tau_derivative_1 = spline_a_k1tau_derivative_1(k1tau_range)

# Second derivative of the spline: d^2a/dτ^2
spline_a_k1tau_derivative_2 = spline_a_k1tau.derivative(n=2)
a_k1tau_derivative_2 = spline_a_k1tau_derivative_2(k1tau_range)

# Plotting the scale factor and its derivatives
plt.figure(figsize=(12, 6))

# Plot the scale factor a(kτ) k =1
plt.plot(k1tau_range, a_tau, label="$a(k\\tau)$ (Scale Factor)", color="red", linewidth=2)
# Plot the scale factor a(τ)
plt.plot(tau_range, a_tau, label="$a(\\tau)$ (Scale Factor)", color="k", linewidth=2, linestyle='--')

# Plot the first derivative da/dkτ
plt.plot(k1tau_range, a_k1tau_derivative_1, label="$\\frac{da}{dk\\tau}$ (First Derivative)", color="blue", linestyle="solid")
# Plot the first derivative da/dτ
plt.plot(tau_range, a_tau_derivative_1, label="$\\frac{da}{d\\tau}$ (First Derivative)", color="k", linestyle="--")


# Plot the second derivative d^2a/dkτ^2
plt.plot(k1tau_range, a_k1tau_derivative_2, label="$\\frac{d^2a}{dk\\tau^2}$ (Second Derivative)", color="green", linestyle="solid")
# Plot the second derivative d^2a/dτ^2
plt.plot(tau_range, a_tau_derivative_2, label="$\\frac{d^2a}{d\\tau^2}$ (Second Derivative)", color="k", linestyle=":")


# Add labels, title, and grid
plt.title("Scale Factor and its Derivatives wrt Conformal Time $k\\tau$")
plt.xlabel("Conformal Time $k\\tau$")
plt.ylabel("Scale Factor / Derivatives")
plt.legend(loc="best")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()


# In[9]:


fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Subplot 1: Scale Factor
axs[0].plot(k1tau_range, a_tau, label="$a(k\\tau)$ (Scale Factor)", color="red", linewidth=2)
axs[0].plot(tau_range, a_tau, label="$a(\\tau)$ (Scale Factor)", color="k", linestyle='--', linewidth=2)
axs[0].set_title("Scale Factor $a(k\\tau)$ and $a(\\tau)$")
axs[0].set_xlabel("Conformal Time $k\\tau$")
axs[0].set_ylabel("Scale Factor $a$")
axs[0].legend(loc="best")
axs[0].grid(which="both", linestyle="--", linewidth=0.5)

# Subplot 2: First Derivative
axs[1].plot(k1tau_range, a_k1tau_derivative_1, label="$\\frac{da}{dk\\tau}$", color="blue", linewidth=2)
axs[1].plot(tau_range, a_tau_derivative_1, label="$\\frac{da}{d\\tau}$", color="k", linestyle="--", linewidth=2)
axs[1].set_title("First Derivative $\\frac{da}{dk\\tau}$ and $\\frac{da}{d\\tau}$")
axs[1].set_xlabel("Conformal Time $k\\tau$")
axs[1].grid(which="both", linestyle="--", linewidth=0.5)
axs[1].legend(loc="best")

# Subplot 3: Second Derivative
axs[2].plot(k1tau_range, a_k1tau_derivative_2, label="$\\frac{d^2a}{dk\\tau^2}$", color="green", linewidth=2)
axs[2].plot(tau_range, a_tau_derivative_2, label="$\\frac{d^2a}{d\\tau^2}$", color="k", linestyle="--", linewidth=2)
axs[2].set_title("Second Derivative $\\frac{d^2a}{dk\\tau^2}$ and $\\frac{d^2a}{d\\tau^2}$")
axs[2].set_xlabel("Conformal Time $k\\tau$")
axs[2].grid(which="both", linestyle="--", linewidth=0.5)
axs[2].legend(loc="best")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# # PHASE II: APPLYING WKB APPROXIMATION
# 
# ## Original Bessel Approximation

# In[10]:


#The Working Spherical Bessel Case:

def plot_wkb_approximation(l):
    
    # Calculate epsilon, x0, and constant based on l
    epsilon = 1 / np.sqrt(l * (l + 1))
    x0 = 1 / epsilon
    constant = np.sqrt(epsilon) / 2
    
    # Define x_array and corresponding w_array
    x_array = np.linspace(1e-3, 15 + 2 * l, 1000)
    w_array = x0 - x_array  # w = x0 - x
    #print('w_array:', w_array)
    #print('w_array min:', np.amin(w_array), 'w_array max:', np.amax(w_array))

    # Define Q(w) function using x = x0 - w
    Q_func = lambda w: (1 / (x0 - w)**2) - epsilon**2
    
    # Calculate S0 using adaptive integration
    S0 = np.array([quad(lambda w: np.sqrt(np.abs(Q_func(w))), 0, wi)[0] for wi in w_array])
    
    # Calculate expr and Q_vals, preserving signs
    expr = (3 * S0) / (2 * epsilon)
    Q_vals = Q_func(w_array)
    Q_factor = np.sign(Q_vals) * (np.abs(Q_vals) ** (-1/4))

    # Calculate the general solution y using signed powers
    y_general = 2 * np.sqrt(np.pi) * constant * (np.sign(expr) * np.abs(expr) ** (1/6))                 * Q_factor * airy(np.sign(expr) * np.abs(expr) ** (2/3))[0]

    # Plot y/x vs spherical Bessel function for comparison
    spherical_bessel = spherical_jn(l, x_array)
    plt.figure(figsize=(10, 6))

    # Define colors for the WKB and Bessel function
    wkb_color = '#DB7093'  # Vibrant rose for WKB approximation
    bessel_color = '#4682B4'  # Soft teal for the Bessel function
    
    # Background shading for Regions 1 and 3
    plt.axvspan(0, max(w_array), color='#FFB6C1', alpha=0.2, label='Region 1 (w > 0)')
    plt.axvspan(min(w_array), 0, color='#DDA0DD', alpha=0.2, label='Region 3 (w < 0)')
    
    # Plot the WKB Approximation and Bessel function
    plt.plot(w_array, y_general / x_array, label=r'$y/x$ (WKB Approximation)', color=wkb_color, linewidth=1.5)
    plt.plot(w_array, spherical_bessel, label=f'Spherical Bessel $j_{l}$', color=bessel_color, linewidth=1.5, linestyle='--')
    
    # Titles and labels with enhanced, stylish colors
    plt.title(f'WKB Approximation vs Spherical Bessel for $l = {l}$', fontsize=16, weight='bold', color='#333333')
    plt.xlabel('$w$', fontsize=14, color='#333333')
    plt.ylabel(r'$y/x$', fontsize=14, color='#333333')
    
    # Adding grid, legend, and axis enhancements
    plt.axvline(x=0, linestyle='--', color='gray', linewidth=1.2, label='$x_0$')
    plt.axhline(y=0, linestyle='--', color='slategray', linewidth=1.2)
    plt.grid(color='lightgrey', linestyle=':', linewidth=0.8)
    plt.legend(fontsize=12, loc='best', frameon=False)
    
    # Enhance plot aesthetics
    plt.tight_layout(pad=2)
    #plt.savefig('wkb_l5_w.png')
    plt.show()

# Example usage for l = 5
plot_wkb_approximation(5)


# ## Testing the Approximation in the Radiation Era
# 
# 
# ### What is my initial condition?
# 
# I did the following when h=1 to acquire the true value! Around 65.
# 
#         h0 = y_general_test_a[0] / a_test_rad[0]  # Limit as a -> 0
#         print(h0)

# In[26]:


import numpy as np
from numpy import sqrt, sin, cos, pi
import matplotlib.pyplot as plt
from scipy import special
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
from scipy.misc import derivative
from scipy.special import airy, spherical_jn
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline, splrep, splev
from scipy.integrate import cumtrapz
from scipy.optimize import root_scalar
from scipy.optimize import newton


# Cosmological parameters
H0 = 70.0  # Hubble constant in km/s/Mpc
H0_s = H0 / (3.086e19)  # Convert H0 to s^-1
H0_s_m = H0_s * (1/(3e8)) #H0(1/s)*1/(s/m) to be m^-1

omega_m = 0.3          # Matter density parameter
omega_lambda = 0.7     # Cosmological constant density parameter
omega_r = 2.9e-4       # Radiation density parameter
omega_all = 1 #omega_m + omega_lambda + omega_r
omega_k = 1 - omega_all  # Curvature density parameter

a_min = 1e-2
a_mr_eq = 2.9*1e-4 #a at matter-radiation equality
a_ml_eq = 0.77
a_max = 1

def radiation_wkb_approximation(k):

################################################################################################################

    #Define Hubble for the Radiation Only Era
    
    def H_radiation(a):
        """Hubble parameter during the radiation-dominated era."""
        return H0_s * np.sqrt((omega_r / a**4) + (omega_k / a**2))
    
    
    #DEFINE a(t): a_t_rad
    
    # Define scale factor range
    a_min = 1e-2
    a_max = 1  # Up to a = 1
    a_range_rad = np.logspace(np.log10(a_min), np.log10(a_max), num=500)
    a_range_rad= a_range_rad

    # Compute t(a) for the radiation-dominated era
    t_a_rad = np.array([
        quad(lambda a: 1 / H_radiation(a), a_min, a, limit=1000)[0]
        for a in a_range_rad
    ])

    # Create a spline for a(t)
    spline_a_t_rad = UnivariateSpline(t_a_rad, a_range_rad, s=0)

    # Define time range for evaluating the spline
    t_range_rad = np.linspace(t_a_rad.min(), t_a_rad.max(), len(a_range_rad))
    a_t_rad = spline_a_t_rad(t_range_rad)
#     a_t_rad = a_t_rad/1e6
#     print('max a(t)',np.amax(a_t_rad))

################################################################################################################

    #DEFINE a(tau): a_tau
    
    # Define the integral to compute conformal time tau
    def compute_tau(spline_a_t, t_min, t_max):
        """Compute conformal time tau for a given spline a(t)."""
        result, _ = quad(lambda t: 1 / spline_a_t(t), t_min, t_max, limit=1000)
        return result

    # Compute tau for the full radiation-dominated era
    tau_t_rad = compute_tau(spline_a_t_rad, t_a_rad.min(), t_a_rad.max()) #computes one at a time

    # Generate sampled tau values for t_range
    tau_t_array_rad = np.array([compute_tau(spline_a_t_rad, t_a_rad.min(), t) for t in t_range_rad])
    #print('max tau(t) value', np.amax(tau_t_array_rad))
    tau_t_array_rad = tau_t_array_rad / 1e18

    
    # Create a spline for a(tau)
    spline_a_tau = UnivariateSpline(tau_t_array_rad, a_t_rad, s=0)

    # Define a range of tau values for evaluation
    tau_range = np.linspace(tau_t_array_rad.min(), tau_t_array_rad.max(), 500)
    a_tau = spline_a_tau(tau_range)
    

################################################################################################################

    
    # DEFINE a(k*tau): a_k_tau
    
    # Define the new spline a(ktau) 
    def compute_extended_a_k_tau(tau_array, spline_a_tau, k, num_points=500):
    
        # Extend tau range for the specific k
        tau_min_k = tau_array.min() * k
        tau_max_k = tau_array.max() * k
        ktau_range = np.linspace(tau_min_k, tau_max_k, num_points)

        # Create a spline for a(k * tau)
        spline_a_ktau = UnivariateSpline(k * tau_array, spline_a_tau(tau_array), s=0)

        # Evaluate a(k * tau) over the extended range
        a_k_tau = spline_a_ktau(ktau_range)

        return ktau_range, a_k_tau, spline_a_ktau


    ktau_range, a_k_tau, spline_a_ktau = compute_extended_a_k_tau(tau_t_array_rad, spline_a_tau, k)


################################################################################################################

    ### Defining the parameters for the wkb approximation ###
    
    
    # Calculate epsilon,h(0):
    epsilon = 1
    h0 = 1 #65
    
    ### Variables of Interest
    
    # a(ktau) = a(x)
    a_array = a_k_tau
    
    # w = astar - a, meant to match relative position 
    a0 = 0 # Q(x) = 0
    w_array = a0 - a_array 

    # x = ktau
    x_array = ktau_range
    #x_array[x_array == 0] = 1e-10 #SHOULD replace with 1e-10
    #x_array[x_array != 0] #would remove all

    
    ### Define Q (in each variable)
    
    Q_func = lambda z: np.full_like(z,-1)

    # a(x):
    Q_vals_a = Q_func(a_array)
    Q_factor_a = np.sign(Q_vals_a) * (np.abs(Q_vals_a) ** (-1/4)) #Factor will be used in y calculation
    
    # w = astar - a(x):
    Q_vals_w = Q_func(w_array)
    Q_factor_w = np.sign(Q_vals_w) * (np.abs(Q_vals_w) ** (-1/4))


    # x = ktau:
    Q_vals_x = Q_func(x_array)
    Q_factor_x = np.sign(Q_vals_x) * (np.abs(Q_vals_x) ** (-1/4))

#     print('Q factor (a)', np.sign(Q_factor_a)) -1
#     print('Q factor (w)', np.sign(Q_factor_w)) -1
#     print('Q factor (x)', np.sign(Q_factor_x)) -1


    ### Define S0 (in each variable)
    
    
    # a(x):
    S0_a = np.array([quad(lambda a: np.sign(Q_func(a)) * np.sqrt(np.abs(Q_func(a))), 0, ai)[0] for ai in a_array])
    
    # w = astar - a(x):
    S0_w = np.array([quad(lambda w: np.sign(Q_func(w)) * np.sqrt(np.abs(Q_func(w))), 0, wi)[0] for wi in w_array])

    # x = ktau:
    S0_x = np.array([quad(lambda x: np.sign(Q_func(x)) * np.sqrt(np.abs(Q_func(x))), 0, xi)[0] for xi in x_array])

#     print('S factor (a)', np.sign(S0_a)) -1
#     print('S factor (w)', np.sign(S0_w)) 1
#     print('S factor (x)', np.sign(S0_x)) -1

    
    ### Airy Input and Function (in each variable)
    
    # a(x): 
    expr_a = (3 * S0_a) / (2 * epsilon)
    airy_input_a = np.sign(expr_a) * np.abs(expr_a) ** (2/3)
    airy_func_a = airy(airy_input_a)[0] #Takes only Ai
    
    # w = astar - a(x):
    expr_w = (3 * S0_w) / (2 * epsilon)
    airy_input_w = np.sign(expr_w) * np.abs(expr_w) ** (2/3)
    airy_func_w = airy(airy_input_w)[0] #Takes only Ai

    # x = ktau:
    expr_x = (3 * S0_x) / (2 * epsilon) #DOES EPSILON CHANGE WITH A FACTOR OF K
    airy_input_x = np.sign(expr_x) * np.abs(expr_x) ** (2/3)
    airy_func_x = airy(airy_input_x)[0] #Takes only Ai

#     print('Expr factor (a)', np.sign(expr_a)) -1
#     print('Expr factor (w)', np.sign(expr_w)) 1
#     print('Expr factor (x)', np.sign(expr_x)) -1

    
     ### General Solution Calculation (in each variable)
    
    # a(x):
    y_gen_a = 2 * np.sqrt(np.pi) * h0 * (np.sign(expr_a) * np.abs(expr_a) ** (1/6)) * Q_factor_a * airy_func_a
    
    # w = astar - a(x):
    y_gen_w = 2 * np.sqrt(np.pi) * h0 * (np.sign(expr_w) * np.abs(expr_w) ** (1/6)) * Q_factor_w * airy_func_w

    # x = ktau:
    y_gen_x = 2 * np.sqrt(np.pi) * h0 * (np.sign(expr_x) * np.abs(expr_x) ** (1/6)) * Q_factor_x * airy_func_x

    
################################################################################################################

    ### Define h from analytical solutions (Pritchard)
    
    
    analytical_h_a = h0 * spherical_jn(0, a_array)
    analytical_h_x = h0 * spherical_jn(0, x_array)
    

    term_pritchard = np.sqrt(tau_t_array_rad)*tau_t_array_rad+2
    term_pritchard = np.sqrt(tau_t_array_rad + 1e-10) * (tau_t_array_rad + 2) * (1e6)
    
    
    # Create a mask to exclude x_array == 0
    non_zero_mask = x_array != 0
    x_array_nonzero = x_array[non_zero_mask]
    y_gen_x_nonzero = y_gen_x[non_zero_mask]
    term_pritchard_nonzero = term_pritchard[non_zero_mask]
    analytical_h_x_nonzero = analytical_h_x[non_zero_mask]
    
 ################################################################################################################   
    # Create a large figure for the subplots
    fig, axes = plt.subplots(6, 3, figsize=(12, 20))
    fig.suptitle("Radiation Era WKB Approximation: Various Quantities", fontsize=16)

    #ROW 1: Q
    # Plots: x_axis vs Q
    #a vs Q_a
    axes[0, 0].plot(a_array, Q_vals_a, label="$Q_a$", color='blue')
    axes[0, 0].set_title("a vs Q_a")
    axes[0, 0].set_xlabel("$a(x) = a(k \\tau)$")
    axes[0, 0].set_ylabel("$Q_x$")
    axes[0, 0].grid()
    axes[0, 0].legend()

    # w vs Q_w
    axes[0, 1].plot(w_array, Q_vals_w, label="$Q_w$", color='blue')
    axes[0, 1].set_title("w vs Q_w")
    axes[0, 1].set_xlabel("$w$")
    axes[0, 1].set_ylabel("$Q_w$")
    axes[0, 1].grid()
    axes[0, 1].legend()

    #x vs Q_x
    axes[0, 2].plot(x_array, Q_vals_x, label="$Q_x$", color='blue')
    axes[0, 2].set_title("x vs Q_x")
    axes[0, 2].set_xlabel("$k \\tau)$")
    axes[0, 2].set_ylabel("$Q_x$")
    axes[0, 2].grid()
    axes[0, 2].legend()

    #ROW 2: S0
    # Plots: x_axis a vs S0
    #a vs S0_a
    axes[1, 0].plot(a_array, S0_a, label="$S_0(a)$", color='orange')
    axes[1, 0].set_title("a vs $S_0(a)$")
    axes[1, 0].set_xlabel("$a(x) = a(k \\tau)$")
    axes[1, 0].set_ylabel("$S_0(a)$")
    axes[1, 0].grid()
    axes[1, 0].legend()

    #w vs S0_w
    axes[1, 1].plot(w_array, S0_w, label="$S_0(w)$", color='orange')
    axes[1, 1].set_title("w vs $S_0(w)$")
    axes[1, 1].set_xlabel("$w$")
    axes[1, 1].set_ylabel("$S_0(w)$")
    axes[1, 1].grid()
    axes[1, 1].legend()

    #x vs S0_x
    axes[1, 2].plot(x_array, S0_x, label="$S_0(x)$", color='orange')
    axes[1, 2].set_title("x vs $S_0(x)$")
    axes[1, 2].set_xlabel("$x$")
    axes[1, 2].set_ylabel("$S_0(x)$")
    axes[1, 2].grid()
    axes[1, 2].legend()


    #ROW 3
    # Plots: x_axis vs airy_arg
    #a vs airy_arg_a

    axes[2, 0].plot(a_array, airy_func_a, label="Airy Function $w$", color='purple')
    axes[2, 0].set_title("a vs Airy Function (a)")
    axes[2, 0].set_xlabel("$a(x) = a(k \\tau)$")
    axes[2, 0].set_ylabel("Airy Function")
    axes[2, 0].grid()
    axes[2, 0].legend()

    #w vs airy_arg_w
    axes[2, 1].plot(w_array, airy_func_w, label="Airy Function $w$", color='purple')
    axes[2, 1].set_title("w vs Airy Function (w)")
    axes[2, 1].set_xlabel("$w$")
    axes[2, 1].set_ylabel("Airy Function (w)")
    axes[2, 1].grid()
    axes[2, 1].legend()

    #x vs airy_arg_x
    axes[2, 2].plot(x_array, airy_func_x, label="Airy Function $x$", color='purple')
    axes[2, 2].set_title("x vs Airy Function (x)")
    axes[2, 2].set_xlabel("$x$")
    axes[2, 2].set_ylabel("Airy Function (x)")
    axes[2, 2].grid()
    axes[2, 2].legend()


    #ROW 4
    # Plots: x_axis vs y
    #a vs y_a
    axes[3, 0].plot(a_array, y_gen_a, label="$y(a)$", color='red')
    axes[3, 0].set_title("a vs $y(a)$")
    axes[3, 0].set_xlabel("$a(x) = a(k \\tau)$")
    axes[3, 0].set_ylabel("$y(a)$")
    axes[3, 0].grid()
    axes[3, 0].legend()

    #w vs y_w
    axes[3, 1].plot(w_array, y_gen_w, label="$y(w)$", color='red')
    axes[3, 1].set_title("w vs $y(w)$")
    axes[3, 1].set_xlabel("$w$")
    axes[3, 1].set_ylabel("$y(w)$")
    axes[3, 1].grid()
    axes[3, 1].legend()

    #x vs y_x
    axes[3, 2].plot(x_array, y_gen_x, label="$y(x)$", color='red')
    axes[3, 2].set_title("x vs $y(x)$")
    axes[3, 2].set_xlabel("x")
    axes[3, 2].set_ylabel("$y(x)$")
    axes[3, 2].grid()
    axes[3, 2].legend()

    
    #ROW 5
    # Plots: x_axis vs h and analytical_h
    #a vs h_a and analytical_h
    axes[4, 0].plot(a_array, y_gen_a, label="$y(a)$", color='r', linestyle='dashed')
    axes[4, 0].plot(a_array, analytical_h_a*a_array, label="Analytical $h(a)*a$", color='mediumorchid')
    axes[4, 0].set_title("a vs $y(a)$ and Analytical $h*a$")
    axes[4, 0].set_xlabel("$a(x) = a(k \\tau)$")
    axes[4, 0].set_ylabel("$Amplitude$")
    axes[4, 0].grid()
    axes[4, 0].legend()

    #w vs Analytical h_w
    axes[4, 1].plot(w_array, y_gen_a, label="$y(w)$", color='r', linestyle='dashed')
    axes[4, 1].plot(w_array, analytical_h_a*a_array, label="Analytical $h(a)*a$", color='mediumorchid')
    axes[4, 1].set_title("w vs $y(w)$ and Analytical $h(w)$")
    axes[4, 1].set_xlabel("$w$")
    axes[4, 1].set_ylabel("$Amplitude$")
    axes[4, 1].grid()
    axes[4, 1].legend()

    #x vs Analytical h_x
    axes[4, 2].plot(x_array, y_gen_x, label="$y(x)$", color='r', linestyle='dashed')
    axes[4, 2].plot(x_array, analytical_h_x*x_array, label="Analytical $h(x)*x$", color='mediumorchid')
    axes[4, 2].set_title("x vs $y(x)$ and Analytical $h(x)*x$")
    axes[4, 2].set_xlabel("$x$")
    axes[4, 2].set_ylabel("$Amplitude$")
    axes[4, 2].grid()
    axes[4, 2].legend()
    
    

    #ROW 6
    # Plots: x_axis vs h and analytical_h
    #a vs h_a and analytical_h
    axes[5, 0].plot(a_array, y_gen_a/(a_array), label="$h_a = y(a)/a$", color='g', linestyle='dashed')
    axes[5, 0].plot(a_array, analytical_h_a, label="Analytical $h(a)$", color='magenta')
    axes[5, 0].set_title("a vs $h(a)$ and Analytical $h(a)$")
    axes[5, 0].set_xlabel("$a(x) = a(k \\tau)$")
    axes[5, 0].set_ylabel("$h(a)$")
    axes[5, 0].grid()
    axes[5, 0].legend()
    
    #w vs Analytical h_w
    axes[5, 1].plot(w_array, y_gen_a/(a_array), label="$h_w = y(w)/a$", color='g', linestyle='dashed')
    axes[5, 1].plot(w_array, analytical_h_a, label="Analytical $h(w)$", color='magenta')
    axes[5, 1].set_title("w vs Analytical $h(w)$")
    axes[5, 1].set_xlabel("$w$")
    axes[5, 1].set_ylabel("$h(w)$")
    axes[5, 1].grid()
    axes[5, 1].legend()

#     #x vs Analytical h_x
#     axes[5, 2].plot(x_array, y_gen_x/(term_pritchard*x_array), label="$h_x$ = y(x)/$x scaled$", color='g', linestyle='dashed')
#     axes[5, 2].plot(x_array, analytical_h_x, label="Analytical $h(x)$", color='magenta')
#     axes[5, 2].set_title("x vs Analytical $h(x)$")
#     axes[5, 2].set_xlabel("$x$")
#     axes[5, 2].set_ylabel("$h(x)$")
#     axes[5, 2].grid()
#     axes[5, 2].legend()

    # x vs Analytical h(x), excluding x_array == 0
    axes[5, 2].plot(x_array_nonzero, y_gen_x_nonzero / (term_pritchard_nonzero * x_array_nonzero), 
                    label="$h_x$ = y(x)/$x$ (scaled)", color="g", linestyle="dashed")
    axes[5, 2].plot(x_array_nonzero, analytical_h_x_nonzero, label="Analytical $h(x)$", color="magenta")
    axes[5, 2].set_title("x vs Analytical $h(x)$")
    axes[5, 2].set_xlabel("$x$")
    axes[5, 2].set_ylabel("$h(x)$")
    axes[5, 2].grid()
    axes[5, 2].legend()

    
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#     print('a(t) max:',np.amax(a_t_rad))
#     print('a(t) min:', np.amin(a_t_rad))
    
#     print('a(tau) max:',np.amax(a_tau))
#     print('a(tau) min:', np.amin(a_tau))
    
#     print('a(ktau) max:',np.amax(a_k_tau))
#     print('a(ktau) min:', np.amin(a_k_tau))
    
    

#Test of Functiion
radiation_wkb_approximation(k=1)

