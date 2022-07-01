# Current density for an S/F/S hybrid structure
#
# author:  Leo Kollwitz
# email:   leo.kollwitz@gmail.com
# date:    15th July 2022
#
# This python script contains an implementation of a scattering matrix approach used to calculate the current density
# of an S/F/S hybrid structure depending on various parameters such as temperature or polarization strength.
# The script was created in the scope of my Bachelors thesis, in which the calculations and details are explained.
# The full thesis can be found at https://github.com/lKollwitz/bachelor_thesis.


# necessary libraries to run the script
import numpy as np
import matplotlib.pyplot as plt
import cmath
from time import time
from numba import jit

# starting time for run time comparisons. Not strictly needed
START_TIME = time()

# global configuration for the diagrams created by this script
plt.style.use(['grid'])

# ===================== global constants ===============================================================================
# 2x2 and 4x4 identity matrix
ID2 = np.identity(2)
ID4 = np.identity(4)

# third Pauli matrix in 4x4 spin x particle-hole space
TAU3 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]])


# ===================== functions ======================================================================================
def update_params(params, ind, value):
    # params: complex array, parameters of the system
    # ind: integer
    # value: complex float
    #
    # sets a new value for parameter at index ind
    # returns new params

    params[ind] = value
    return params


@jit(nopython=True)
def build_block_diagonal_matrix_2x2(a1, a2):
    # a1, a2: real / complex float
    #
    # returns diagonal real / complex 2x2 matrix with entries a1 and a2 on the main diagonal
    return np.array([[a1, 0], [0, a2]])


@jit(nopython=True)
def build_block_diagonal_matrix_4x4(a1, a2):
    # a1, a2: complex 2x2 arrays
    #
    # returns block diagonal 4x4 matrix with entries a1 and a2 on the main diagonal
    zeros = np.zeros((2, 2), dtype=np.complex_)
    mat1 = np.concatenate((a1, zeros), axis=1)
    mat2 = np.concatenate((zeros, a2), axis=1)

    return np.concatenate((mat1, mat2))


@jit(nopython=True)
def dagger(a):
    # a: real / complex matrix
    #
    # returns the hermitian conjugate of a
    return np.conjugate(a).transpose()


@jit(nopython=True)
def calc_k(v, e_f, k0, k_p):
    # v: real float, potential of region 1, 2 or 3
    # e_f: real float, Fermi energy of the particle
    # k0: real float, wave number used for normalization, equals to wave number of region 1 for zero parallel component
    # k_p: real float, parallel momentum component of the incoming particle
    #
    # returns normalized wave number

    #regions 1 and 3
    if v < e_f:
        k = cmath.sqrt(2 * (e_f - v) - k_p**2)
    #region 2
    if v > e_f:
        k = cmath.sqrt(2 * (v - e_f) + k_p ** 2)

    return k / k0


@jit(nopython=True)
def calc_sqrt_S(s):
    # s: complex 4x4 diagonal matrix
    #
    # returns one realization of the square root of s
    sqrtS11 = cmath.sqrt(s[0, 0])
    sqrtS22 = cmath.sqrt(s[1, 1])
    sqrtS33 = cmath.sqrt(s[2, 2])
    sqrtS44 = cmath.sqrt(s[3, 3])

    return np.array([[sqrtS11, 0, 0, 0],
                [0, sqrtS22, 0, 0],
                [0, 0, sqrtS33, 0],
                [0, 0, 0, sqrtS44]])


@jit(nopython=True)
def calc_r_gamma(k1, kappa, k3, width):
    # k1, k3 : complex float, wave number in region to the left and right of the interface
    # kappa: complex float, wave number inside the scattering barrier
    # width: real float, width of the barrier
    #
    # returns the variables r and Gamma that make up the scattering matrix (Eq. 3.23 in thesis)
    a = cmath.sqrt(k1 / k3)  # alpha
    b = cmath.sqrt(k1 * k3) / kappa  # beta

    s = np.sinh(kappa * width)
    c = np.cosh(kappa * width)

    r = 0.5 * ((a - 1/a)*c - 1j*(b + 1/b)*s)
    gamma = 2 / ((a + 1 / a) * c - 1j * (b - 1 / b) * s)

    return r, gamma


@jit(nopython=True)
def calc_sigma(k1, kappa2, k3, width):
    # k1, k3 : complex float, wave number in region to the left and right of the interface
    # kappa: complex float, wave number inside the scattering barrier
    # width: real float, width of the barrier
    #
    # returns the variable sigma that makes up the scattering matrix, when the potential V3 is greater than the Fermi
    # energy E_F. This case is however equivalent to having a purely imaginary k3 and using the function calc_r_gamma
    # above. This function was mainly written to check if these two cases are indeed the same, which they are.
    kappa3 = np.imag(k3)

    s = np.sinh(kappa2 * width)
    c = np.cosh(kappa2 * width)

    num = kappa3 * c + kappa2 * s
    den = k1 * (c + (kappa3 / kappa2) * s)

    sigma = num/den

    return sigma


@jit(nopython=True)
def calc_t(k_p, params):
    # kp: real float, momentum component parallel to the interface
    # params: complex array, parameters of the system
    #
    # returns 2x2 complex matrix t (Eq. 3.28 thesis)

    # get relevant parameters from params
    e_f = params[0]
    v1 = params[1]
    v2u = params[2]
    v2d = params[3]
    v3 = params[4]
    k0 = params[5]
    width_u = params[6]
    width_d = params[7]

    # calculate wave numbers
    k1 = calc_k(v1, e_f, k0, k_p)
    k2u = calc_k(v2u, e_f, k0, k_p)
    k2d = calc_k(v2d, e_f, k0, k_p)
    k3 = calc_k(v3, e_f, k0, k_p)

    # check if incoming momentum vector lies outside of other Fermi surface. This would result in imaginary momentum
    # component and zero transmission. See. Fig. 2.3 in thesis.
    if (np.real(k1) <= 0) or (np.real(k3) <= 0):
        t11 = 0
        t22 = 0

    # if that is not the case, calculate t after equation mentioned above
    else:
        r_u, gamma_u = calc_r_gamma(k1, k2u, k3, width_u)
        r_d, gamma_d = calc_r_gamma(k1, k2d, k3, width_d)

        gam_r_abs_u = np.abs(gamma_u * r_u)
        gam_r_abs_d = np.abs(gamma_d * r_d)

        t11 = r_u * (1 - gam_r_abs_u) / gam_r_abs_u
        t22 = r_d * (1 - gam_r_abs_d) / gam_r_abs_d

    t = build_block_diagonal_matrix_2x2(t11, t22)

    return t


@jit(nopython=True)
def calc_S1_S2(k_p, params):
    # kp: real float, momentum component parallel to the interface
    # params: complex array, parameters of the system
    #
    # returns two 2x2 complex matrices that are the components of the 4x4 auxiliary scattering matrix (Eq. 2.43 thesis)
    # Here, they are called S1 and S2, because S and Sul will be used for the corresponding 4x4 matrices (see below)

    # the two blocks below are calculated twice (here and for calc_t) the way I implemented it. Since they are by far
    # not the most time-consuming operations, I did not bother to rewrite it slightly more efficiently. However, it
    # might be worth if one wishes to decrease computation time.

    # get relevant parameters from params
    e_f = params[0]
    v1 = params[1]
    v2u = params[2]
    v2d = params[3]
    v3 = params[4]
    k0 = params[5]
    width_u = params[6]
    width_d = params[7]

    # calculate wave numbers
    k1 = calc_k(v1, e_f, k0, k_p)
    k2u = calc_k(v2u, e_f, k0, k_p)
    k2d = calc_k(v2d, e_f, k0, k_p)
    k3 = calc_k(v3, e_f, k0, k_p)

    #prevent division by 0
    if k1 == 0:
        k1 += 0.00000001

    if k3 == 0:
        k3 += 0.00000001

    # check if incoming momentum vector lies outside of other Fermi surface. This would result in imaginary momentum
    # component and zero transmission. See. Fig. 2.3 in thesis. Here only for k3, since it will be assumed that V3 >= V1
    elif np.real(k3) == 0:
        sigma_u = calc_sigma(k1, k2u, k3, width_u)
        sigma_d = calc_sigma(k1, k2d, k3, width_d)

        S1_11 = (1 - 1j * sigma_u) / (1 + 1j * sigma_u)
        S1_22 = (1 - 1j * sigma_d) / (1 + 1j * sigma_d)

        S2_11 = 1
        S2_22 = 1

    # if that is not the case, calculate t after equation mentioned above
    else:
        r_u, gamma_u = calc_r_gamma(k1, k2u, k3, width_u)
        r_d, gamma_d = calc_r_gamma(k1, k2d, k3, width_d)

        gam_r_abs_u = np.abs(gamma_u * r_u)
        gam_r_abs_d = np.abs(gamma_d * r_d)

        S1_11 = (gamma_u * r_u) / gam_r_abs_u
        S1_22 = (gamma_d * r_d) / gam_r_abs_d

        S2_11 = (gamma_u * np.conj(r_u)) / gam_r_abs_u
        S2_22 = (gamma_d * np.conj(r_d)) / gam_r_abs_d
    
    S1 = build_block_diagonal_matrix_2x2(S1_11, S1_22)
    S2 = build_block_diagonal_matrix_2x2(S2_11, S2_22)

    return S1, S2


@jit(nopython=True)
def calc_tau12_tau21(k_p, params):
    # kp: real float, momentum component parallel to the interface
    # params: complex array, parameters of the system
    #
    # returns two 2x2 complex matrices that are the hopping amplitudes from (Eq. 2.45) in the thesis. The index 12
    # corresponds from going left to right and 21 corresponds to going right to left
    s1, s2 = calc_S1_S2(k_p, params)
    t = calc_t(k_p, params)
    tau12 = t @ s2
    tau21 = dagger(t) @ s1

    return tau12, tau21


@jit(nopython=True)
def calc_S_Sul(k_p, params):
    # kp: real float, momentum component parallel to the interface
    # params: complex array, parameters of the system
    #
    # returns two 4x4 complex matrices that are the auxiliary scattering matrices in particle hole space from (Eq. 2.47)
    # in the thesis. Since everything depends on k_p ** 2, the minus in the definition of the tilde operation can be
    # ignored.
    S1, S2 = calc_S1_S2(k_p, params)

    S1_tilde, S2_tilde = np.transpose(S1), np.transpose(S2)

    S = build_block_diagonal_matrix_4x4(S1, S1_tilde)
    Sul = build_block_diagonal_matrix_4x4(S2, S2_tilde)

    return S, Sul


@jit(nopython=True)
def calc_tau_tau_ul(k_p, sqrtSd, sqrtSuld, params):
    # kp: real float, momentum component parallel to the interface
    # sqrtSd: complex 4x4 matrix, hermitian conjugate of left matrix in Eq. 2.48
    # sqrtSuld: complex 4x4 matrix, hermitian conjugate of right matrix in Eq. 2.48
    # params: complex array, parameters of the system
    #
    # returns two 4x4 complex matrices that are the hopping amplitudes in particle hole space from (Eq. 2.47)
    # in the thesis. Since everything depends on k_p ** 2, the minus in the definition of the tilde operation can be
    # ignored.
    tau12, tau21 = calc_tau12_tau21(k_p, params)
    S1, S2 = calc_S1_S2(k_p, params)

    tau12_tilde = S1.transpose() @ np.conj(tau12) @ S2.transpose()
    tau21_tilde = S2.transpose() @ np.conj(tau21) @ S1.transpose()

    tau_old = build_block_diagonal_matrix_4x4(tau12, tau12_tilde)
    tau_ul_old = build_block_diagonal_matrix_4x4(tau21, tau21_tilde)

    tau = sqrtSd @ tau_old @ sqrtSuld
    tau_ul = sqrtSuld @ tau_ul_old @ sqrtSd

    return tau, tau_ul


@jit(nopython=True)
def calc_z(energy, temp, delta_contour):
    # energy: real float, energy for the Green's function
    # temp: real float, temperature in units k_B * T_c with T_c the critical temperature of the superconductor
    # delta_contour: real float, parameter that determines how far away the contour is from the real axis
    #
    # returns the contour used for the integral in Eq. (3.40)

    im_part = delta_contour * (0.5 * np.pi * temp + 1 - 1 / ((1 + energy ** 2) ** 1.5))

    # im_part = delta_contour  # for contour based on Keldysh. see Sec. 3.4
    return energy + 1j * im_part

@jit(nopython=True)
def calc_dz_de(energy,  delta_contour):
    # energy: real float, energy for the Green's function
    # delta_contour: real float, parameter that determines how far away the contour is from the real axis
    #
    # returns the derivative of contour used for the integral in Eq. (3.40) (see Eq. 3.39 in thesis)

    return 1 + 1j * delta_contour * (3 * energy / ((1 + energy ** 2) ** 2.5))
    # return 1   # for contour based on Keldysh. see Sec. 3.4


@jit(nopython=True)
def calc_G_hom(energy, params, ul=False):
    # energy: real float, energy for the Green's function
    # params: complex array, parameters of the system
    # ul: boolean, if True, G_hom_ul will be calculated
    #
    # returns 4x4 complex matrix that is the normalized homogenous solution to the Usadel equation (see. Eq. 3.31) for
    # the contour used for the integral in Eq. (3.40)

    # check, whether G_hom for left or right side shall be calculated
    if ul:
        Delta = params[9]
    else:
        Delta = params[8]

    # get relevant parameters from params
    temp = params[11]
    delta_contour = params[12]

    Delta_conj = np.conj(Delta)
    z = calc_z(energy, temp, delta_contour)
    q = 1 / (cmath.sqrt(z**2 - Delta * Delta_conj))

    return q * np.array([[z, 0, 0, Delta],
                        [0, z, -Delta, 0],
                        [0, Delta_conj, -z, 0],
                        [-Delta_conj, 0, 0, -z]], dtype=np.complex_)


@jit(nopython=True)
def calc_G12(G_hom, sqrtS, sqrtSd):
    # G_hom: 4x4 complex matrix, normalized homogeneous solution to the Usadel equation (see function above)
    # sqrtSd: complex 4x4 matrix, left matrix in Eq. 2.48
    # sqrtSd: complex 4x4 matrix, hermitian conjugate of left matrix in Eq. 2.48
    #
    # returns 4x4 complex matrix G1 or G2 from Eq. (2.51) in thesis
    # notation for G1, to get G2, you need to swap sqrtS and sqrtSd and for G1_ul simply give the function the
    # underlined parameters
    G12 = sqrtSd @ G_hom @ sqrtS

    return G12


@jit(nopython=True)
def calc_g0(G1, G2):
    # G1: complex 4x4 matrix, see function above
    # G2: complex 4x4 matrix, see function above
    #
    # returns complex 4x4 matrix g0 from Eq. (2.52) in thesis

    mat1 = np.linalg.inv(ID4 + (G1 @ G2))
    mat2 = G1 - ID4
    g0 = 2*(mat1 @ mat2) + ID4

    return g0


@jit(nopython=True)
def calc_g1(tau, g0_ul):
    # tau: complex 4x4 matrix, hopping amplitude in particle hole space
    # g0_ul: complex 4x4 matrix, g0 for the right side of the interface, see function above
    #
    # returns complex 4x4 matrix g1, see Eq. (1.53) in thesis
    return tau @ g0_ul @ dagger(tau)


@jit(nopython=True)
def calc_t_hat(g0, g1):
    # g0: complex 4x4 matrix, see calc_g0
    # g1: complex 4x4 matrix, see calc_g1
    #
    # returns complex 4x4 t-matrix, see Eq. (1.54) in thesis
    mat1 = np.linalg.inv(ID4 + (g1 @ g0))
    return mat1 @ g1


@jit(nopython=True)
def calc_g_in_g_out(g0, t_hat):
    # g0: complex 4x4 matrix, see calc_g0
    # t_hat: complex 4x4 matrix, see calc_t_hat
    #
    # returns two complex 4x4 t-matrices, see Eq. (1.55) and (1.56) in thesis
    g0p1 = g0 + ID4
    g0m1 = g0 - ID4

    g_in = g0 - (g0m1 @ t_hat @ g0p1)
    g_out = g0 - (g0p1 @ t_hat @ g0m1)

    return g_in, g_out


@jit(nopython=True)
def calc_I(g_in, g_out, sqrtS, sqrtSd):
    # g_in: complex 4x4 matrix, see calc_g_in_g_out
    # g_out: complex 4x4 matrix, see calc_g_in_g_out
    # sqrtSd: complex 4x4 matrix, left matrix in Eq. (2.48)
    # sqrtSd: complex 4x4 matrix, hermitian conjugate of left matrix in Eq. (2.48)
    #
    # returns complex 4x4 matrix current, see Eq. (1.57) in thesis
    mat1 = sqrtS @ g_out @ sqrtSd
    mat2 = sqrtSd @ g_in @ sqrtS

    return mat1 - mat2


@jit(nopython=True)
def integrand_I(k_p, energy, params):
    # kp: real float, momentum component parallel to the interface
    # energy: real float, energy for the Green's function
    # params: complex array, parameters of the system
    #
    # returns complex 4x4 matrix current I. Basically the explicit
    # implementation instead of just the definition found in the function above

    # calculate auxiliary scattering matrix (ph-space) and roots
    S, Sul = calc_S_Sul(k_p, params)
    sqrtS = calc_sqrt_S(S)
    sqrtSul = calc_sqrt_S(Sul)
    sqrtSd = dagger(sqrtS)
    sqrtSuld = dagger(sqrtSul)

    # hopping amplitudes in ph-space
    tau, tau_ul = calc_tau_tau_ul(k_p, sqrtSd, sqrtSuld, params)

    # isotropic Green's functions left and right to the interface
    G_hom = calc_G_hom(energy, params, ul=False)
    G_hom_ul = calc_G_hom(energy, params, ul=True)
    G1 = calc_G12(G_hom, sqrtS, sqrtSd)
    G2 = calc_G12(G_hom, sqrtSd, sqrtS)
    G1_ul = calc_G12(G_hom_ul, sqrtSul, sqrtSuld)
    G2_ul = calc_G12(G_hom_ul, sqrtSuld, sqrtSul)

    # auxiliary ballistic propagators
    g0 = calc_g0(G1, G2)
    g0_ul = calc_g0(G1_ul, G2_ul)
    g1 = calc_g1(tau, g0_ul)
    g1_ul = calc_g1(tau_ul, g0)

    # t-matrix
    t_hat = calc_t_hat(g0, g1)

    # incoming and outgoing ballistic propagators
    g_in, g_out = calc_g_in_g_out(g0, t_hat)

    # explicit matrix current
    I = calc_I(g_in, g_out, sqrtS, sqrtSd)

    return I


@jit(nopython=True)
def integrate_I(energy, params, points=300):
    # energy: real float, energy for the Green's function
    # params: complex array, parameters of the system
    # points: int, number of points used for integration. For most parameters 300 is sufficient
    #
    # returns complex float, value of the integral in Eq. (1.58) in thesis without any physical constants

    # get relevant parameters
    e_f = params[0]
    v1 = params[1]
    v3 = params[4]

    v_min = np.min(np.array([v1, v3]))
    k_max = cmath.sqrt(2*(e_f - v_min))

    k_p_arr = np.linspace(0, k_max, points)

    I = np.zeros((points, 4, 4), dtype=np.complex_)

    for i, kp in enumerate(k_p_arr):
        I_temp = integrand_I(kp, energy, params)
        I[i] = I_temp * kp  # 2 dimensional

    int_I = np.zeros((4, 4), dtype=np.complex_)

    # only main diagonal is integrated, everything else is set to 0
    for i in range(4):
        int_I[i, i] = np.trapz(I[:, i, i], k_p_arr)

    return int_I / (2 * np.pi)  # one factor of two pi cancels with integration over the angle


def integrand_j(energy, params, points=300):
    # energy: real float, energy for the Green's function
    # params: complex array, parameters of the system
    # points: int, number of points used for integration. For most parameters 300 is sufficient
    #
    # returns complex float, spectrum of the current density, Eq. (2.38) without the physical constants. Sign cancels
    # with the one from Eq. (1.58)

    params = tuple(params)
    G_grad_G = integrate_I(energy, params, points=points)
    mat1 = TAU3 @ G_grad_G

    return 0.125 * 2 * np.trace(mat1)


@jit(nopython=True)
def energy_domain(E_cutoff, delta, points):
    # E_cutoff: real float, boundary of the integration domain for integral in Eq. (2.40)
    # delta: complex float, order parameter
    # point: int, number of sample points in the domain
    #
    # returns real array of energies with non-uniform sample point distribution. Most of the points lie within the
    # energy gap and only few far away from the gap, where the contributions to the current density are small
    d = np.abs(delta)

    # number of points in each outer part (positive and negative energies) of the domain
    p0 = int(0.10*points)

    # number of points closely outside the energy gap, for each side
    p1 = int(0.20*points)

    # number of points within the energy gap
    p2 = points - 2 * p1 - 2 * p0

    # region 0 far away from the energy gap
    e0_neg = np.linspace(-E_cutoff, -5*d, p0+1)[:-1]

    # region 1 closely outside the energy gap
    e1_neg = np.linspace(-5*d, -d, p1+1)[:-1]

    # region within the energy gap
    e2 = np.linspace(-d, d, p2)

    # extending to positive energies
    e1_pos = -e1_neg[::-1]
    e0_pos = -e0_neg[::-1]

    e = np.concatenate((e0_neg, e1_neg, e2, e1_pos, e0_pos))

    return e


def integrand_j_enhance_peaks(energy_arr, j_spectrum, params, points_peak=20, width_peak=2):
    # energy_arr: float array, domain for the evaluation of spectrum of j
    # j_spectrum: complex array, spectrum of j evaluated at the energies from energy_arr
    # params: complex array, parameters of the system
    # points_peak: int, number of additional points around the peak
    # width: real float, width around the peak
    #
    # returns real float array, complex float array of energy and spectrum of j with additional points around the
    # maximum of the spectrum. Only need when integrating along contour E+i*delta. When integration over negative and
    # positive energies, it might be a good idea to do the same with the minimum as well.

    # index, where the maximum occurs
    i_max = np.argmax(np.abs(j_spectrum))

    # energy, where the mximum occurs
    e_max = energy_arr[i_max]

    # additional energies around the maximum
    e_max_arr = np.linspace(e_max-width_peak, e_max+width_peak, points_peak)
    j_peak = np.zeros(points_peak, dtype=np.complex_)

    # evaluate spectrum j around the peak
    for i, e in enumerate(e_max_arr):
        try:
            j_peak[i] = integrand_j(e, tuple(params))
        except:
            j_peak[i] = np.NaN
            print("Error in enhance for E = %s" % e)

    # combine original and additional values
    e_arr_all = np.concatenate((energy_arr, e_max_arr))
    j_arr_all = np.concatenate((j_spectrum, j_peak))
    ind_sorted = np.argsort(e_arr_all)

    return e_arr_all[ind_sorted], j_arr_all[ind_sorted]


def integrate_j(params, E_cutoff=30, points=300):
    # params: complex array, parameters of the system
    # E_cutoff: real float, boundary of the integration domain for integral in Eq. (2.40)
    # points: int, number of points used for the integration of j
    #
    # returns real float, current density j without the physical constants

    # get relevant parameters
    delta = params[8]
    temp = params[11]
    delta_contour = params[12]

    # init energy and j array
    energy = energy_domain(E_cutoff, delta, points)
    dj_dE = np.zeros(points, dtype=np.complex_)

    # evaluate j for each energy
    for k, e in enumerate(energy):
        integ = integrand_j(e, tuple(params), points=points)
        dj_dE[k] = integ

    # # maybe get better resolution of possible peaks
    # energy, dj_dE = integrand_j_enhance_peaks(energy, dj_dE, params)

    # get values along the contour
    z = calc_z(energy, temp, delta_contour)
    dz_de = calc_dz_de(energy, delta_contour)

    # full integrand of Eg. (2.40)
    full_integ = np.tanh(z/(2*temp)) * dj_dE * dz_de

    return np.real(np.trapz(full_integ, energy))


def calc_delta(params, phi):
    # params: complex array, parameters of the system
    # phi: real float, phase difference between the order parameters
    #
    # returns complex float, order parameter after Eq. (1.30)

    # get relevant parameters
    Tc = params[10]
    temp = params[11]

    Delta_0 = 1.764 * Tc

    if temp > 0:
        x = np.real(cmath.sqrt((Tc - temp) / temp))
        return Delta_0 * np.tanh(1.74 * x) * np.exp(1j * phi)
    if temp == 0:
        return Delta_0 * np.exp(1j * phi)


def plot_j_kernel_phi(E_cutoff, phi_arr, points, params, save_string):
    # E_cutoff: real float, boundary of the integration domain for spectrum of j
    # phi_arr: real float array, phase differences of the order parameters. One plot for each
    # points: int, number of points of the plot
    # params: complex array, parameters of the system
    # save_string: string or not string, used to name saved data. if not string, nothing will be saved.
    #
    # returns nothing, plots spectrum of j for given params and saves data. Used for Fig. 3.2

    # get relevant parameters
    Delta = params[8]

    # number of points used in integrand_j_enhance peaks
    points_enhance = 200

    # init arrays
    energy_arr = np.zeros((len(phi_arr), points))
    energy_arr_enhanced = np.zeros((len(phi_arr), points+points_enhance))
    dj_dE = np.zeros((len(phi_arr), points), dtype=np.complex_)
    dj_dE_enhanced = np.zeros((len(phi_arr), points+points_enhance), dtype=np.complex_)

    # evaluate spectrum of j for each phi
    for i, phi in enumerate(phi_arr):
        Delta_ul = calc_delta(params, phi)
        params = update_params(params, 9, Delta_ul)

        energy_arr[i] = energy_domain(E_cutoff, Delta, points)

        for k, e in enumerate(energy_arr[i]):
            integ = integrand_j(e, tuple(params))
            dj_dE[i, k] = integ

        # energy_arr_enhanced[i], dj_dE_enhanced[i] = integrand_j_enhance_peaks(energy_arr[i], dj_dE[i], params, points_peak=points_enhance)
        print("phi = ", phi)

    plt.figure(figsize=(20, 11))
    plt.tight_layout()

    # plot the data
    for i in range(len(phi_arr)):
        plt.plot(energy_arr[i], np.real(dj_dE[i]), label="phi = " + str(np.round(phi_arr[i], 2)))
    plt.xlabel("E", fontsize=20)
    plt.ylabel("j_kernel", fontsize=20)
    plt.title(
        '$E_F = %s, V_{1} = %s, V_{2,u} = %s, V_{2,d} = %s, V_{3} = %s, d = %s, T = %s$ \nFull params %s' % (
            params[0], params[2], params[3], params[4], params[5], params[7], params[11], np.round(params, 2)), fontsize=10)
    plt.legend(fontsize=15)
    plt.show()

    # save data
    if isinstance(save_string, str):
        np.save(save_string + "_j_kernel_array", dj_dE)
        np.save(save_string + "_phi_array", phi_arr)
        np.save(save_string + "_energy_array", energy_arr)
        plt.savefig(save_string + "_plot")


def calc_points(params):
    # params: complex array, parameters of the system
    #
    # returns number of points used for the integral in Eg. (2.40). Low temperatures require more points than high
    # temperatures

    # get relevant params
    T_c = params[10]
    T = params[11]

    t = T / T_c

    # values found empirically
    if t < 0.05:
        return 1000
    elif t < 0.21:
        return int(-5000*t + 1250)
    elif t < 1:
        return 200
    else:
        return 1


def plot_j_phi(parameter, parameter_index, parameter_string, phi_arr, params, save_string=False):
    # parameter: real or complex float array, values of the parameter of which will be iterated (most often temperature)
    # parameter_index: int, index in params
    # parameter_string: string, name used for naming files, plots
    # phi_arr: real float array, phase differences of the order parameters used for finding the critical current density
    # params: complex array, parameters of the system
    # save_string: string or not string, used to name saved data. if not string, nothing will be saved.
    #
    # returns nothing, plots of j(phi) for given params and saves data. Used for Fig. 3.3

    # init j array
    j_phi = np.zeros((len(parameter), len(phi_arr)))

    # iterate over parameter values
    for k, p in enumerate(parameter):
        params = update_params(params, parameter_index, p)

        Delta = calc_delta(params, 0)
        params = update_params(params, 8, Delta)

        points = calc_points(params)

        # iterate over phase differences
        for l , phi in enumerate(phi_arr):
            Delta_ul = calc_delta(params, phi)
            params = update_params(params, 9, Delta_ul)

            j_phi[k, l] = integrate_j(params, points=points)
            print("p = % s, phi = % s" % (p, phi))


    print("Total time: ", time()-START_TIME, "s")

    # plot values
    plt.figure(figsize=(20, 11))
    plt.tight_layout()
    for k, p in enumerate(parameter):
        plt.plot(phi_arr/np.pi, j_phi[k], label=parameter_string + '=%s' % np.round(p, 2))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$phi/pi$', fontsize=20)
    plt.ylabel('j', fontsize=20)
    plt.title(
        '$E_F = %s, V_{1} = %s, V_{2,u} = %s, V_{2,d} = %s, V_{3} = %s, d = %s$ \nFull params %s' % (
            params[0], params[1], params[2], params[3], params[4], params[6], np.round(params, 2)), fontsize=10)
    plt.legend(fontsize=15)

    # save values
    if isinstance(save_string, str):
        np.save(save_string + "_j_array", j_phi)
        np.save(save_string + "_phi_array", phi_arr)
        np.save(save_string + "_" + parameter_string + "_array", parameter)
        # plt.savefig(save_string + "_plot")

    plt.show()


def main():
    ###### initial parameters #######
    E_F = 1  # Fermi energy
    V1 = 0
    V2u =  E_F + 0.01
    V2d = 17
    V3 = V1 + 0

    k0 = calc_k(V1, E_F, 1, 0) #length scale for scattering matrix
    kappa0_u = calc_k(V2u, E_F, k0, 0)
    kappa0_d = calc_k(V2d, E_F, k0, 0)
    width_u = 0.05 / kappa0_u
    width_d = 2.4 / kappa0_d

    Tc = 1
    Temp = 0.3 * Tc

    Delta_init = 1.764 * Tc
    Delta_ul_init = 1.764j * Tc

    delta_contour = 1

    # PARAMS
    #         0    1   2    3    4   5   6        7        8           9              10  11    12
    params = [E_F, V1, V2u, V2d, V3, k0, width_u, width_d, Delta_init, Delta_ul_init, Tc, Temp, delta_contour]

    save_string = "run_5"


    #####dimensions######
    # length of coresponding arrays
    DIM_barrier_width = 300
    DIM_phi = 45
    DIM_V2d = 25
    DIM_temp_low = 18
    DIM_temp_high = 35

    ######arrays for varying parameter
    phi_arr = np.linspace(0 * np.pi, 1 * np.pi, DIM_phi)
    delta_contour_arr = np.linspace(0.001, 1, 10)
    d_arr = np.linspace(0.001, 2, DIM_barrier_width)

    Temp_arr_low = np.linspace(0.005*Tc, 0.16*Tc, DIM_temp_low, endpoint=False)
    Temp_arr_high = np.linspace(0.16*Tc, 0.999*Tc, DIM_temp_high)

    Temp_arr_low = np.array([0.01, 0.09])
    Temp_arr_high = np.array([0.28, 0.53, 0.78, 0.95])

    Temp_arr = np.concatenate((Temp_arr_low, Temp_arr_high))
    V2d_arr = np.logspace(np.log10(V2u), 4, DIM_V2d)
    width_d_arr = np.linspace(0.001, 8, DIM_barrier_width) / kappa0_d

    # PLOTS
    print(save_string)
    plot_j_phi(width_d_arr, 7, "T", phi_arr, params, save_string)
    # plot_j_kernel_phi(2.5, phi_arr, 1000, params, save_string)

    
main()