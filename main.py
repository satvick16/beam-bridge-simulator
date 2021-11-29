from math import pi
import matplotlib.pyplot as plt
import numpy as np

from numpy.core.numeric import full
from numpy.matrixlib.defmatrix import bmat

n = 1280  # number of locations to evaluate bridge failure
L = 1280  # length of bridge in mm
portion_extra_len_on_left = 1/3
support_1_x = int((L-1280)*(1/3) + 30/2)  # currently 15
support_2_x = int(support_1_x + 1060)  # currently 1075

# support_1_x = 0
# support_2_x = 1060

x = np.linspace(0, L, num=L)
SFD_PL = np.zeros(L)

# 2. Define cross-sections
# There are many (more elegant ways) to construct cross-section objects
# From sample code
bft = np.full(L, 100)  # Top Flange Width
# tft = np.full(L, 2.54)  # Top Flange Thickness
tft = np.concatenate((np.full(534, 2.54), np.full(
    60, 3.81), np.full(207, 2.54), np.full(1280-801, 2.54)))
hw = np.full(L, 85)  # web height
tw = np.full(L, 1.27)  # Web Thickness (Assuming 2 separate webs)
bfb = np.full(L, 75)  # Bottom Flange Width

tfb = np.concatenate((np.full(801, 1.27), np.full(L-801, 2.54+1.27)))
tfb = np.concatenate((np.full(801, 1.27), np.full(
    1045-801, 3.81), np.full(60, 5.08), np.full(1280-1105, 3.81)))


diaphram_width = 1
# vector indicating whether there is a diaphram here or not
a = np.full(L, False)
a[0:diaphram_width] = True
a[30:30 + diaphram_width] = True
a[L-diaphram_width:L] = True
a[L-diaphram_width - 30:L - 30] = True

# for i in range(320, L, 320):
#     a[i:i+diaphram_width] = True

a[535] = True
a[565] = True
a[1045] = True
a[1075] = True

# 3. Define Material Properties

Sigma_T = 30
Sigma_C = 6
E = 4000
TauU = 4
TauG = 2
mu = 0.2

# applies a point load at location xP and returns the SFD and BMD associated with it


def apply_pl(xP, P, x, SFD):
    # calculating reaction forces
    sup_2_p = (xP - support_1_x)*P / (support_2_x - support_1_x)
    sup_1_p = P - sup_2_p

    # updating the SFD diagram
    SFD[support_1_x:] += sup_1_p
    SFD[xP:] -= P
    SFD[support_2_x:] += sup_2_p

    # calculating the BMD
    BMD = np.cumsum(SFD)
    return SFD, BMD


def calc_section_properties(bft, tft, hw, tw, bfb, tfb, a):
    # calculating y bar
    # top flange area + bottom flange area + web areas
    total_area = bft * tft + bfb * tfb + 2 * hw * tw

    # ( top flange y_bar * top flange area + bottom flange y_bar * bottom flange area + 2 * web y_bar + web area ) / total_area
    y_bar = ((tfb+hw+0.5*tft) * bft*tft + (0.5*tfb) * bfb *
             tfb + 2 * (tfb+0.5*hw) * hw*tw) / total_area

    I = np.array((bft*tft**3)/12 + bft*tft*((tfb+hw+0.5*tft) - y_bar)**2 +  # top flange
                 (bfb*tfb**3)/12 + bfb*tfb*((0.5*tfb) - y_bar)**2 +  # bottom flange
                 2 * np.array([(tw*hw**3)/12 + tw*hw*((tfb+0.5*hw) - y_bar)**2]))  # web members
    I = np.reshape(I, (I.shape[1]))

    # this calculation assumes that y_bar is cutting through the web members
    # Q = A_bottom*d_bottom + A_web*d_web + A_web*d_web
    A_bottom = abs(bfb*tfb)
    d_bottom = abs(y_bar - (tfb/2))

    A_web = abs((y_bar - tfb) * tw)
    d_web = abs(y_bar - tfb)/2

    Q = A_bottom*d_bottom + A_web*d_web + A_web*d_web

    return y_bar, I, Q


def v_fail(I, Q, b, tau):
    return tau*I*b / Q  # shear force that would cause bridge to fail in shear


def V_fail_buckling(I, Q, tw, hw, E, mu, a):
    dist_to_diaphragm = np.ones(len(a))

    start_point = 0
    for i in range(len(a)):
        # find distance to diaphragm (both sides)
        # add distances
        if a[i] == True:
            dist_to_diaphragm[start_point+1:i] = i-start_point
            start_point = i

    case_4_fail_stress = (5*(pi**2)*E) / (12*(1-mu**2)) * \
        ((tw/dist_to_diaphragm)**2 + (tw/hw)**2)

    case_4_fail_force = case_4_fail_stress * I * \
        (tw*2) / Q  # using tau = VQ/Ib, where b = tw*2

    return case_4_fail_force


def glue_V_fail_buckling(I, tw, hw):
    glue_tab_width = 10

    Q_glue = bft*tft*((hw+tft+tfb)-(tw/2)-y_bar)

    case_4_fail_force = 2 * I * (glue_tab_width*2) / Q_glue

    return case_4_fail_force


def M_fail_Mat(sigma_T, sigma_C, BMD, I, y_bar, tft, hw, tfb):
    total_height = tft + hw + tfb
    fail_T = np.zeros(len(BMD))
    fail_C = np.zeros(len(BMD))

    for i in range(len(BMD)):
        if BMD[i] > 0:
            fail_T[i] = sigma_T*I[i] / (y_bar[i])
            fail_C[i] = sigma_C*I[i] / (total_height[i] - y_bar[i])
        else:
            fail_T[i] = sigma_T*I[i] / -(total_height[i] - y_bar[i])
            fail_C[i] = sigma_C*I[i] / -y_bar[i]

    return fail_T, fail_C


def M_fail_Buck(sigma_C, E, mu, BMD, I, y_bar, tft, hw, tfb, bfb, bft, a, Q, tw):
    total_height = tft + hw + tfb

    fail_buck_1 = np.zeros(len(BMD))
    fail_buck_2 = np.zeros(len(BMD))
    fail_buck_3 = np.zeros(len(BMD))

    for i in range(len(BMD)):
        if BMD[i] > 0:
            # sigma crit for: flange buckling, flange tip buckling, web buckling
            sigma_case_1 = ((4*pi**2*E) / (12*(1-mu**2))) * \
                (tft[i] / bfb[i])**2
            sigma_case_2 = ((0.425*pi**2*E) / (12*(1-mu**2))) * \
                (tft[i] / (0.5*(bft[i]-bfb[i])))**2
            sigma_case_3 = ((6*pi**2*E) / (12*(1-mu**2))) * \
                (tw[i] / (total_height[i] - 1.27 - y_bar[i]))**2

            fail_buck_1[i] = sigma_case_1*I[i] / (total_height[i] - y_bar[i])
            fail_buck_2[i] = sigma_case_2*I[i] / (total_height[i] - y_bar[i])
            fail_buck_3[i] = sigma_case_3*I[i] / \
                (total_height[i] - 1.27 - y_bar[i])
        else:
            # sigma crit for: flange buckling, flange tip buckling, web buckling
            sigma_case_1 = ((4*pi**2*E) / (12*(1-mu**2))) * \
                (tfb[i] / bfb[i])**2
            sigma_case_2 = 0
            sigma_case_3 = ((6*pi**2*E) / (12*(1-mu**2))) * \
                (tw[i] / (y_bar[i] - 1.27))**2

            fail_buck_1[i] = sigma_case_1*I[i] / -(y_bar[i])
            fail_buck_2[i] = None
            fail_buck_3[i] = sigma_case_3*I[i] / -(y_bar[i] - 1.27)

    return fail_buck_1, fail_buck_2, fail_buck_3


def fail_load(P, SFD, BMD, V_Mat, V_Buck, M_MatT, M_MatC, M_Buck1, M_Buck2, M_Buck3):
    min_fail_moment = np.zeros(len(BMD))
    min_fail_shear = np.zeros(len(BMD))

    for i in range(len(BMD)):
        min_fail_moment[i] = min(abs(M_MatT[i]), abs(M_MatC[i]),
                                 abs(M_Buck1[i]), abs(M_Buck2[i]), abs(M_Buck3[i]))

    for i in range(len(BMD)):
        min_fail_shear[i] = min(abs(V_Mat[i]), abs(V_Buck[i]))

    moment_ratio_to_fail = np.empty(len(BMD))
    moment_ratio_to_fail[:] = np.nan
    max_force_moment = np.zeros(len(BMD))
    for i in range(len(BMD)):
        # failure moment/applied moment
        if abs(BMD[i]) > 10:
            moment_ratio_to_fail[i] = min_fail_moment[i]/BMD[i]
        # debugging
        # print(moment_ratio_to_fail[i])
        max_force_moment[i] = P * moment_ratio_to_fail[i]

    shear_ratio_to_fail = np.empty(len(SFD))
    shear_ratio_to_fail[:] = np.nan
    max_force_shear = np.zeros(len(SFD))
    for i in range(len(SFD)):
        # failure shear/applied shear
        if abs(SFD[i]) > 10:
            shear_ratio_to_fail[i] = min_fail_shear[i]/SFD[i]
        max_force_shear[i] = P * shear_ratio_to_fail[i]

    return max_force_shear, max_force_moment


def fail_load_2(P, SFD_PL, BMD, V_Mat, V_Buck, M_MatT, M_MatC, M_Buck1, M_Buck2, M_Buck3, V_glue):
    load = 1
    counter = 0

    has_failed = False

    while not has_failed:
        counter += 1
        SFD_PL, BMD = apply_pl(565, load, x, SFD_PL)
        SFD_PL, BMD = apply_pl(1265, load, x, SFD_PL)

        fails_at = []

        for i in range(len(BMD)):
            if abs(abs(BMD[i]) >= abs(M_MatT[i])):
                fails_at.append(f"{i} mm -- material tension moment")
                has_failed = True
            elif abs(abs(BMD[i]) >= abs(M_MatC[i])):
                fails_at.append(f"{i} mm -- material compression moment")
                has_failed = True
            elif abs(abs(BMD[i]) >= abs(M_Buck1[i])):
                fails_at.append(f"{i} mm -- flange buckling (case 1)")
                has_failed = True
            elif abs(abs(BMD[i]) >= abs(M_Buck2[i])):
                fails_at.append(f"{i} mm -- flange tip buckling (case 2)")
                has_failed = True
            elif abs(abs(BMD[i]) >= abs(M_Buck3[i])):
                fails_at.append(f"{i} mm -- web buckling (case 3)")
                has_failed = True
            elif abs(abs(SFD_PL[i]) >= abs(V_Mat[i])):
                fails_at.append(f"{i} mm -- material shear")
                has_failed = True
            elif abs(abs(SFD_PL[i]) >= abs(V_Buck[i])):
                fails_at.append(f"{i} mm -- shear buckling (case 4)")
                has_failed = True
            elif abs(abs(SFD_PL[i]) >= abs(V_glue[i])):
                fails_at.append(f"{i} mm -- glue shear")
                has_failed = True

    max_point_load = P + counter

    return max_point_load, fails_at


def plot_sfd_bmd(SFD, BMD):
    plt.subplot(2, 1, 1)
    plt.plot(x, SFD_PL)
    plt.xlabel("length of bridge (mm)")
    plt.ylabel("Shear Force (N)")
    plt.title("Shear Force Diagram")

    plt.subplot(2, 1, 2)
    plt.plot(x, BMD)
    plt.xlabel("length of bridge (mm)")
    plt.ylabel("Bending Moment (N*mm)")
    plt.title("Bending Moment Diagram")

    plt.subplots_adjust(left=0.17, hspace=0.6)
    plt.gca().invert_yaxis()
    plt.show()


def plot_M_fail_mat(BMD, ax1):
    y_bar, I, Q = calc_section_properties(bft, tft, hw, tw, bfb, tfb, a)
    fail_t, fail_c = M_fail_Mat(Sigma_T, Sigma_C, BMD, I, y_bar, tft, hw, tfb)

    ax1.plot(x, fail_t, label="tension")
    ax1.plot(x, BMD, label="bmd")
    ax1.plot(x, fail_c, label="compression")
    ax1.invert_yaxis()
    ax1.legend()
    ax1.set_xlabel("x-position (mm)")
    ax1.set_ylabel("moment (Nmm)")
    ax1.set_title("matboard moment failure", fontsize=10)


def plot_V_fail_mat(SFD_PL, ax):
    y_bar, I, Q = calc_section_properties(bft, tft, hw, tw, bfb, tfb, a)
    fail = v_fail(I, Q, 2*tw, TauU)

    ax.plot(x, fail, label="global regular tau")
    ax.plot(x, SFD_PL, label="sfd")
    ax.legend()
    ax.set_xlabel("x-position (mm)")
    ax.set_ylabel("shear force (N)")
    ax.set_title("matboard shear failure", fontsize=10)


def plot_v_fail_buckling(SFD, ax):
    y_bar, I, Q = calc_section_properties(bft, tft, hw, tw, bfb, tfb, a)
    case_4_fail_force = V_fail_buckling(I, Q, tw, hw, E, mu, a)

    ax.plot(x, case_4_fail_force, label="shear buckling (case 4a)")
    ax.plot(x, SFD, label="shear force")
    ax.legend(loc=6)
    ax.set_ylim(-2500, 2500)
    ax.legend()
    ax.set_xlabel("x-position (mm)")
    ax.set_ylabel("shear force (N)")
    ax.set_title("shear buckling", fontsize=10)


def plot_glue_V_fail_buckling(SFD_PL, I, tw, hw, ax):
    fail_force = glue_V_fail_buckling(I, tw, hw)

    ax.plot(x, fail_force, label="glue shear buckling (case 4b)")
    ax.plot(x, SFD_PL, label="sfd")
    ax.legend()
    ax.set_xlabel("x-position (mm)")
    ax.set_ylabel("shear force (N)")
    ax.set_title("glue shear buckling", fontsize=10)


# def plot_max_forces(max_force_shear, max_force_moment):
#     plt.plot(x, max_force_shear, label="Failure force in shear")
#     plt.plot(x, max_force_moment, label="Failure force due to moment")
#     plt.ylabel("Force (N)")
#     plt.xlabel("Position (mm)")
#     plt.legend()
#     plt.ylim(-100000, 100000)

#     # plt.plot(x, BMD)
#     # plt.plot(x, SFD_PL)

#     plt.gca().invert_yaxis()
#     plt.show()


def plot_moment_failure(mt, mc, c1, c2, c3, BMD, ax):
    # plt.plot(x, mt, label="Max moment in tension")
    # plt.plot(x, mc, label="Max moment in compression")
    ax.plot(x, c1, label="flange buckling Case 1")
    ax.plot(x, c2, label="flange tip buckling Case 2")
    ax.plot(x, c3, label="web buckling Case 3")
    ax.plot(x, BMD, label="bmd")
    ax.legend()
    ax.set_xlabel("x-position (mm)")
    ax.set_ylabel("moment (Nmm)")
    ax.set_title("thin plate buckling", fontsize=10)
    ax.invert_yaxis()


def train_loading_case_end(SFD_PL):
    train_load = 400/6

    SFD_PL, BMD = apply_pl(409, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(585, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(749, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(925, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(1089, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(1265, train_load, x, SFD_PL)

    return SFD_PL, BMD


def train_loading_case_mid(SFD_PL):
    train_load = 400/6

    SFD_PL, BMD = apply_pl(117, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(293, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(457, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(633, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(797, train_load, x, SFD_PL)
    SFD_PL, BMD = apply_pl(973, train_load, x, SFD_PL)

    return SFD_PL, BMD


def get_area_used():
    sum = 2*hw*(tw/1.27) + bfb*(tfb/1.27) + bft*(tft/1.27) + np.full(L, 20)
    diaphram_area = 0
    for i in range(len(a)):
        if a[i]:
            diaphram_area += (bfb[i]-tw[i]*2)*hw[i]
    return int(np.cumsum(sum)[-1]) + diaphram_area


print(get_area_used())

# def deflections(x, BMD, I, E):
#     curvature = BMD / E / I

#     x_bar_ba = (5 / 8) * 550
#     d_ba = np.trapz(curvature[15:550]) * x_bar_ba

#     x_bar_cb = (5 / 8) * (1280-mid)
#     d_ca = d_ba + np.trapz(curvature[mid:1265]) * x_bar_cb

#     delta_mid = ((mid * d_ca) / (2 * mid)) - d_ba

#     return delta_mid


point_load = 200
SFD_PL, BMD = apply_pl(565, point_load, x, SFD_PL)
SFD_PL, BMD = apply_pl(1265, point_load, x, SFD_PL)
# SFD_PL, BMD = train_loading_case_end(SFD_PL)
# SFD_PL, BMD = train_loading_case_mid(SFD_PL)

y_bar, I, Q = calc_section_properties(bft, tft, hw, tw, bfb, tfb, a)

v_mat = v_fail(I, Q, 2.54, TauU)
v_buck = V_fail_buckling(I, Q, tw, hw, E, mu, a)

mt, mc = M_fail_Mat(Sigma_T, Sigma_C, BMD, I, y_bar, tft, hw, tfb)
c1, c2, c3 = M_fail_Buck(Sigma_C, E, mu, BMD, I, y_bar,
                         tft, hw, tfb, bfb, bft, a, Q, tw)

max_force_shear, max_force_moment = fail_load(
    point_load, SFD_PL, BMD, v_mat, v_buck, mt, mc, c1, c2, c3)

glue = glue_V_fail_buckling(I, tw, hw)

# plot_sfd_bmd(SFD_PL, BMD)
# plot_max_forces(max_force_shear, max_force_moment)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

fig.tight_layout()

plot_M_fail_mat(BMD, ax1)
plot_v_fail_buckling(SFD_PL, ax2)
plot_moment_failure(mt, mc, c1, c2, c3, BMD, ax3)
plot_V_fail_mat(SFD_PL, ax4)
plot_glue_V_fail_buckling(SFD_PL, I, tw, hw, ax5)

plt.show()

max_point_load, failures = fail_load_2(
    point_load, SFD_PL, BMD, v_mat, v_buck, mt, mc, c1, c2, c3, glue)
print(max_point_load)
print(failures)

# d_mid = deflections(x, BMD, I, E)
# print(d_mid)
