import builtins
import numpy as np
import symengine as se
from importlib import import_module
from core.cbfs.cbf_wrappers import symbolic_cbf_wrapper_singleagent
from core.mathematics.symbolic_functions import ramp

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
mod = "models." + vehicle + "." + control_level + ".system"

# Programmatic import
try:
    module = import_module(mod)
    globals().update({"f": getattr(module, "f")})
    globals().update({"ss": getattr(module, "xs")})
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e

# Defining Physical Params
R = 1.0
cx1 = -2.0
cy1 = 0.0
cx2 = 1.25
cy2 = -1.25

# Define Physical Distances
dx1 = ss[0] - cx1
dy1 = ss[1] - cy1
dx2 = ss[0] - cx2
dy2 = ss[1] - cy2

# Define more quantities
phidot = f(np.zeros((len(ss),)), True)[6]
thedot = f(np.zeros((len(ss),)), True)[7]

# Speed CBF Symbolic
h_oa1_symbolic = dx1**2 + dy1**2 - R**2
h_oa2_symbolic = dx2**2 + dy2**2 - R**2
h_oa3_symbolic = 5 * ss[2] + f(np.zeros((len(ss),)), True)[2]
h_oa4_symbolic = 10 * (
    -phidot * se.sin(ss[6]) * se.cos(ss[7])
    - thedot * se.cos(ss[6]) * se.sin(ss[7])
    + (se.cos(ss[6]) * se.cos(ss[7]) - np.cos(np.pi / 2) * 0.1)
)
# h_oa4_symbolic = -10 * (
#     phidot * (se.sin(ss[6]) * se.cos(ss[7]))
#     + thedot * (se.cos(ss[7]) + se.sin(ss[6]) * se.cos(ss[7]) + se.cos(ss[6]) * se.sin(ss[7]))
#     + (se.sin(ss[7]) - se.sin(ss[6]) * se.cos(ss[7]) - se.cos(ss[6]) * se.cos(ss[7]))
# )

dhdx_oa1_symbolic = (se.DenseMatrix([h_oa1_symbolic]).jacobian(se.DenseMatrix(ss))).T
dhdx_oa2_symbolic = (se.DenseMatrix([h_oa2_symbolic]).jacobian(se.DenseMatrix(ss))).T
dhdx_oa3_symbolic = (se.DenseMatrix([h_oa3_symbolic]).jacobian(se.DenseMatrix(ss))).T
dhdx_oa4_symbolic = (se.DenseMatrix([h_oa4_symbolic]).jacobian(se.DenseMatrix(ss))).T

d2hdx2_oa1_symbolic = dhdx_oa1_symbolic.jacobian(se.DenseMatrix(ss))
d2hdx2_oa2_symbolic = dhdx_oa2_symbolic.jacobian(se.DenseMatrix(ss))
d2hdx2_oa3_symbolic = dhdx_oa3_symbolic.jacobian(se.DenseMatrix(ss))
d2hdx2_oa4_symbolic = dhdx_oa4_symbolic.jacobian(se.DenseMatrix(ss))

h_oa1_func = symbolic_cbf_wrapper_singleagent(h_oa1_symbolic, ss)
h_oa2_func = symbolic_cbf_wrapper_singleagent(h_oa2_symbolic, ss)
h_oa3_func = symbolic_cbf_wrapper_singleagent(h_oa3_symbolic, ss)
h_oa4_func = symbolic_cbf_wrapper_singleagent(h_oa4_symbolic, ss)

dhdx_oa1_func = symbolic_cbf_wrapper_singleagent(dhdx_oa1_symbolic, ss)
dhdx_oa2_func = symbolic_cbf_wrapper_singleagent(dhdx_oa2_symbolic, ss)
dhdx_oa3_func = symbolic_cbf_wrapper_singleagent(dhdx_oa3_symbolic, ss)
dhdx_oa4_func = symbolic_cbf_wrapper_singleagent(dhdx_oa4_symbolic, ss)

d2hdx2_oa1_func = symbolic_cbf_wrapper_singleagent(d2hdx2_oa1_symbolic, ss)
d2hdx2_oa2_func = symbolic_cbf_wrapper_singleagent(d2hdx2_oa2_symbolic, ss)
d2hdx2_oa3_func = symbolic_cbf_wrapper_singleagent(d2hdx2_oa3_symbolic, ss)
d2hdx2_oa4_func = symbolic_cbf_wrapper_singleagent(d2hdx2_oa4_symbolic, ss)

# Tau Formulation for PCA-CBF
dvx = f(np.zeros((len(ss),)), True)[0]
dvy = f(np.zeros((len(ss),)), True)[1]

tau_sym = se.Symbol("tau", real=True)

# tau* for computing tau
epsilon = 1e-3
tau1_star_symbolic = -(dx1 * dvx + dy1 * dvy) / (dvx**2 + dvy**2 + epsilon)
dtau1stardx_symbolic = (se.DenseMatrix([tau1_star_symbolic]).jacobian(se.DenseMatrix(ss))).T
d2tau1stardx2_symbolic = dtau1stardx_symbolic.jacobian(se.DenseMatrix(ss))
tau1_star = symbolic_cbf_wrapper_singleagent(tau1_star_symbolic, ss)
dtau1stardx = symbolic_cbf_wrapper_singleagent(dtau1stardx_symbolic, ss)
d2tau1stardx2 = symbolic_cbf_wrapper_singleagent(d2tau1stardx2_symbolic, ss)

tau2_star_symbolic = -(dx2 * dvx + dy2 * dvy) / (dvx**2 + dvy**2 + epsilon)
dtau2stardx_symbolic = (se.DenseMatrix([tau2_star_symbolic]).jacobian(se.DenseMatrix(ss))).T
d2tau2stardx2_symbolic = dtau2stardx_symbolic.jacobian(se.DenseMatrix(ss))
tau2_star = symbolic_cbf_wrapper_singleagent(tau2_star_symbolic, ss)
dtau2stardx = symbolic_cbf_wrapper_singleagent(dtau2stardx_symbolic, ss)
d2tau2stardx2 = symbolic_cbf_wrapper_singleagent(d2tau2stardx2_symbolic, ss)

# tau for computing PCA-CBF
Tmax = 10.0
kh = 1000.0
tau_star_sym = se.Symbol("tau_star", real=True)
tau_symbolic = tau_star_sym * ramp(tau_star_sym, kh, 0.0) - (tau_star_sym - Tmax) * ramp(
    tau_star_sym, kh, Tmax
)
dtaudtaustar_symbolic = se.diff(tau_symbolic, tau_star_sym)
d2taudtaustar2_symbolic = se.diff(dtaudtaustar_symbolic, tau_star_sym)
tau = symbolic_cbf_wrapper_singleagent(tau_symbolic, [tau_star_sym])
dtaudtaustar = symbolic_cbf_wrapper_singleagent(dtaudtaustar_symbolic, [tau_star_sym])
d2taudtaustar2 = symbolic_cbf_wrapper_singleagent(d2taudtaustar2_symbolic, [tau_star_sym])

# Predictive Obstacle Avoidance CBF1
h1_predictive_oa_symbolic = (dx1 + tau_sym * dvx) ** 2 + (dy1 + tau_sym * dvy) ** 2 - R**2
dhdx1_predictive_oa_symbolic = (
    se.DenseMatrix([h1_predictive_oa_symbolic]).jacobian(se.DenseMatrix(ss))
).T
dh1dtau_predictive_oa_symbolic = se.diff(h1_predictive_oa_symbolic, tau_sym)
d2h1dx2_predictive_oa_symbolic = dhdx1_predictive_oa_symbolic.jacobian(se.DenseMatrix(ss))
d2h1dtau2_predictive_oa_symbolic = se.diff(dh1dtau_predictive_oa_symbolic, tau_sym)
d2h1dtaudx_predictive_oa_symbolic = (
    se.DenseMatrix([dh1dtau_predictive_oa_symbolic]).jacobian(se.DenseMatrix(ss))
).T
h1_predictive_oa = symbolic_cbf_wrapper_singleagent(h1_predictive_oa_symbolic, ss)
dh1dx_predictive_oa = symbolic_cbf_wrapper_singleagent(dhdx1_predictive_oa_symbolic, ss)
dh1dtau_predictive_oa = symbolic_cbf_wrapper_singleagent(dh1dtau_predictive_oa_symbolic, ss)
d2h1dx2_predictive_oa = symbolic_cbf_wrapper_singleagent(d2h1dx2_predictive_oa_symbolic, ss)
d2h1dtaudx_predictive_oa = symbolic_cbf_wrapper_singleagent(d2h1dtaudx_predictive_oa_symbolic, ss)
d2h1dtau2_predictive_oa = symbolic_cbf_wrapper_singleagent(d2h1dtau2_predictive_oa_symbolic, ss)

# Predictive Obstacle Avoidance CBF2
h2_predictive_oa_symbolic = (dx2 + tau_sym * dvx) ** 2 + (dy2 + tau_sym * dvy) ** 2 - R**2
dhdx2_predictive_oa_symbolic = (
    se.DenseMatrix([h2_predictive_oa_symbolic]).jacobian(se.DenseMatrix(ss))
).T
dh2dtau_predictive_oa_symbolic = se.diff(h2_predictive_oa_symbolic, tau_sym)
d2h2dx2_predictive_oa_symbolic = dhdx2_predictive_oa_symbolic.jacobian(se.DenseMatrix(ss))
d2h2dtau2_predictive_oa_symbolic = se.diff(dh2dtau_predictive_oa_symbolic, tau_sym)
d2h2dtaudx_predictive_oa_symbolic = (
    se.DenseMatrix([dh2dtau_predictive_oa_symbolic]).jacobian(se.DenseMatrix(ss))
).T
h2_predictive_oa = symbolic_cbf_wrapper_singleagent(h2_predictive_oa_symbolic, ss)
dh2dx_predictive_oa = symbolic_cbf_wrapper_singleagent(dhdx2_predictive_oa_symbolic, ss)
dh2dtau_predictive_oa = symbolic_cbf_wrapper_singleagent(dh2dtau_predictive_oa_symbolic, ss)
d2h2dx2_predictive_oa = symbolic_cbf_wrapper_singleagent(d2h2dx2_predictive_oa_symbolic, ss)
d2h2dtaudx_predictive_oa = symbolic_cbf_wrapper_singleagent(d2h2dtaudx_predictive_oa_symbolic, ss)
d2h2dtau2_predictive_oa = symbolic_cbf_wrapper_singleagent(d2h2dtau2_predictive_oa_symbolic, ss)

# Relaxed Predictive Collision Avoidance
relaxation = 0.1  # Quadrotor Simulation


def h0_oa1(ego):
    return h_oa1_func(ego)


def h0_oa2(ego):
    return h_oa2_func(ego)


def dh0dx_oa1(ego):
    ret = dhdx_oa1_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def dh0dx_oa2(ego):
    ret = dhdx_oa2_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2h0dx2_oa1(ego):
    ret = d2hdx2_oa1_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2h0dx2_oa2(ego):
    ret = d2hdx2_oa2_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def h_poa1(ego):
    func = h1_predictive_oa(ego)

    try:
        ret = func.subs({tau_sym: tau([tau1_star(ego)])})
    except AttributeError:
        ret = func

    return ret


def h_poa2(ego):
    func = h2_predictive_oa(ego)

    try:
        ret = func.subs({tau_sym: tau([tau2_star(ego)])})
    except AttributeError:
        ret = func

    return ret


def dhdx_poa1(ego):
    func1 = dh1dx_predictive_oa(ego)
    func2 = dh1dtau_predictive_oa(ego)

    try:
        ret1 = func1.subs({tau_sym: tau([tau1_star(ego)])})
    except AttributeError:
        ret1 = func1

    try:
        ret2 = func2.subs({tau_sym: tau([tau1_star(ego)])})
    except AttributeError:
        ret2 = func2

    ret = ret1 + ret2 * dtaudtaustar([tau1_star(ego)]) * dtau1stardx(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx_poa2(ego):
    func1 = dh2dx_predictive_oa(ego)
    func2 = dh2dtau_predictive_oa(ego)

    try:
        ret1 = func1.subs({tau_sym: tau([tau2_star(ego)])})
    except AttributeError:
        ret1 = func1

    try:
        ret2 = func2.subs({tau_sym: tau([tau2_star(ego)])})
    except AttributeError:
        ret2 = func2

    ret = ret1 + ret2 * dtaudtaustar([tau2_star(ego)]) * dtau2stardx(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


# Necessary for stochastic systems
def d2hdx2_poa1(ego):
    func1 = d2h1dx2_predictive_oa(ego)
    func2 = dh1dtau_predictive_oa(ego)

    try:
        ret1 = func1.subs({tau_sym: tau([tau1_star(ego)])})
    except AttributeError:
        ret1 = func1

    try:
        ret2 = func2.subs({tau_sym: tau([tau1_star(ego)])})
    except AttributeError:
        ret2 = func2

    d2hdx2_eval = ret1
    dtaustardx_eval = dtau1stardx(ego)
    dtaudtaustar_eval = dtaudtaustar([tau1_star(ego)])
    dhdtau_eval = ret2
    d2hdtau2_eval = d2h1dtau2_predictive_oa(ego)
    d2taudtaustar2_eval = d2taudtaustar2([tau1_star(ego)])
    d2taustardx2_eval = d2tau1stardx2(ego)
    outer = np.outer(dtaustardx_eval, dtaustardx_eval)

    ret = (
        d2hdx2_eval
        + dtaudtaustar_eval * d2hdtau2_eval * dtaudtaustar_eval * outer
        + dhdtau_eval * d2taudtaustar2_eval * outer
        + dhdtau_eval * d2taustardx2_eval * dtaudtaustar_eval
    )

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_poa2(ego):
    func1 = d2h2dx2_predictive_oa(ego)
    func2 = dh2dtau_predictive_oa(ego)

    try:
        ret1 = func1.subs({tau_sym: tau([tau2_star(ego)])})
    except AttributeError:
        ret1 = func1

    try:
        ret2 = func2.subs({tau_sym: tau([tau2_star(ego)])})
    except AttributeError:
        ret2 = func2

    d2hdx2_eval = ret1
    dtaustardx_eval = dtau2stardx(ego)
    dtaudtaustar_eval = dtaudtaustar([tau2_star(ego)])
    dhdtau_eval = ret2
    d2hdtau2_eval = d2h2dtau2_predictive_oa(ego)
    d2taudtaustar2_eval = d2taudtaustar2([tau2_star(ego)])
    d2taustardx2_eval = d2tau2stardx2(ego)
    outer = np.outer(dtaustardx_eval, dtaustardx_eval)

    ret = (
        d2hdx2_eval
        + dtaudtaustar_eval * d2hdtau2_eval * dtaudtaustar_eval * outer
        + dhdtau_eval * d2taudtaustar2_eval * outer
        + dhdtau_eval * d2taustardx2_eval * dtaudtaustar_eval
    )

    return np.squeeze(np.array(ret).astype(np.float64))


def h_oa1(ego):
    return relaxation * h0_oa1(ego) + h_poa1(ego)


def h_oa2(ego):
    return relaxation * h0_oa2(ego) + h_poa2(ego)


def h_oa3(ego):
    return h_oa3_func(ego)


def h_oa4(ego):
    return h_oa4_func(ego)


def dhdx_oa1(ego):
    ret = relaxation * dh0dx_oa1(ego) + dhdx_poa1(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx_oa2(ego):
    ret = relaxation * dh0dx_oa2(ego) + dhdx_poa2(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx_oa3(ego):
    ret = dhdx_oa3_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx_oa4(ego):
    ret = dhdx_oa4_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


# Necessary for stochastic systems
def d2hdx2_oa1(ego):
    ret = relaxation * d2h0dx2_oa1(ego) + d2hdx2_poa1(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_oa2(ego):
    ret = relaxation * d2h0dx2_oa2(ego) + d2hdx2_poa2(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_oa3(ego):
    ret = d2hdx2_oa3_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_oa4(ego):
    ret = d2hdx2_oa4_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))
