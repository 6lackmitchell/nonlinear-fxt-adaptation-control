import builtins
import numpy as np
import symengine as se
from importlib import import_module
from core.cbfs.cbf_wrappers import symbolic_cbf_wrapper_singleagent

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
cx1 = -2.5
cy1 = 0.0
cx2 = 1.25
cy2 = -1.25

# Speed CBF Symbolic
h_oa1_symbolic = (ss[0] - cx1) ** 2 + (ss[1] - cy1) ** 2 - R**2
h_oa2_symbolic = (ss[0] - cx2) ** 2 + (ss[1] - cy2) ** 2 - R**2
dhdx_oa1_symbolic = (se.DenseMatrix([h_oa1_symbolic]).jacobian(se.DenseMatrix(ss))).T
dhdx_oa2_symbolic = (se.DenseMatrix([h_oa2_symbolic]).jacobian(se.DenseMatrix(ss))).T
d2hdx2_oa1_symbolic = dhdx_oa1_symbolic.jacobian(se.DenseMatrix(ss))
d2hdx2_oa2_symbolic = dhdx_oa2_symbolic.jacobian(se.DenseMatrix(ss))
h_oa1_func = symbolic_cbf_wrapper_singleagent(h_oa1_symbolic, ss)
h_oa2_func = symbolic_cbf_wrapper_singleagent(h_oa2_symbolic, ss)
dhdx_oa1_func = symbolic_cbf_wrapper_singleagent(dhdx_oa1_symbolic, ss)
dhdx_oa2_func = symbolic_cbf_wrapper_singleagent(dhdx_oa2_symbolic, ss)
d2hdx2_oa1_func = symbolic_cbf_wrapper_singleagent(d2hdx2_oa1_symbolic, ss)
d2hdx2_oa2_func = symbolic_cbf_wrapper_singleagent(d2hdx2_oa2_symbolic, ss)


def h_oa1(ego):
    return h_oa1_func(ego)


def h_oa2(ego):
    return h_oa2_func(ego)


def dhdx_oa1(ego):
    ret = dhdx_oa1_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx_oa2(ego):
    ret = dhdx_oa2_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_oa1(ego):
    ret = d2hdx2_oa1_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_oa2(ego):
    ret = d2hdx2_oa2_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))
