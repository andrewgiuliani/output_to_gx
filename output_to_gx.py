import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
from qsc import Qsc
from simsopt._core import load
from simsopt.geo import CurveRZFourier, CurveXYZFourier, ToroidalFlux, SurfaceXYZTensorFourier
from simsopt.field import BiotSavart
from scipy.interpolate import InterpolatedUnivariateSpline
from simsopt.util.fourier_interpolation import fourier_interpolation
from chebpy.api import chebfun

def fourier_interpolation2(x, y, f):
    # interpolate to x
    f_at_x = []
    for j in range(f.shape[1]):
        fx = fourier_interpolation(f[:, j], x)
        f_at_x.append(fx)
    f_at_x = np.array(f_at_x).T

    # interpolate to y
    f_at_xy = []
    for i in range(x.size):
        fxy = fourier_interpolation(f[i, :], [y[i]])
        f_at_xy.append(fxy)
    f_at_xy = np.array(f_at_xy).flatten()
    return f_at_xy

def reparametrizeBoozer(axis, field=None, ppp=10):
    def x(t):
        ind = np.array(t)
        out = np.zeros((ind.size,3))
        axis.gamma_impl(out, ind)
        return out[:, 0]
    def y(t):
        ind = np.array(t)
        out = np.zeros((ind.size,3))
        axis.gamma_impl(out, ind)
        return out[:, 1]
    def z(t):
        ind = np.array(t)
        out = np.zeros((ind.size,3))
        axis.gamma_impl(out, ind)
        return out[:, 2]
    
    # Convert to chebfun for convenience
    xc = chebfun(x, [0, 1])
    yc = chebfun(y, [0, 1])
    zc = chebfun(z, [0, 1])
    
    
    xpc = xc.diff()
    ypc = yc.diff()
    zpc = zc.diff()
    
    # Find nodes that are equispaced in arc length
    speed = np.sqrt(xpc*xpc + ypc*ypc + zpc*zpc)
    #G0 = (B*speed).sum()
    #arclength_B0_over_G0 = (speed*B/G0).cumsum()/(speed*B/G0).sum()
    if field is not None:
        def modB(t):
            ind = np.array(t)
            out = np.zeros((ind.size,3))
            axis.gamma_impl(out, ind)
            field.set_points(out)
            absB = field.AbsB()
            return absB
        B  = chebfun(modB, [0, 1])
        arclength = (speed*B).cumsum()/(speed*B).sum()
    else:
        arclength = speed.cumsum()/speed.sum()


    
    npts = ppp*axis.order*axis.nfp
    if npts % 2 == 0:
        npts+=1

    quadpoints_phi = [0.]
    quadpoints_varphi = np.linspace(0, 1, npts, endpoint=False)
    for qp in quadpoints_varphi[1:]:
        phi = (arclength-qp).roots()[0]
        quadpoints_phi.append(phi)
    
    axis_nonuniform = CurveRZFourier(quadpoints_phi, axis.order, axis.nfp, axis.stellsym)
    axis_nonuniform.x = axis.x
    axis_uniform = CurveXYZFourier(quadpoints_varphi, npts//2)
    axis_uniform.least_squares_fit(axis_nonuniform.gamma())
    return axis_uniform



def output_to_gx(axis, surfaces, iotas, tf, s=0.1, alpha=0, npoints=51, length=10*np.pi):
    S, VARPHI, THETA = np.meshgrid(tf, surfaces[0].quadpoints_phi, surfaces[0].quadpoints_theta, indexing='ij')
    XYZ  = np.array([s.gamma() for s in surfaces])
    XYZ_axis = np.stack([axis.gamma() for _ in range(S.shape[1])], axis=1)
    XYZ = np.concatenate((XYZ_axis[None, ...], XYZ), axis=0)


    # evaluate covariate basis functions at points on surface with label s
    s_XYZ = np.zeros(XYZ.shape[1:])
    for i in range(s_XYZ.shape[0]):
        for j in range(s_XYZ.shape[1]):
            s_XYZ[i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2)(s)
            s_XYZ[i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2)(s)
            s_XYZ[i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2)(s)


    # evaluate covariate basis functions at points on surface with label s
    s_dXYZ_dS = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dS.shape[0]):
        for j in range(s_dXYZ_dS.shape[1]):
            s_dXYZ_dS[i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2).derivative()(s)
            s_dXYZ_dS[i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2).derivative()(s)
            s_dXYZ_dS[i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2).derivative()(s)

    dXYZ_dVARPHI  = np.array([s.gammadash1() for s in surfaces])
    dXYZ_axis_dVARPHI = np.stack([axis.gammadash() for _ in range(S.shape[1])], axis=1)
    dXYZ_dVARPHI = np.concatenate((dXYZ_axis_dVARPHI[None, ...], dXYZ_dVARPHI), axis=0)
    s_dXYZ_dVARPHI = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dVARPHI.shape[0]):
        for j in range(s_dXYZ_dVARPHI.shape[1]):
            s_dXYZ_dVARPHI[i, j, 0] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 0], ext=2)(s)
            s_dXYZ_dVARPHI[i, j, 1] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 1], ext=2)(s)
            s_dXYZ_dVARPHI[i, j, 2] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 2], ext=2)(s)

    dXYZ_dTHETA  = np.array([s.gammadash2() for s in surfaces])
    dXYZ_axis_dTHETA = np.stack([np.zeros(axis.gamma().shape) for _ in range(S.shape[1])], axis=1)
    dXYZ_dTHETA = np.concatenate((dXYZ_axis_dTHETA[None, ...], dXYZ_dTHETA), axis=0)

    s_dXYZ_dTHETA = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dTHETA.shape[0]):
        for j in range(s_dXYZ_dTHETA.shape[1]):
            s_dXYZ_dTHETA[i, j, 0] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 0], ext=2)(s)
            s_dXYZ_dTHETA[i, j, 1] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 1], ext=2)(s)
            s_dXYZ_dTHETA[i, j, 2] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 2], ext=2)(s)
    
    J = np.sum(s_dXYZ_dS * np.cross(s_dXYZ_dVARPHI, s_dXYZ_dTHETA), axis=-1)
    gradS      = np.cross(s_dXYZ_dVARPHI, s_dXYZ_dTHETA)/J[:, :, None]
    gradVARPHI = np.cross(s_dXYZ_dTHETA, s_dXYZ_dS)/J[:, :, None]
    gradTHETA = np.cross(s_dXYZ_dS, s_dXYZ_dVARPHI)/J[:, :, None]
    
    gradS_dot_gradTHETA = np.sum(gradS*gradTHETA, axis=-1)
    gradS_dot_gradVARPHI = np.sum(gradS*gradVARPHI, axis=-1)
    gradTHETA_dot_gradVARPHI = np.sum(gradTHETA*gradVARPHI, axis=-1)
    gradS_dot_gradS = np.sum(gradS*gradS, axis=-1)
    gradVARPHI_dot_gradVARPHI = np.sum(gradVARPHI*gradVARPHI, axis=-1)
    gradTHETA_dot_gradTHETA = np.sum(gradTHETA*gradTHETA, axis=-1)
    modB = BiotSavart(coils).set_points(s_XYZ.reshape((-1, 3))).AbsB().reshape(s_XYZ.shape[:-1])

    iota = InterpolatedUnivariateSpline(tf, iotas, ext=2)(s)
    varphi = np.linspace(0, length, npoints)
    theta = alpha+iota*varphi
    
    gradS_dot_gradTHETA_on_fl = fourier_interpolation2(varphi, theta, gradS_dot_gradTHETA)
    gradS_dot_gradVARPHI_on_fl = fourier_interpolation2(varphi, theta, gradS_dot_gradVARPHI)
    gradTHETA_dot_gradVARPHI_on_fl = fourier_interpolation2(varphi, theta, gradTHETA_dot_gradVARPHI)
    gradS_dot_gradS_on_fl = fourier_interpolation2(varphi, theta, gradS_dot_gradS)
    gradVARPHI_dot_gradVARPHI_on_fl = fourier_interpolation2(varphi, theta, gradVARPHI_dot_gradVARPHI)
    gradTHETA_dot_gradTHETA_on_fl = fourier_interpolation2(varphi, theta, gradTHETA_dot_gradTHETA)
    modB_on_fl = fourier_interpolation2(varphi, theta, modB)
    J_on_fl = fourier_interpolation2(varphi, theta, J)
    
    out_dict = {'gradS_dot_gradTHETA':gradS_dot_gradTHETA_on_fl, \
                'gradS_dot_gradVARPHI':gradS_dot_gradVARPHI_on_fl, \
                'gradTHETA_dot_graVARPHI':gradTHETA_dot_gradVARPHI_on_fl, \
                'gradS_dot_gradS':gradS_dot_gradS_on_fl,\
                'gradVARPHI_dot_gradVARPHI':gradVARPHI_dot_gradVARPHI_on_fl,\
                'gradTHETA_dot_gradTHETA':gradTHETA_dot_gradTHETA_on_fl,\
                'modB':modB_on_fl, 'J':J_on_fl}
    return out_dict


iID = 0
df = pd.read_pickle('files/QUASR_full.pkl')
[surfaces, axis, coils] = load(f'.json')

# reparametrize the axis in boozer toroidal varphi
axis_uniform = reparametrizeBoozer(axis, field=BiotSavart(coils))
quadpoints_varphi = np.linspace(0, 1, surfaces[0].quadpoints_phi.size*surfaces[0].nfp, endpoint=False)
axis = CurveXYZFourier(quadpoints_varphi, axis_uniform.order)
axis.x = axis_uniform.x

# put surface on entire torus
surfaces_fp = surfaces
surfaces_ft = []
for s in surfaces_fp:
    snew = SurfaceXYZTensorFourier(quadpoints_phi=np.linspace(0, 1, s.quadpoints_phi.size*s.nfp, endpoint=False),\
                                   quadpoints_theta=np.linspace(0, 1, s.quadpoints_theta.size*s.nfp, endpoint=False),\
                                   nfp=s.nfp,\
                                   stellsym=s.stellsym,\
                                   mpol=s.mpol, ntor=s.ntor)
    snew.x = s.x
    surfaces_ft.append(snew)

iota_profile = df.iloc[iID].iota_profile
tf_profile = df.iloc[iID].tf_profile
out = output_to_gx(axis, surfaces_ft, iota_profile, tf_profile, s=1.)
