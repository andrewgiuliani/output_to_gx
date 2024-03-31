#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
from simsopt._core import load
from simsopt.geo import BoozerSurface, Volume
from simsopt.geo import CurveRZFourier, CurveXYZFourier, ToroidalFlux, SurfaceXYZTensorFourier
from simsopt.field import BiotSavart
from scipy.interpolate import InterpolatedUnivariateSpline
from simsopt.util.fourier_interpolation import fourier_interpolation
from chebpy.api import chebfun
from pyevtk.hl import gridToVTK
try:
    from ground.base import get_context
except ImportError:
    get_context = None

try:
    from bentley_ottmann.planar import contour_self_intersects
except ImportError:
    contour_self_intersects = None


def is_self_intersecting(surface, angle=0.):
    cs = surface.cross_section(angle)
    R = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
    Z = cs[:, 2]

    context = get_context()
    Point, Contour = context.point_cls, context.contour_cls
    contour = Contour([Point(R[i], Z[i]) for i in range(cs.shape[0])])
    return contour_self_intersects(contour)


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

def compute_surfaces(surfaces, coils, tf_profile, iota_profile, nsurfaces=10):
    tf_outer = ToroidalFlux(surfaces[-1], BiotSavart(coils)).J()
    tf_targets = np.linspace(0, 1, nsurfaces+1)[1:]**2

    current_sum = np.sum([np.abs(c.current.get_value()) for c in coils])
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
    
    new_iota_profile = [iota_profile[0]]
    new_surface_list = []
    new_tf_profile = [0.]
    for tf_target in tf_targets:
        idx = np.argmin(np.abs(tf_profile[1:]-tf_target))
        phis = np.linspace(0, 1/surfaces[idx].nfp, 2*surfaces[idx].mpol+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*surfaces[idx].ntor+1, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=surfaces[idx].mpol, ntor=surfaces[idx].ntor, \
                quadpoints_phi=phis, quadpoints_theta=thetas,\
                stellsym=surfaces[idx].stellsym, nfp=surfaces[idx].nfp)
        surface.x = surfaces[idx].x

        boozer_surface = BoozerSurface(BiotSavart(coils), surface,  ToroidalFlux(surface, BiotSavart(coils)), tf_target*tf_outer)
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota_profile[idx], G=G0)
        if not res['success']:
            surface.x = surfaces[idx].x
            boozer_surface.need_to_run_code=True
            res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-9, maxiter=500, constraint_weight=100., iota=iota_profile[idx], G=G0)
        
        is_inter = np.any([is_self_intersecting(surface, a) for a in np.linspace(0, 2*np.pi/surface.nfp, 10)])
        print(res['success'])
        if res['success'] and not is_inter:
            new_surface_list.append(surface)
            new_iota_profile.append(res['iota'])
            new_tf_profile.append(tf_target)
    return new_surface_list, new_iota_profile, new_tf_profile


def output_to_gx(axis, surfaces, iotas, tf, s=0.1, alpha=0, npoints=51, length=10*np.pi, filename='out'):
    S, VARPHI, THETA = np.meshgrid(tf, surfaces[0].quadpoints_phi, surfaces[0].quadpoints_theta, indexing='ij')
    XYZ  = np.array([s.gamma() for s in surfaces])
    XYZ_axis = np.stack([axis.gamma() for _ in range(S.shape[1])], axis=1)
    XYZ = np.concatenate((XYZ_axis[None, ...], XYZ), axis=0)
    gridToVTK(str(filename), XYZ[..., 0].copy(), XYZ[..., 1].copy(), XYZ[..., 2].copy())
    

    k = min([3, XYZ.shape[0]-1])

    # evaluate covariate basis functions at points on surface with label s
    s_XYZ = np.zeros(XYZ.shape[1:])
    for i in range(s_XYZ.shape[0]):
        for j in range(s_XYZ.shape[1]):
            s_XYZ[i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2, k=k)(s)
            s_XYZ[i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2, k=k)(s)
            s_XYZ[i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2, k=k)(s)


    # evaluate covariate basis functions at points on surface with label s
    s_dXYZ_dS = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dS.shape[0]):
        for j in range(s_dXYZ_dS.shape[1]):
            s_dXYZ_dS[i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2, k=k).derivative()(s)
            s_dXYZ_dS[i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2, k=k).derivative()(s)
            s_dXYZ_dS[i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2, k=k).derivative()(s)

    dXYZ_dVARPHI  = np.array([s.gammadash1() for s in surfaces])
    dXYZ_axis_dVARPHI = np.stack([axis.gammadash() for _ in range(S.shape[1])], axis=1)
    dXYZ_dVARPHI = np.concatenate((dXYZ_axis_dVARPHI[None, ...], dXYZ_dVARPHI), axis=0)
    s_dXYZ_dVARPHI = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dVARPHI.shape[0]):
        for j in range(s_dXYZ_dVARPHI.shape[1]):
            s_dXYZ_dVARPHI[i, j, 0] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 0], ext=2, k=k)(s)
            s_dXYZ_dVARPHI[i, j, 1] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 1], ext=2, k=k)(s)
            s_dXYZ_dVARPHI[i, j, 2] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 2], ext=2, k=k)(s)

    dXYZ_dTHETA  = np.array([s.gammadash2() for s in surfaces])
    dXYZ_axis_dTHETA = np.stack([np.zeros(axis.gamma().shape) for _ in range(S.shape[1])], axis=1)
    dXYZ_dTHETA = np.concatenate((dXYZ_axis_dTHETA[None, ...], dXYZ_dTHETA), axis=0)

    s_dXYZ_dTHETA = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dTHETA.shape[0]):
        for j in range(s_dXYZ_dTHETA.shape[1]):
            s_dXYZ_dTHETA[i, j, 0] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 0], ext=2, k=k)(s)
            s_dXYZ_dTHETA[i, j, 1] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 1], ext=2, k=k)(s)
            s_dXYZ_dTHETA[i, j, 2] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 2], ext=2, k=k)(s)
    
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

    iota = InterpolatedUnivariateSpline(tf, iotas, ext=2, k=k)(s)
    varphi = np.linspace(0, length, npoints)
    theta = alpha+iota*varphi
    
    gradS_dot_gradTHETA_on_fl = fourier_interpolation(gradS_dot_gradTHETA, varphi, y=theta)
    gradS_dot_gradVARPHI_on_fl = fourier_interpolation(gradS_dot_gradVARPHI, varphi, y=theta)
    gradTHETA_dot_gradVARPHI_on_fl = fourier_interpolation(gradTHETA_dot_gradVARPHI, varphi, y=theta)
    gradS_dot_gradS_on_fl = fourier_interpolation(gradS_dot_gradS, varphi, y=theta)
    gradVARPHI_dot_gradVARPHI_on_fl = fourier_interpolation(gradVARPHI_dot_gradVARPHI, varphi, y=theta)
    gradTHETA_dot_gradTHETA_on_fl = fourier_interpolation(gradTHETA_dot_gradTHETA, varphi, y=theta)
    modB_on_fl = fourier_interpolation(modB, varphi, y=theta)
    J_on_fl = fourier_interpolation(J, varphi, y=theta)
    
    out_dict = {'varphi_on_fl':varphi, 'theta_on_fl':theta,\
                'gradS_dot_gradTHETA_on_fl':gradS_dot_gradTHETA_on_fl, \
                'gradS_dot_gradVARPHI_on_fl':gradS_dot_gradVARPHI_on_fl, \
                'gradTHETA_dot_graVARPHI_on_fl':gradTHETA_dot_gradVARPHI_on_fl, \
                'gradS_dot_gradS_on_fl':gradS_dot_gradS_on_fl,\
                'gradVARPHI_dot_gradVARPHI_on_fl':gradVARPHI_dot_gradVARPHI_on_fl,\
                'gradTHETA_dot_gradTHETA_on_fl':gradTHETA_dot_gradTHETA_on_fl,\
                'modB_on_fl':modB_on_fl, 'J_on_fl':J_on_fl,\
                'gradS_dot_gradTHETA_on_s':gradS_dot_gradTHETA, \
                'gradS_dot_gradVARPHI_on_s':gradS_dot_gradVARPHI, \
                'gradTHETA_dot_gradVARPHI_on_s':gradTHETA_dot_gradVARPHI, \
                'gradS_dot_gradS_on_s':gradS_dot_gradS,\
                'gradVARPHI_dot_gradVARPHI_on_s':gradVARPHI_dot_gradVARPHI,\
                'gradTHETA_dot_gradTHETA_s':gradTHETA_dot_gradTHETA,\
                'modB_on_s':modB, 'J_on_s':J}
    return out_dict


#iID = 251778
iID = 0
df = pd.read_pickle('QUASR_full.pkl')
ID = df.iloc[iID].ID
fID = ID // 1000
#import subprocess;subprocess.run(["scp", f"agiuliani@popeye:/mnt/home/agiuliani/ceph/parameter_scan/QUASR_26032024/simsopt_serials/{fID:04}/serial{ID:07}.json", "./"])
iota_profile = df.iloc[iID].iota_profile
tf_profile = df.iloc[iID].tf_profile

[surfaces, axis, coils] = load(f'serial{ID:07}.json')
surfaces, iota_profile, tf_profile = compute_surfaces(surfaces, coils, tf_profile, iota_profile, nsurfaces=10)

# reparametrize the axis in boozer toroidal varphi
#axis_uniform = reparametrizeBoozer(axis, field=None)
axis_uniform = reparametrizeBoozer(axis, field=BiotSavart(coils))
quadpoints_varphi = np.linspace(0, 1, surfaces[0].quadpoints_phi.size*surfaces[0].nfp, endpoint=False)
axis = CurveXYZFourier(quadpoints_varphi, axis_uniform.order)
axis.x = axis_uniform.x


sdim1_max = np.max([s.quadpoints_phi.size for s in surfaces])
sdim2_max = np.max([s.quadpoints_theta.size for s in surfaces])

# put surface on entire torus
surfaces_fp = surfaces
surfaces_ft = []
for s in surfaces_fp:
    snew = SurfaceXYZTensorFourier(quadpoints_phi=np.linspace(0, 1, sdim1_max*s.nfp, endpoint=False),\
                                   quadpoints_theta=np.linspace(0, 1, sdim2_max*s.nfp, endpoint=False),\
                                   nfp=s.nfp,\
                                   stellsym=s.stellsym,\
                                   mpol=s.mpol, ntor=s.ntor)
    snew.x = s.x
    surfaces_ft.append(snew)




# CHECK THAT WE ARE CONVERGING TO THE CORRECT VARPHI ANGLES
#import ipdb;ipdb.set_trace()
#from simsopt.geo import BoozerSurface, Volume
#s0=surfaces[0]
#iota = iota_profile[0]
#current_sum = np.sum([np.abs(c.current.get_value()) for c in coils])
#G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
#err_prev = np.inf
#for i in range(10):
#    boozer_surface = BoozerSurface(BiotSavart(coils), s0,  Volume(s0), s0.volume()/4)
#    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)
#    snew = SurfaceXYZTensorFourier(quadpoints_phi=np.linspace(0, 1, s0.quadpoints_phi.size*s.nfp, endpoint=False),\
#                                   quadpoints_theta=np.linspace(0, 1, s0.quadpoints_theta.size*s.nfp, endpoint=False),\
#                                   nfp=s0.nfp,\
#                                   stellsym=s0.stellsym,\
#                                   mpol=s0.mpol, ntor=s0.ntor)
#    snew.x = s0.x
#    err = np.mean(np.linalg.norm(snew.gamma() - axis.gamma()[:, None, :], axis=-1))
#    print(err, err/err_prev, s0.minor_radius())
#    err_prev = err
#import ipdb;ipdb.set_trace()


# CHECK THAT WE ARE CONVERGING TO THE CORRECT VOLUME










out = output_to_gx(axis, surfaces_ft, iota_profile, tf_profile, s=1e-1, npoints=512)
variables = [k for k in out.keys() if k != 'varphi_on_fl' and k !='theta_on_fl' and k[-1] != 's']

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
nrows = 4
ncols = 2
for j, variable in enumerate(variables):
    plt.subplot(nrows, ncols, j + 1)
    plt.plot(out['varphi_on_fl'], out[variable])
    plt.xlabel('Standard toroidal angle $\phi$')
    plt.title(variable)

#plt.figtext(0.5, 0.995, f'surface s={surface}, field line alpha={alpha} from file {vmec_output_file}', ha='center', va='top')
plt.tight_layout()
plt.show()
