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

def fourier_interpolation2(fk, x, y):
    r"""
    This function interpolates a periodic 2D function using a Fourier interpolant. Using 1D Fourier interpolation
    already implemented in SIMSOPT, the 2D fourier interpolant is evaluated at the coordinates defined in `x` and `y`.

    Note: `x` and `y` are assumed to be in radians, i.e., not normalized to [0, 1) as is sometimes assumed in SIMSOPT.

    Args:
        fk: the function values used to define the Fourier interpolant.  This is a 2D array.
        x: x-coordinates where we want to evaluate the Fourier interpolant. This is a 1D array.
        y: y-coordinates where we want to evaluate the Fourier interpolant. This is a 1D array.

    Returns:
        f_at_xy: the value of the Fourier interpolant at the coordinates defined in `x` and `y`.
    """

    # interpolate to x
    f_at_x = []
    for j in range(fk.shape[1]):
        fx = fourier_interpolation(fk[:, j], x)
        f_at_x.append(fx)
    f_at_x = np.array(f_at_x).T

    # interpolate to y
    f_at_xy = []
    for i in range(x.size):
        fxy = fourier_interpolation(f_at_x[i, :], [y[i]])
        f_at_xy.append(fxy)
    f_at_xy = np.array(f_at_xy).flatten()
    return f_at_xy

def is_self_intersecting(surface, angle=0.):
    r"""
    This function checks whether the input surface is self-intersecting.  This check is done by
    computing a cross-section of the input surface at the standard cylindrical angle `angle`.  Then,
    this function checks whether the cross-section (approximated as a piecewise linear spline) self-intersects
    using the Bentley-Ottmann algorithm.  Note that this function may falsely return that the surface is not
    self-intersecting, even though it is at a different cylindrical angle `angle`.  It is recommended that this
    function is run at multiple cylindrical angles to be certain that the surface is not self-intersecting everywhere.

    Args:
        surface: the surface that we want to check is self-intersecting or not.
        angle:  the standard cylindrical angle at which the function checks the surface's cross section
                is self-intersecting.
    Returns:
        True if the surface self-intersects at the standard cylindrical angle `angle`, False otherwise.
    """
    cs = surface.cross_section(angle)
    R = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
    Z = cs[:, 2]

    context = get_context()
    Point, Contour = context.point_cls, context.contour_cls
    contour = Contour([Point(R[i], Z[i]) for i in range(cs.shape[0])])
    return contour_self_intersects(contour)


def reparametrize(axis, weighting=None, ppp=10):
    r"""
    This function reparametetrizes the input magnetic axis to have uniform weighted incremental arclength.

    Args:
        axis: the input magnetic axis that is an instance of CurveRZFourier
        weighting: the weighting function applied to the incremental arclength.  If `None`, then the function
               uses a weight of 1., i.e., the resulting curve will have uniform incrememental arclength.
        ppp:  the number of points per period used to define the new reparametrized curve.  A number greater than 10
              is recommended.
    Returns:
       axis_uniform: a new CurveXYZFourier that is a reparametrization of the input `axis` with uniform (weighted) incremental
                    arclength.
    """
    
    assert type(axis) == CurveRZFourier

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
    if weighting is not None:
        w  = chebfun(weighting, [0, 1])
        arclength = (speed*w).cumsum()/(speed*w).sum()
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
    r"""
    Compute nsurfaces magnetic surfaces, that are uniformly spaced in minor radius, in the magnetic field given by BiotSavart(coils).

    Args:
        surfaces: list of surfaces that are known to lie in the field BiotSavart(coils).
        coils: electromagnetic coils that generate a magnetic field in which we aim to compute surfaces
        tf_profile: the toroidal fluxes associated to the input surfaces
        iota_profile: rotational transform associated to the input surfaces
        nsurfaces: the number of surfaces that we aim to compute in the input magnetic field given by BiotSavart(coils).
    
    Returns:
        new_surface_list: list of computed magnetic surfaces that uniformly spaced in minor radius.
        new iota_profile: array of the rotational transforms of the surfaces in `new_surface_list`.
        new_tf_profile: array of toroidal fluxes through the surfaces in `new_surface_list`.
    """

    tf_outer = ToroidalFlux(surfaces[-1], BiotSavart(coils)).J()
    tf_targets = np.linspace(0, 1, nsurfaces+1)[1:]**2
    
    current_sum = np.sum([np.abs(c.current.get_value()) for c in coils])
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
    
    new_iota_profile = []
    new_tf_profile = []
    new_surface_list = []
    for tf_target in tf_targets:
        idx = np.argmin(np.abs(tf_profile-tf_target))
        phis = np.linspace(0, 1/surfaces[idx].nfp, 2*surfaces[idx].mpol+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*surfaces[idx].ntor+1, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=surfaces[idx].mpol, ntor=surfaces[idx].ntor, \
                quadpoints_phi=phis, quadpoints_theta=thetas,\
                stellsym=surfaces[idx].stellsym, nfp=surfaces[idx].nfp)
        surface.x = surfaces[idx].x
        
        boozer_surface = BoozerSurface(BiotSavart(coils), surface,  ToroidalFlux(surface, BiotSavart(coils)), tf_target*tf_outer)
        try:
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota_profile[idx], G=G0)
        except:
            res = {'success': False}

        if not res['success']:
            surface.x = surfaces[idx].x
            boozer_surface.need_to_run_code=True
            
            try:
                res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-9, maxiter=500, constraint_weight=100., iota=iota_profile[idx], G=G0)
            except:
               continue

        is_inter = np.any([is_self_intersecting(surface, a) for a in np.linspace(0, 2*np.pi/surface.nfp, 10)])
        print(res['success'])
        if res['success'] and not is_inter:
            new_surface_list.append(surface)
            new_iota_profile.append(res['iota'])
            new_tf_profile.append(tf_target)
    
    return new_surface_list, np.array(new_iota_profile), np.array(new_tf_profile)


def output_to_gx(axis, surfaces, iotas, tf, field, s=0.1, alpha=0, npoints=1024, length=10*np.pi, nsurfaces=None, filename=None):
    r"""
    Compute geometric quantities useful for gyrokinetic simulations alone field lines.  This function takes as input
    a magnetic axis, a number of surfaces, as well as the rotational transform and torodial flux associated to the axis 
    and surfaces. It is assumed that the input surfaces are parametrized in Boozer coordinates and are instances of 
    `SurfaceXYZTensorFourier` and that the input magnetic axis is an instance of `CurveRZFourier`.
    
    The first step of the algorithm is the reparametrize the toroidal angle on the magnetic axis from toroidal :math:`\phi` to
    Boozer :math:`\varphi`.  This is done using pycheb, the Python version of the chebfun package.

    Next, the algorithm uses spline interpolation radially outward from the magnetic axis, and Fourier interpolation toroidally to 
    construct the geometric quantities sampled on field lines.
    
    The geometric quantities computed by this function are :math:`\nabla s \cdot \nabla s`, :math:`\nabla \varphi \cdot \nabla \varphi`, 
    :math:`\nabla \theta \cdot \nabla \theta`, :math:`\nabla s \cdot \nabla \varphi`, :math:`\nabla \s \cdot \nabla \theta`, 
    :math:`\nabla \theta \cdot \nabla \varphi`, and :math:`\|\mathbf B\|`, and Boozer coordiantes Jacobian. 

    The value of ``s`` provided as input must lie between 0 and 1, and corresponds to the normalized toroidal flux on which
    the geometric quantities are calculated.
    
    The angles varphi, theta are assumed to be in radians and are not normalized by 2pi as is done in SurfaceXYZTensorFourier.
    
    NOTE: if large enough islands exist in `field`, then the computed geometric data likely cannot be trusted.

    Args:
        axis: magnetic axis of the input magnetic field as a CurveRZFourier
        surfaces: list of magnetic surfaces parametrized in Boozer coordinates as SurfaceXYZTensorFourier
        iotas: list of rotational transforms on the axis and surfaces
        tf: list of toroidal fluxes through the axis and surfaces, though the toroidal flux through the axis is necessarily 0.
        field: magnetic field in which the axis and surfaces lie, e.g., it could be a BiotSavart(coils) field.
        s: the normalized toroidal flux on which the geometric quantities are to be computed.
        alpha: the label of the field line on which these quantities are computed, alpha = theta - iota*varphi
        npoints: number of points on the field line where geometric quantities are sampled
        length: toroidal length of the field line where geometric quantities are sampled.
        nsurfaces: either 'None' or some integer > 0.  If `None', then the number of surfaces used to represent the magnetic field is the same as len(surfaces).
                   If some nonzero integer, then `nsurfaces` surfaces are computed in `field`, uniformly spaced in minor radius.
        filename: either `None' or a string.  If a string is provided, then a png file is saved to disk to visualize the geometric quantities and a
                  vtk file is output the visualize some of the geometric quantities in paraview.
    Returns:
        out_dict: a dictionary containing geometric quantities sampled on fieldlines when the dictionary key ends with `on_fl` and on the surface with label `s` 
                  when the dictionary key ends with `on_s`.  Quantities samples along a field line are stored as a 1D array with `npoints` entries uniformly spaced
                  in the Boozer toroidal angle.  Quantities sampled on a surface are stored as a 2D array, uniformly spaced in Boozer angles on the full torus.
    """

    assert type(axis) == CurveRZFourier
    assert np.all([type(s) == SurfaceXYZTensorFourier for s in surfaces])
    assert (s >= 0) and (s<=1.)
    assert nsurfaces is None or nsurfaces > 0

    if nsurfaces is not None:
        new_surfaces, new_iotas, new_tf = compute_surfaces(surfaces, field, tf[1:], iotas[1:], nsurfaces=nsurfaces)
        surfaces = new_surfaces
        iotas = np.concatenate(([iotas[0]], new_iotas))
        tf = np.concatenate(([0.], new_tf))

    sdim1_max = np.max([s.quadpoints_phi.size for s in surfaces])
    sdim2_max = np.max([s.quadpoints_theta.size for s in surfaces])
    
    def modB(t):
        ind = np.array(t)
        out = np.zeros((ind.size,3))
        axis.gamma_impl(out, ind)
        field.set_points(out)
        absB = field.AbsB()
        return absB

    # reparametrize the axis in boozer toroidal varphi
    axis_uniform = reparametrize(axis, weighting=modB)
    quadpoints_varphi = np.linspace(0, 1, sdim1_max*surfaces[0].nfp, endpoint=False)
    axis = CurveXYZFourier(quadpoints_varphi, axis_uniform.order)
    axis.x = axis_uniform.x

    # put surface on entire torus
    surfaces_fp = surfaces
    surfaces_ft = []
    for stemp in surfaces_fp:
        snew = SurfaceXYZTensorFourier(quadpoints_phi=np.linspace(0, 1, sdim1_max*stemp.nfp, endpoint=False),\
                                       quadpoints_theta=np.linspace(0, 1, sdim2_max*stemp.nfp, endpoint=False),\
                                       nfp=stemp.nfp,\
                                       stellsym=stemp.stellsym,\
                                       mpol=stemp.mpol, ntor=stemp.ntor)
        snew.x = stemp.x
        surfaces_ft.append(snew)
    surfaces = surfaces_ft

    S, VARPHI, THETA = np.meshgrid(tf, surfaces[0].quadpoints_phi, surfaces[0].quadpoints_theta, indexing='ij')
    XYZ  = np.array([s.gamma() for s in surfaces])
    XYZ_axis = np.stack([axis.gamma() for _ in range(S.shape[1])], axis=1)
    XYZ = np.concatenate((XYZ_axis[None, ...], XYZ), axis=0)
    
    k = min([3, XYZ.shape[0]-1])
    
    # evaluate what the surface coordinates on label s 
    s_XYZ = np.zeros(XYZ.shape[1:])
    s_VARPHI = np.zeros(VARPHI.shape[1:])
    for i in range(s_XYZ.shape[0]):
        for j in range(s_XYZ.shape[1]):
            s_XYZ[i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2, k=k)(s)
            s_XYZ[i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2, k=k)(s)
            s_XYZ[i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2, k=k)(s)
            # varphi coordinates on flux surface
            s_VARPHI[i,j] = InterpolatedUnivariateSpline(tf, VARPHI[:,i,j], ext=2, k=k)(s)
    
    # evaluate covariate basis functions at points on surface with label s
    dXYZ_dS = np.zeros(XYZ.shape)
    for m in range(dXYZ_dS.shape[0]):
        for i in range(dXYZ_dS.shape[1]):
            for j in range(dXYZ_dS.shape[2]):
                dXYZ_dS[m, i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2, k=k).derivative()(S[m, i, j])
                dXYZ_dS[m, i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2, k=k).derivative()(S[m, i, j])
                dXYZ_dS[m, i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2, k=k).derivative()(S[m, i, j])

    # evaluate covariate basis functions at points on surface with label s
    s_dXYZ_dS = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dS.shape[0]):
        for j in range(s_dXYZ_dS.shape[1]):
            s_dXYZ_dS[i, j, 0] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 0], ext=2, k=k).derivative()(s)
            s_dXYZ_dS[i, j, 1] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 1], ext=2, k=k).derivative()(s)
            s_dXYZ_dS[i, j, 2] = InterpolatedUnivariateSpline(tf, XYZ[:, i, j, 2], ext=2, k=k).derivative()(s)

    dXYZ_dVARPHI  = np.array([s.gammadash1() for s in surfaces])
    dXYZ_axis_dVARPHI = np.stack([axis.gammadash() for _ in range(S.shape[1])], axis=1)
    
    # divide by 2pi since the varphi angle varies from 0 to 1 in SurfaceXYZTensorFourier
    dXYZ_dVARPHI = np.concatenate((dXYZ_axis_dVARPHI[None, ...], dXYZ_dVARPHI), axis=0)/(2*np.pi)
    s_dXYZ_dVARPHI = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dVARPHI.shape[0]):
        for j in range(s_dXYZ_dVARPHI.shape[1]):
            s_dXYZ_dVARPHI[i, j, 0] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 0], ext=2, k=k)(s)
            s_dXYZ_dVARPHI[i, j, 1] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 1], ext=2, k=k)(s)
            s_dXYZ_dVARPHI[i, j, 2] = InterpolatedUnivariateSpline(tf, dXYZ_dVARPHI[:, i, j, 2], ext=2, k=k)(s)

    dXYZ_dTHETA  = np.array([s.gammadash2() for s in surfaces])
    dXYZ_axis_dTHETA = np.stack([np.zeros(axis.gamma().shape) for _ in range(S.shape[1])], axis=1)
    
    # divide by 2pi since the theta angle varies from 0 to 1 in SurfaceXYZTensorFourier
    dXYZ_dTHETA = np.concatenate((dXYZ_axis_dTHETA[None, ...], dXYZ_dTHETA), axis=0)/(2*np.pi)
    s_dXYZ_dTHETA = np.zeros(XYZ.shape[1:])
    for i in range(s_dXYZ_dTHETA.shape[0]):
        for j in range(s_dXYZ_dTHETA.shape[1]):
            s_dXYZ_dTHETA[i, j, 0] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 0], ext=2, k=k)(s)
            s_dXYZ_dTHETA[i, j, 1] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 1], ext=2, k=k)(s)
            s_dXYZ_dTHETA[i, j, 2] = InterpolatedUnivariateSpline(tf, dXYZ_dTHETA[:, i, j, 2], ext=2, k=k)(s)


    iota = InterpolatedUnivariateSpline(tf, iotas, ext=2, k=k)(s)
    varphi = np.linspace(0, length, npoints)
    theta = alpha+iota*varphi
    
    if k > 0:
        dds_iota = InterpolatedUnivariateSpline(tf, iotas, ext=2, k=k).derivative(n=1)(s)
    else:
        dds_iota = 0.0
    shat = -2 * s * dds_iota/iota

    
    # using the dual relations, determine the contravariant basis functions
    J = np.sum(s_dXYZ_dS * np.cross(s_dXYZ_dVARPHI, s_dXYZ_dTHETA), axis=-1)
    gradS      = np.cross(s_dXYZ_dVARPHI, s_dXYZ_dTHETA)/J[:, :, None]
    gradVARPHI = np.cross(s_dXYZ_dTHETA, s_dXYZ_dS)/J[:, :, None]
    gradTHETA = np.cross(s_dXYZ_dS, s_dXYZ_dVARPHI)/J[:, :, None]
    # gradALPHA = gradTHETA - iota * gradVARPHI - s_VARPHI[:, :, None] * gradS * dds_iota
    
    
    gradS_dot_gradTHETA = np.sum(gradS*gradTHETA, axis=-1)
    gradS_dot_gradVARPHI = np.sum(gradS*gradVARPHI, axis=-1)
    gradTHETA_dot_gradVARPHI = np.sum(gradTHETA*gradVARPHI, axis=-1)
    gradS_dot_gradS = np.sum(gradS*gradS, axis=-1)
    gradVARPHI_dot_gradVARPHI = np.sum(gradVARPHI*gradVARPHI, axis=-1)
    gradTHETA_dot_gradTHETA = np.sum(gradTHETA*gradTHETA, axis=-1)
    modB = field.set_points(s_XYZ.reshape((-1, 3))).AbsB().reshape(s_XYZ.shape[:-1])

    gradpar = -iota * J/modB # \vec{b} \cdot \nabla \theta. equality follows from \vec{B} = \nabla \psi x \nabla \alpha. 
    # gradALPHA_dot_gradALPHA = np.sum(gradALPHA*gradALPHA, axis=-1)
    # gradS_dot_gradALPHA = np.sum(gradS*gradALPHA, axis=-1)

    # calculate drifts
    B = field.set_points(s_XYZ.reshape((-1, 3))).B().reshape(s_XYZ.shape)
    gradB = field.set_points(s_XYZ.reshape((-1, 3))).GradAbsB().reshape(s_XYZ.shape)
    # gradALPHA_dot_B_cross_gradB = np.sum(gradALPHA * np.cross(B, gradB), axis=-1)
    gradS_dot_B_cross_gradB = np.sum(gradS * np.cross(B, gradB), axis=-1)
    
    
    gradS_dot_gradTHETA_on_fl = fourier_interpolation2(gradS_dot_gradTHETA, varphi, theta)
    gradS_dot_gradVARPHI_on_fl = fourier_interpolation2(gradS_dot_gradVARPHI, varphi, theta)
    gradTHETA_dot_gradVARPHI_on_fl = fourier_interpolation2(gradTHETA_dot_gradVARPHI, varphi, theta)
    gradS_dot_gradS_on_fl = fourier_interpolation2(gradS_dot_gradS, varphi, theta)
    gradVARPHI_dot_gradVARPHI_on_fl = fourier_interpolation2(gradVARPHI_dot_gradVARPHI, varphi, theta)
    gradTHETA_dot_gradTHETA_on_fl = fourier_interpolation2(gradTHETA_dot_gradTHETA, varphi, theta)
    modB_on_fl = fourier_interpolation2(modB, varphi, theta)
    J_on_fl = fourier_interpolation2(J, varphi, theta)


    gradALPHA_on_fl = np.zeros((npoints, 3))
    gradB_on_fl = np.zeros((npoints, 3))
    B_on_fl = np.zeros((npoints, 3))
    for i in range(3):
        gradB_on_fl[:,i] = fourier_interpolation2(gradB[i], varphi, theta)
        B_on_fl[:,i] = fourier_interpolation2(B[i], varphi, theta)
        gradVARPHI_on_fl = fourier_interpolation2(gradVARPHI[i], varphi, theta)
        gradTHETA_on_fl = fourier_interpolation2(gradTHETA[i], varphi, theta)
        gradS_on_fl = fourier_interpolation2(gradS[i], varphi, theta)
        gradALPHA_on_fl[:,i] = gradTHETA_on_fl - iota * gradVARPHI_on_fl - dds_iota * varphi * gradS_on_fl
        
    gradALPHA_dot_B_cross_gradB_on_fl = np.sum(gradALPHA_on_fl * np.cross(B_on_fl, gradB_on_fl), axis=-1)

    gradpar_on_fl = fourier_interpolation2(gradpar, varphi, theta)
    gradALPHA_dot_gradALPHA_on_fl = gradTHETA_dot_gradTHETA_on_fl + iota**2 * gradVARPHI_dot_gradVARPHI_on_fl \
        + varphi**2 * dds_iota**2 * gradS_dot_gradS_on_fl \
        - 2 * (iota * gradTHETA_dot_gradVARPHI_on_fl + varphi * dds_iota * gradS_dot_gradTHETA_on_fl + varphi * dds_iota * iota * gradS_dot_gradVARPHI_on_fl)
    gradS_dot_gradALPHA_on_fl = gradS_dot_gradTHETA_on_fl - iota * gradS_dot_gradVARPHI_on_fl - varphi * dds_iota * gradS_dot_gradS_on_fl

    gradS_dot_B_cross_gradB_on_fl =  fourier_interpolation2(gradS_dot_B_cross_gradB, varphi, theta) 
    out_dict = {'varphi_on_fl':varphi, 'theta_on_fl':theta,\
                'gradS_dot_gradTHETA_on_fl':gradS_dot_gradTHETA_on_fl, \
                'gradS_dot_gradVARPHI_on_fl':gradS_dot_gradVARPHI_on_fl, \
                'gradTHETA_dot_gradVARPHI_on_fl':gradTHETA_dot_gradVARPHI_on_fl, \
                'gradS_dot_gradS_on_fl':gradS_dot_gradS_on_fl,\
                'gradVARPHI_dot_gradVARPHI_on_fl':gradVARPHI_dot_gradVARPHI_on_fl,\
                'gradTHETA_dot_gradTHETA_on_fl':gradTHETA_dot_gradTHETA_on_fl,\
                'modB_on_fl':modB_on_fl, 'J_on_fl':J_on_fl,\
                'XYZ_on_s':s_XYZ, \
                'dXYZ_dS_on_s':s_dXYZ_dS, \
                'dXYZ_dVARPHI_on_s':s_dXYZ_dVARPHI, \
                'dXYZ_dTHETA_on_s': s_dXYZ_dTHETA, \
                'gradS_dot_gradTHETA_on_s':gradS_dot_gradTHETA, \
                'gradS_dot_gradVARPHI_on_s':gradS_dot_gradVARPHI, \
                'gradTHETA_dot_gradVARPHI_on_s':gradTHETA_dot_gradVARPHI, \
                'gradS_dot_gradS_on_s':gradS_dot_gradS,\
                'gradVARPHI_dot_gradVARPHI_on_s':gradVARPHI_dot_gradVARPHI,\
                'gradTHETA_dot_gradTHETA_s':gradTHETA_dot_gradTHETA,\
                'modB_on_s':modB, 'J_on_s':J,\
                'gradpar_on_fl': gradpar_on_fl,\
                'gradALPHA_dot_gradALPHA_on_fl' : gradALPHA_dot_gradALPHA_on_fl,\
                'gradS_dot_gradALPHA_on_fl' : gradS_dot_gradALPHA_on_fl,\
                'gradS_dot_B_cross_gradB_on_fl' : gradS_dot_B_cross_gradB_on_fl,\
                'gradALPHA_dot_B_cross_gradB_on_fl' : gradALPHA_dot_B_cross_gradB_on_fl,
                }

    if filename is not None:
        pointdata = {'S':S, 'VARPHI':VARPHI, 'THETA': THETA, 'gradS': tuple([dXYZ_dS[..., i].copy() for i in range(3)]), 'gradVARPHI': tuple([dXYZ_dVARPHI[..., i].copy() for i in range(3)]), 'gradTHETA': tuple([dXYZ_dTHETA[..., i].copy() for i in range(3)])}
        gridToVTK(filename, XYZ[..., 0].copy(), XYZ[..., 1].copy(), XYZ[..., 2].copy(), pointData=pointdata)

        variables = [k for k in out_dict.keys() if k != 'varphi_on_fl' and k !='theta_on_fl' and k[-1] != 's']
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        nrows = 4
        ncols = 2
        plot_variables = ['modB_on_fl', 'gradpar_on_fl', 'gradALPHA_dot_B_cross_gradB_on_fl', 'gradALPHA_dot_B_cross_gradB_on_fl', 'gradS_dot_B_cross_gradB_on_fl', 'gradALPHA_dot_gradALPHA_on_fl', 'gradS_dot_gradALPHA_on_fl', 'gradS_dot_gradS_on_fl']
        for j, variable in enumerate(plot_variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(out_dict['varphi_on_fl'], out_dict[variable])
            plt.xlabel(r'Boozer toroidal angle $\varphi$')
            plt.title(variable)
        plt.tight_layout()
        plt.savefig(filename+'.png')
    return out_dict

#

