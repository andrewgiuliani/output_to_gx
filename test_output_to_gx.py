#!/usr/bin/env python
import unittest
import pandas as pd
import numpy as np
from output_to_gx import reparametrizeBoozer, output_to_gx, compute_surfaces
from simsopt._core import load
from simsopt.geo import BoozerSurface, Volume
from simsopt.geo import CurveRZFourier, CurveXYZFourier, ToroidalFlux, SurfaceXYZTensorFourier
from simsopt.field import BiotSavart

class BoozerSurfaceTests(unittest.TestCase):
    def test_devices(self):
        
        ID1 = 887713
        iota_profile1 = np.array([0.5007765829001171, 0.5004388941281233, 0.4995670761840123])
        tf_profile1 = np.array([0.0, 0.251052309282534, 1.0])

        ID2 = 672363 
        iota_profile2 = np.array([0.32510646555107375, 0.3288998333997113, 0.34026726655744766, 0.3591816994926266, 0.38563322588019033, 0.4194226996065182, 0.4598485446164341, 0.5055202692881338])
        tf_profile2 = np.array([0.0, 0.024422725969058666, 0.09635207405078684, 0.21208882681079555, 0.3663510986586298, 0.5531342185832635, 0.7665243506879468, 1.0])

        for (ID, iota, tf) in zip([ID1, ID2], [iota_profile1, iota_profile2], [tf_profile1, tf_profile2]):
            [surfaces, axis, coils] = load(f'files/serial{ID:07}.json')
            self.subtest_volume_values(axis, surfaces, coils, tf, iota)
            self.subtest_varphi(ID, iota, axis, surfaces, BiotSavart(coils))
            self.subtest_compute_quantities(axis, surfaces, iota, tf, BiotSavart(coils))

    def subtest_varphi(self, ID, iota_profile, axis, surfaces, field):
        axis_uniform = reparametrizeBoozer(axis, field=field)
        quadpoints_varphi = np.linspace(0, 1, surfaces[0].quadpoints_phi.size*surfaces[0].nfp, endpoint=False)
        axis = CurveXYZFourier(quadpoints_varphi, axis_uniform.order)
        axis.x = axis_uniform.x

        s0=surfaces[0]
        iota = iota_profile[0]
        current_sum = np.sum([np.abs(c.current.get_value()) for c in field.coils])
        G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
        err_prev = np.inf
        for i in range(10):
            boozer_surface = BoozerSurface(field, s0,  Volume(s0), s0.volume()/4)
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)
            snew = SurfaceXYZTensorFourier(quadpoints_phi=np.linspace(0, 1, s0.quadpoints_phi.size*s0.nfp, endpoint=False),\
                                           quadpoints_theta=np.linspace(0, 1, s0.quadpoints_theta.size*s0.nfp, endpoint=False),\
                                           nfp=s0.nfp,\
                                           stellsym=s0.stellsym,\
                                           mpol=s0.mpol, ntor=s0.ntor)
            snew.x = s0.x
            err = np.mean(np.linalg.norm(snew.gamma() - axis.gamma()[:, None, :], axis=-1))
            print(err, err/err_prev, s0.minor_radius())
            err_prev = err

    def subtest_compute_quantities(self, axis, surfaces, iotas, tf, field):
        out = output_to_gx(axis, surfaces, iotas, tf, field, s=0.5, npoints=512, filename='out')

    def subtest_volume_values(self, axis, surfaces, coils, tf_profile, iota_profile):
        current_sum = np.sum([np.abs(c.current.get_value()) for c in coils])
        G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
        tf_outer = ToroidalFlux(surfaces[-1], BiotSavart(coils)).J()
        tf_target = 0.123
        idx = np.argmin(np.abs(tf_profile[1:]-tf_target))
        phis = np.linspace(0, 1/surfaces[idx].nfp, 2*surfaces[idx].mpol+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*surfaces[idx].ntor+1, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=surfaces[idx].mpol, ntor=surfaces[idx].ntor, \
                quadpoints_phi=phis, quadpoints_theta=thetas,\
                stellsym=surfaces[idx].stellsym, nfp=surfaces[idx].nfp)
        surface.x = surfaces[idx].x
        
        boozer_surface = BoozerSurface(BiotSavart(coils), surface,  ToroidalFlux(surface, BiotSavart(coils)), tf_target*tf_outer)
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota_profile[idx], G=G0)
        
        phis = np.linspace(0, 1, surfaces[idx].nfp*(2*surfaces[idx].mpol+1), endpoint=False)
        thetas = np.linspace(0, 1, surfaces[idx].nfp*(2*surfaces[idx].ntor+1), endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=surfaces[idx].mpol, ntor=surfaces[idx].ntor, \
                quadpoints_phi=phis, quadpoints_theta=thetas,\
                stellsym=surfaces[idx].stellsym, nfp=surfaces[idx].nfp)
        surface.x = boozer_surface.surface.x
        
        for nsurfaces in [5, 10, 15]:
            tmp_surfaces, tmp_iota_profile, tmp_tf_profile = compute_surfaces(surfaces, coils, tf_profile, iota_profile, nsurfaces=nsurfaces)
            out = output_to_gx(axis, tmp_surfaces, tmp_iota_profile, tmp_tf_profile, BiotSavart(coils), s=0.123, npoints=512, filename='out')
            err1 = np.mean(np.linalg.norm(surface.gamma()-out['XYZ_on_s'], axis=-1))
            err2 = np.mean(np.linalg.norm(surface.gammadash1()/(2.*np.pi)-out['dXYZ_dVARPHI_on_s'], axis=-1))
            err3 = np.mean(np.linalg.norm(surface.gammadash2()/(2.*np.pi)-out['dXYZ_dTHETA_on_s'], axis=-1))
            print(err1, err2, err3)

