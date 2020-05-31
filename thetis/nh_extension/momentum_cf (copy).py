r"""
3D momentum equation in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from thetis.equation import Term, Equation

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']

class MomentumEquation(Equation):
    """
    Hydrostatic 3D momentum equation :eq:`mom_eq_split` for mode split models
    """
    def __init__(self, function_space,
                 bathymetry, options, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        # TODO rename for reflect the fact that this is eq for the split eqns
        super(MomentumEquation, self).__init__(function_space)
        self.function_space = function_space
        self.bathymetry = bathymetry
        self.options = options
        self.mesh = self.function_space.mesh()
        self.test = TestFunction(function_space)
        self.trial = TrialFunction(function_space)
        self.normal = FacetNormal(self.mesh)
        self.boundary_markers = sorted(self.function_space.mesh().exterior_facets.unique_markers)
        self.boundary_len = self.function_space.mesh().boundary_len
        # negigible depth set for wetting and drying
        self.threshold = self.options.wetting_and_drying_threshold
        # define measures with a reasonable quadrature degree
        p, q = self.function_space.ufl_element().degree()
        self.quad_degree = (2*p + 1, 2*q + 1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.dS = self.dS_h + self.dS_v
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)


    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        f = 0

        omega = fields['omega']
        eta = fields['elev_3d']
        h_3d = eta + self.bathymetry
        h_total = conditional(h_3d <= self.threshold, zero(h_3d.ufl_shape), h_3d)

        # momentum
        mom = conditional(h_3d <= self.threshold, zero(solution.ufl_shape), solution)
        vel = conditional(h_3d <= self.threshold, zero(solution.ufl_shape), solution / h_3d)

        # construct forms
        F1 = as_vector((mom[0] * vel[0] + 0.5 * g_grav * h_total**2, mom[0] * vel[1], mom[0] * vel[2]))
        F2 = as_vector((mom[0] * vel[1], mom[1] * vel[1] + 0.5 * g_grav * h_total**2, mom[1] * vel[2]))
        F3 = as_vector((mom[0] * omega, mom[1] * omega, mom[2] * omega))
        f += -(dot(Dx(self.test, 0), F1) + dot(Dx(self.test, 1), F2) + dot(Dx(self.test, 2), F3))*self.dx

        # intercell flux
        if False:#self.options.use_hllc_flux:
            # set up modified vectors and evaluate hllc fluxes
            w_plus = as_vector((h_total, mom[0], mom[1], mom[2], omega))('+')
            w_minus = as_vector((h_total, mom[0], mom[1], mom[2], omega))('-')
            flux_plus = self.interior_flux(self.normal('+'), eta.function_space(), w_plus, w_minus)
            flux_minus = self.interior_flux(self.normal('-'), eta.function_space(), w_minus, w_plus)
            f += (dot(flux_minus, self.test('-')) + dot(flux_plus, self.test('+')))*self.dS

        else:
            use_lax_friedrichs = self.options.use_lax_friedrichs_velocity
            include_elev_grad = True
            # advection
            uv_av = avg(vel)
            w_av = avg(omega) # in sigma form
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1]
                     + avg(omega)*self.normal('-')[2])
            s = 0.5*(sign(un_av) + 1.0)
            mom_up = mom('-')*s + mom('+')*(1-s)
            if True:
                f += (# equation in x direction
                      mom_up[0]*uv_av[0]*jump(self.test[0], self.normal[0])
                      + mom_up[0]*uv_av[1]*jump(self.test[0], self.normal[1])
                      + mom_up[0]*w_av*jump(self.test[0], self.normal[2])
                      # equation in y direction
                      + mom_up[1]*uv_av[0]*jump(self.test[1], self.normal[0])
                      + mom_up[1]*uv_av[1]*jump(self.test[1], self.normal[1])
                      + mom_up[1]*w_av*jump(self.test[1], self.normal[2])
                      # equation in z direction
                      + mom_up[2]*uv_av[0]*jump(self.test[2], self.normal[0])
                      + mom_up[2]*uv_av[1]*jump(self.test[2], self.normal[1])
                      + mom_up[2]*w_av*jump(self.test[2], self.normal[2]))*(self.dS_v + self.dS_h)
                # Lax-Friedrichs stabilization
                if use_lax_friedrichs:
                    lax_friedrichs_factor = self.options.lax_friedrichs_velocity_scaling_factor
                    gamma = 0.5*abs((uv_av[0]*self.normal('-')[0]
                                     + uv_av[1]*self.normal('-')[1]
                                     + w_av*self.normal('-')[2]))*lax_friedrichs_factor # TODO check if uv_p1 is essential
                    f += gamma*(jump(self.test[0])*jump(mom[0])
                                + jump(self.test[1])*jump(mom[1])
                                + jump(self.test[2])*jump(mom[2]))*(self.dS_v + self.dS_h) # TODO check if adding dS_v is ok for vert adv
            # NOTE Bottom impermeability condition is naturally satisfied
            # surf/bottom boundary conditions: closed at bed, symmetric at surf
            f += (# equation in x direction
                  mom[0]*vel[0]*self.test[0]*self.normal[0]
                  + mom[0]*vel[1]*self.test[0]*self.normal[1]
                  + mom[0]*omega*self.test[0]*self.normal[2]
                  # equation in y direction
                  + mom[1]*vel[0]*self.test[1]*self.normal[0]
                  + mom[1]*vel[1]*self.test[1]*self.normal[1]
                  + mom[1]*omega*self.test[1]*self.normal[2]
                  # equation in z direction
                  + mom[2]*vel[0]*self.test[2]*self.normal[0]
                  + mom[2]*vel[1]*self.test[2]*self.normal[1]
                  #+ mom[2]*omega*self.test[2]*self.normal[2] # fails with this in cases e.g. bb_bar
                  )*(self.ds_surf)
            # elevation gradient terms
            if include_elev_grad:
                h_star = avg(h_total)
                f += 0.5*g_grav*h_star*h_star*(jump(self.test[0], self.normal[0])
                                               + jump(self.test[1], self.normal[1]))*(self.dS_v + self.dS_h)
                f += 0.5*g_grav*h_total*h_total*(self.test[0] * self.normal[0] 
                                                 + self.test[1] * self.normal[1])*(self.ds_bottom + self.ds_surf)

        # add in boundary fluxes
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
            if funcs is None:
                un = dot(vel, self.normal)
                uv_ext = vel - 2*un*self.normal
                if use_lax_friedrichs:
                    gamma = 0.5*abs(un)*lax_friedrichs_factor
                    f += gamma*(self.test[0]*(mom[0] - h_total*uv_ext[0]) +
                                self.test[1]*(mom[1] - h_total*uv_ext[1]))*ds_bnd
                # elevation gradient terms
                if include_elev_grad:
                    # assume land boundary
                    f += 0.5*g_grav*h_total*h_total*(self.test[0] * self.normal[0] 
                                                     + self.test[1] * self.normal[1])*ds_bnd
            else:
                if 'elev3d' in funcs:
                    eta_ext = funcs['elev3d']
                    h_rie = 0.5*(eta + eta_ext) + self.bathymetry
                    f += 0.5*g_grav*h_rie*h_rie*(self.test[0] * self.normal[0] 
                                                 + self.test[1] * self.normal[1])*ds_bnd
                # TODO add bnd conditions w.r.t inflow velocity or flux


        # bathymetry gradient term
        bath_grad = as_vector((g_grav * h_total * Dx(self.bathymetry, 0), g_grav * h_total * Dx(self.bathymetry, 1), 0))
        f += -dot(bath_grad, self.test)*self.dx

        # source term in vector form
        source_mom = fields_old.get('source_mom')
        if source_mom is not None:
            f += -dot(source_mom, self.test)*self.dx

        return -f

    def interior_flux(self, N, V, wr, wl):
        """ 
        This evaluates the interior fluxes between the positively and negatively restricted vectors wr, wl.

        """
        hr, mur, mvr, mwr, omegar = wr[0], wr[1], wr[2], wr[3], wr[4]
        hl, mul, mvl, mwl, omegal = wl[0], wl[1], wl[2], wl[3], wl[4]

        E = self.threshold
        gravity = Function(V).assign(g_grav)
        g = conditional(And(hr < E, hl < E), zero(gravity('+').ufl_shape), gravity('+'))

        # Do HLLC flux
        hl_zero = conditional(hl <= 0, 0, 1)
        ur = conditional(hr <= 0, zero(as_vector((mur / hr, mvr / hr, mwr / hr)).ufl_shape),
                         hl_zero * as_vector((mur / hr, mvr / hr, mwr / hr)))
        hr_zero = conditional(hr <= 0, 0, 1)
        ul = conditional(hl <= 0, zero(as_vector((mul / hl, mvl / hl, mwl / hl)).ufl_shape),
                         hr_zero * as_vector((mul / hl, mvl / hl, mwl / hl)))
        vr = dot(ur, N)
        vl = dot(ul, N)
        # wave speed depending on wavelength
        c_minus = Min(vr - sqrt(g * hr), vl - sqrt(g * hl))
        c_plus = Min(vr + sqrt(g * hr), vl + sqrt(g * hl))
        # not divided by zero height
        y = (hl * c_minus * (c_plus - vl) - hr * c_plus * (c_minus - vr)) / (hl * (c_plus - vl) - hr * (c_minus - vr))
        c_s = conditional(abs(hr * (c_minus - vr) - hl * (c_plus - vl)) <= 1e-16, zero(y.ufl_shape), y)

        hu_u_l = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mul) / hl)
        hu_v_l = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mvl) / hl)
        hu_w_l = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mwl) / hl)
        hv_v_l = conditional(hl <= 0, zero(mvl.ufl_shape), (hr_zero * mvl * mvl) / hl)
        hv_w_l = conditional(hl <= 0, zero(mvl.ufl_shape), (hr_zero * mvl * mwl) / hl)
        # omega
        hu_o_l = conditional(hl <= 0, zero(mul.ufl_shape), hr_zero * mul * omegal)
        hv_o_l = conditional(hl <= 0, zero(mvl.ufl_shape), hr_zero * mvl * omegal)
        hw_o_l = conditional(hl <= 0, zero(mwl.ufl_shape), hr_zero * mwl * omegal)

        hu_u_r = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mur) / hr)
        hu_v_r = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mvr) / hr)
        hu_w_r = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mwr) / hr)
        hv_v_r = conditional(hr <= 0, zero(mvr.ufl_shape), (hl_zero * mvr * mvr) / hr)
        hv_w_r = conditional(hr <= 0, zero(mvr.ufl_shape), (hl_zero * mvr * mwr) / hr)
        # omega
        hu_o_r = conditional(hr <= 0, zero(mur.ufl_shape), hl_zero * mur * omegar)
        hv_o_r = conditional(hr <= 0, zero(mvr.ufl_shape), hl_zero * mvr * omegar)
        hw_o_r = conditional(hr <= 0, zero(mwr.ufl_shape), hl_zero * mwr * omegar)

       # velocityr = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mvr) / hr)
       # velocity_ur = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mur) / hr)
       # velocity_vr = conditional(hr <= 0, zero(mvr.ufl_shape), (hl_zero * mvr * mvr) / hr)

        F1r = as_vector((hu_u_r + 0.5 * g * hr**2,
                         hu_v_r,
                         0))
        F2r = as_vector((hu_v_r,
                         hv_v_r + 0.5 * g * hr**2,
                         0))
        F3r = as_vector((hu_o_r,
                         hv_o_r,
                         hw_o_r))
        F3r = as_vector((0,
                         0,
                         0))

        F1l = as_vector((hu_u_l + 0.5 * g * hl**2,
                         hu_v_l,
                         0))
        F2l = as_vector((hu_v_l,
                         hv_v_l + 0.5 * g * hl**2,
                         hv_w_l))
        F3l = as_vector((hu_o_l,
                         hv_o_l,
                         0))
        F3l = as_vector((0,
                         0,
                         0))

        F_plus = as_vector((F1r, F2r, F3r))
        F_minus = as_vector((F1l, F2l, F3l))

        W_plus = as_vector((mur, mvr, 0))
        W_minus = as_vector((mul, mvl, 0))

        y = ((sqrt(hr) * vr) + (sqrt(hl) * vl)) / (sqrt(hl) + sqrt(hr))
        y = 0.5 * (vl + vr) #+ sqrt(g * hr) - sqrt(g * hl)
        v_star = conditional(abs(sqrt(hl) + sqrt(hr)) <= 1e-16, zero(y.ufl_shape), y)
        # conditional to prevent dividing by zero
        y = ((c_minus - vr) / (c_minus - c_s)) * (W_plus -
                                                  as_vector((hr * (c_s - v_star) * N[0],
                                                             hr * (c_s - v_star) * N[1],
                                                             0)))
        w_plus = conditional(abs(c_minus - c_s) <= 1e-16, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_plus - vl) / (c_plus - c_s)) * (W_minus -
                                                as_vector((hl * (c_s - v_star) * N[0],
                                                           hl * (c_s - v_star) * N[1],
                                                           0)))
        w_minus = conditional(abs(c_plus - c_s) <= 1e-16, zero(y.ufl_shape), y)

        Flux = ((0.5 * dot(N, F_plus + F_minus)) +
                (0.5 * (-((abs(c_minus) - abs(c_s)) * w_minus) +
                        ((abs(c_plus) - abs(c_s)) * w_plus) +
                        (abs(c_minus) * W_plus) -
                        (abs(c_plus) * W_minus))))

        return Flux

