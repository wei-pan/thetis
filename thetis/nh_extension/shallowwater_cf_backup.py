r"""
Depth averaged shallow water equations in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from thetis.equation import Equation

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']

class BaseShallowWaterEquation(Equation):
    """
    Abstract base class for ShallowWaterEquations, ShallowWaterMomentumEquation
    and FreeSurfaceEquation.

    Provides common functionality to compute time steps and add either momentum
    or continuity terms.
    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        super(BaseShallowWaterEquation, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

class ShallowWaterEquations(BaseShallowWaterEquation):
    """
    2D depth-averaged shallow water equations in conservative form.

    This defines the full 2D SWE equations :eq:`swe_freesurf` -
    :eq:`swe_momentum`.
    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(ShallowWaterEquations, self).__init__(function_space, bathymetry, options)
        # define bunch of members needed to construct forms
        self.function_space = function_space
        self.mesh = self.function_space.mesh()
        self.test = TestFunction(function_space)
        self.trial = TrialFunction(function_space)
        self.normal = FacetNormal(self.mesh)
        self.boundary_markers = sorted(self.function_space.mesh().exterior_facets.unique_markers)
        self.boundary_len = self.function_space.mesh().boundary_len

        self.bathymetry = bathymetry
        self.options = options
        # negigible depth when wetting and drying appear
        self.threshold = self.options.wetting_and_drying_threshold
        # mesh dependent variables
        self.cellsize = CellSize(self.mesh)
        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())
        self.dS = dS(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())

    def mass_term(self, solution):
        f = super(ShallowWaterEquations, self).mass_term(solution)
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        f = 0
        h_2d, hu_2d, hv_2d = solution_old.split()

        # modified water height and velocity
        h_mod_plus = h_2d('+')#h_mod('+') #h_mod = conditional(h_2d <= self.threshold, zero(h_2d.ufl_shape), h_2d)
        h_mod_minus = h_2d('-')#h_mod('-')
        vel_u = conditional(h_2d <= self.threshold, zero(hu_2d.ufl_shape), hu_2d / h_2d)
        vel_v = conditional(h_2d <= self.threshold, zero(hv_2d.ufl_shape), hv_2d / h_2d)

        # modified momentum
        hu_mod_plus = (h_mod_plus) * vel_u('+')
        hu_mod_minus = (h_mod_minus) * vel_u('-')
        hv_mod_plus = (h_mod_plus) * vel_v('+')
        hv_mod_minus = (h_mod_minus) * vel_v('-')

        # conservative advection terms
        hu_v = hu_2d * vel_v#conditional(h_2d <= 0, zero(hu_2d.ufl_shape), (hu_2d * hv_2d) / h_2d)
        hu_u = hu_2d * vel_u#conditional(h_2d <= 0, zero(hu_2d.ufl_shape), (hu_2d * hu_2d) / h_2d)
        hv_v = hv_2d * vel_v#conditional(h_2d <= 0, zero(hv_2d.ufl_shape), (hv_2d * hv_2d) / h_2d)

        # horizontal advection and external pressure gradient
        F1 = as_vector((hu_2d, hu_u + (g_grav / 2) * (h_2d * h_2d), hu_v))
        F2 = as_vector((hv_2d, hu_v, hv_v + (g_grav / 2) * (h_2d * h_2d)))

        # set up modified state vectors and define fluxes
        w_plus = as_vector((h_mod_plus, hu_mod_plus, hv_mod_plus))
        w_minus = as_vector((h_mod_minus, hu_mod_minus, hv_mod_minus))
        flux_plus = self.InteriorFlux(self.normal('+'), self.function_space, w_plus, w_minus)
        flux_minus = self.InteriorFlux(self.normal('-'), self.function_space, w_minus, w_plus)

        f += -(dot(Dx(self.test, 0), F1) + dot(Dx(self.test, 1), F2))*self.dx
        f += (dot(flux_minus, self.test('-')) + dot(flux_plus, self.test('+')))*self.dS
        # add in boundary fluxes
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            flux_bnd = self.BoundaryFlux(self.function_space, solution_old, funcs)
            f += dot(flux_bnd, self.test)*ds_bnd

        # bathymetry gradient term
        bath_grad = as_vector((0, g_grav * h_2d * Dx(self.bathymetry, 0), g_grav * h_2d * Dx(self.bathymetry, 1)))
        f += -dot(bath_grad, self.test)*self.dx

        # source term in vector form
        source = fields_old.get('source')
        if source is not None:
            f += -dot(source, self.test)*self.dx

        return -f

    def InteriorFlux(self, N, V, wr, wl):
        """ 
        This evaluates the interior fluxes between the positively and negatively restricted vectors wr, wl.

        """
        hr, mur, mvr = wr[0], wr[1], wr[2]
        hl, mul, mvl = wl[0], wl[1], wl[2]

        E = self.threshold
        gravity = Function(V.sub(0)).assign(g_grav)
        g = conditional(And(hr < E, hl < E), zero(gravity('+').ufl_shape), gravity('+'))

        # Do HLLC flux
        hl_zero = conditional(hl <= 0, 0, 1)
        ur = conditional(hr <= 0, zero(as_vector((mur / hr, mvr / hr)).ufl_shape),
                         hl_zero * as_vector((mur / hr, mvr / hr)))
        hr_zero = conditional(hr <= 0, 0, 1)
        ul = conditional(hl <= 0, zero(as_vector((mul / hl, mvl / hl)).ufl_shape),
                         hr_zero * as_vector((mul / hl, mvl / hl)))
        vr = dot(ur, N)
        vl = dot(ul, N)
        # wave speed depending on wavelength
        c_minus = Min(vr - sqrt(g * hr), vl - sqrt(g * hl))
        c_plus = Min(vr + sqrt(g * hr), vl + sqrt(g * hl))
        # not divided by zero height
        y = (hl * c_minus * (c_plus - vl) - hr * c_plus * (c_minus - vr)) / (hl * (c_plus - vl) - hr * (c_minus - vr))
        c_s = conditional(eq((hr * (c_minus - vr)), (hl * (c_plus - vl))), zero(y.ufl_shape), y)

        velocityl = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mvl) / hl)
        velocity_ul = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mul) / hl)
        velocity_vl = conditional(hl <= 0, zero(mvl.ufl_shape), (hr_zero * mvl * mvl) / hl)
        velocityr = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mvr) / hr)
        velocity_ur = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mur) / hr)
        velocity_vr = conditional(hr <= 0, zero(mvr.ufl_shape), (hl_zero * mvr * mvr) / hr)

        F1r = as_vector((mur,
                         velocity_ur + ((g / 2) * (hr * hr)),
                         velocityr))
        F2r = as_vector((mvr,
                         velocityr,
                         velocity_vr + ((g / 2) * (hr * hr))))

        F1l = as_vector((mul,
                         velocity_ul + ((g / 2) * (hl * hl)),
                         velocityl))
        F2l = as_vector((mvl,
                         velocityl,
                         velocity_vl + ((g / 2) * (hl * hl))))

        F_plus = as_vector((F1r, F2r))
        F_minus = as_vector((F1l, F2l))

        W_plus = as_vector((hr, mur, mvr))
        W_minus = as_vector((hl, mul, mvl))

        y = ((sqrt(hr) * vr) + (sqrt(hl) * vl)) / (sqrt(hl) + sqrt(hr))
        y = 0.5 * (vl + vr) #+ sqrt(g * hr) - sqrt(g * hl)
        v_star = conditional(abs(sqrt(hl) + sqrt(hr)) <= 1e-8, zero(y.ufl_shape), y)
        # conditional to prevent dividing by zero
        y = ((c_minus - vr) / (c_minus - c_s)) * (W_plus -
                                                  as_vector((0,
                                                            hr * (c_s - v_star) * N[0],
                                                            hr * (c_s - v_star) * N[1])))
        w_plus = conditional(abs(c_minus - c_s) <= 1e-8, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_plus - vl) / (c_plus - c_s)) * (W_minus -
                                                as_vector((0,
                                                          hl * (c_s - v_star) * N[0],
                                                          hl * (c_s - v_star) * N[1])))
        w_minus = conditional(abs(c_plus - c_s) <= 1e-8, zero(y.ufl_shape), y)

        Flux = ((0.5 * dot(N, F_plus + F_minus)) +
                (0.5 * (-((abs(c_minus) - abs(c_s)) * w_minus) +
                        ((abs(c_plus) - abs(c_s)) * w_plus) +
                        (abs(c_minus) * W_plus) -
                        (abs(c_plus) * W_minus))))

        return Flux

    def BoundaryFlux(self, V, w, bc_funcs):
        """ 
        This evaluates the boundary flux between the vector and a solid reflective wall (temporarily zero velocity and same depth) or other boundary conditions options.
        Here, mur and mul denote outside and inside of momentum cell, respectively.

        """
        N = self.normal

        h, mu, mv = split(w)

        if bc_funcs is None: # TODO improve stability with increased time step size
            mul = Constant(0)
            mur = mu
            mvr = mv
            mvl = Constant(0)
            hr = h
            hl = h
        else:
            if 'inflow' in bc_funcs:
                mul = b.value.sub(1) # TODO
                mur = mu
                mvr = mv
                mvl = b.value.sub(2)
                hr = h
                hl = h
            if 'outflow' in bc_funcs:
                mul = mu
                mur = mu
                mvr = mv
                mvl = mv
                hr = h
                hl = h

        # Do HLLC flux
        ul = conditional(hl <= 0, zero(as_vector((mul / hl, mvl / hl)).ufl_shape),
                         as_vector((mul / hl, mvl / hl)))
        ur = conditional(hr <= 0, zero(as_vector((mur / hr, mvr / hr)).ufl_shape),
                         as_vector((mur / hr, mvr / hr)))
        vr = dot(ur, N)
        vl = dot(ul, N)
        # wave speed depending on wavelength
        c_minus = Min(vr - sqrt(g_grav * hr), vl - sqrt(g_grav * hl))
        c_plus = Min(vr + sqrt(g_grav * hr), vl + sqrt(g_grav * hl))
        # not divided by zero height
        y = (hl * c_minus * (c_plus - vl) - hr * c_plus * (c_minus - vr)) / (hl * (c_plus - vl) - hr * (c_minus - vr))
        c_s = conditional(abs(hr * (c_minus - vr) - hl * (c_plus - vl)) <= 1e-8, zero(y.ufl_shape), y)

        velocityl = conditional(hl <= 0, zero(mul.ufl_shape), (mul * mvl) / hl)
        velocity_ul = conditional(hl <= 0, zero(mul.ufl_shape), (mul * mul) / hl)
        velocity_ur = conditional(hr <= 0, zero(mul.ufl_shape), (mur * mur) / hr)
        velocityr = conditional(hr <= 0, zero(mul.ufl_shape), (mur * mvr) / hr)
        velocity_vr = conditional(hr <= 0, zero(mvr.ufl_shape), (mvr * mvr) / hr)
        velocity_vl = conditional(hl <= 0, zero(mvl.ufl_shape), (mvl * mvl) / hl)

        F1r = as_vector((mur,
                         velocity_ur + ((g_grav / 2) * (hr * hr)),
                         velocityr))
        F2r = as_vector((mvr,
                         velocityr,
                         velocity_vr + ((g_grav / 2) * (hr * hr))))

        F1l = as_vector((mul,
                         velocity_ul + ((g_grav / 2) * (hl * hl)),
                         velocityl))
        F2l = as_vector((mvl,
                         velocityl,
                         velocity_vl + ((g_grav / 2) * (hl * hl))))

        F_plus = as_vector((F1r, F2r))
        F_minus = as_vector((F1l, F2l))

        W_plus = as_vector((hr, mur, mvr))
        W_minus = as_vector((hl, mul, mvl))

        y = ((sqrt(hr) * vr) + (sqrt(hl) * vl)) / (sqrt(hl) + sqrt(hr))
        y = 0.5 * (vl + vr) #+ sqrt(g * hr) - sqrt(g * hl)
        v_star = conditional(abs(sqrt(hl) + sqrt(hr)) <= 1e-8, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_minus - vr) / (c_minus - c_s)) * (W_plus -
                                                  as_vector((0,
                                                            hl * (c_s - v_star) * N[0],
                                                            hl * (c_s - v_star) * N[1])))
        w_plus = conditional(abs(c_minus - c_s) <= 1e-8, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_plus - vl) / (c_plus - c_s)) * (W_minus -
                                                as_vector((0,
                                                          hr * (c_s - v_star) * N[0],
                                                          hr * (c_s - v_star) * N[1])))
        w_minus = conditional(abs(c_plus - c_s) <= 1e-8, zero(y.ufl_shape), y)

        Flux = ((0.5 * dot(N, F_plus + F_minus)) +
                (0.5 * (-((abs(c_minus) - abs(c_s)) * w_minus) +
                        ((abs(c_plus) - abs(c_s)) * w_plus) +
                        (abs(c_minus) * W_plus) -
                        (abs(c_plus) * W_minus))))

        return Flux


class ModeSplit2DEquations(BaseShallowWaterEquation):
    r"""
    2D depth-averaged shallow water equations for mode splitting schemes.

    Defines the equations :eq:`swe_freesurf_modesplit` -
    :eq:`swe_momentum_modesplit`.
    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        # TODO remove include_grad_* options as viscosity operator is omitted
        super(ModeSplit2DEquations, self).__init__(function_space, bathymetry, options)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry,
                                options)

        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, options)

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(MomentumSourceTerm(*args), 'source')
        self.add_term(AtmosphericPressureTerm(*args), 'source')

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)


class FreeSurfaceEquation(BaseShallowWaterEquation):
    """
    2D free surface equation :eq:`swe_freesurf` in conservative form.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry,
                 options):
        """
        :arg eta_test: test function of the elevation function space
        :arg eta_space: elevation function space
        :arg u_space: velocity function space
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(FreeSurfaceEquation, self).__init__(eta_space, bathymetry, options)
        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, options)
        self.bathymetry_displacement_mass_term = BathymetryDisplacementMassTerm(eta_test, eta_space, u_space, bathymetry, options)

    def mass_term(self, solution):
        f = super(ShallowWaterEquations, self).mass_term(solution)
        f += -self.bathymetry_displacement_mass_term.residual(solution)
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        uv = fields['uv']
        uv_old = fields_old['uv']
        eta = solution
        eta_old = solution_old
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)


class ShallowWaterMomentumEquation(BaseShallowWaterEquation):
    """
    2D depth averaged momentum equation :eq:`swe_momentum` in conservative form.
    """
    def __init__(self, u_test, u_space, eta_space,
                 bathymetry,
                 options):
        """
        :arg u_test: test function of the velocity function space
        :arg u_space: velocity function space
        :arg eta_space: elevation function space
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(ShallowWaterMomentumEquation, self).__init__(u_space, bathymetry, options)
        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry, options)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        uv = solution
        uv_old = solution_old
        eta = fields['eta']
        eta_old = fields_old['eta']
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)
