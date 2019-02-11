r"""
Strong residual for depth averaged shallow water equations
"""
# TODO: More documentation
from __future__ import absolute_import
import numpy as np

from .utility import *
from .equation import Equation
from .shallowwater_eq import ShallowWaterMomentumTerm, ShallowWaterContinuityTerm

__all__ = [
    'BaseShallowWaterResidual',
    'ShallowWaterResidual',
    'ShallowWaterMomentumResidual',
    'ShallowWaterContinuityResidual',
    'HUDivResidual',
    'ContinuitySourceResidual',
    'HorizontalAdvectionResidual',
    'HorizontalViscosityResidual',
    'ExternalPressureGradientResidual',
    'CoriolisResidual',
    'LinearDragResidual',
    'QuadraticDragResidual',
    'BottomDrag3DResidual',
    'MomentumSourceResidual',
    'WindStressResidual',
    'AtmosphericPressureResidual',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ExternalPressureGradientResidual(ShallowWaterMomentumTerm):
    r"""
    External pressure gradient term, :math:`g \nabla \eta`
    """
    name = 'ExternalPressureGradient'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        f = g_grav * I * inner(grad(eta), adj) * self.dx

        return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        total_h = self.get_total_depth(eta_old)
        head = eta
        grad_eta_by_parts = self.eta_is_dg
        adj = adjoint.split()[0]

        raise NotImplementedError  # FIXME


class HUDivResidual(ShallowWaterContinuityTerm):
    r"""
    Divergence term, :math:`\nabla \cdot (H \bar{\textbf{u}})`
    """
    name = 'HUDiv'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        total_h = self.get_total_depth(eta_old)
        adj = adjoint.split()[1]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        f = I * div(total_h*uv) * adj * self.dx

        return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        adj = adjoint.split()[1]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        i = avg(I)
        return None  # FIXME


class HorizontalAdvectionResidual(ShallowWaterMomentumTerm):
    r"""
    Advection of momentum term, :math:`\bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}}`
    """
    name = 'HorizontalAdvection'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if self.options.use_nonlinear_equations:
            f = I * inner(dot(uv_old, nabla_grad(uv)), adj) * self.dx

            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        i = avg(I)
        return None  # FIXME


class HorizontalViscosityResidual(ShallowWaterMomentumTerm):
    r"""
    Viscosity of momentum term

    If option :attr:`.ModelOptions.use_grad_div_viscosity_term` is ``True``, we
    use the symmetric viscous stress :math:`\boldsymbol{\tau}_\nu = \nu_h ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T )`.

    If option :attr:`.ModelOptions.use_grad_div_viscosity_term` is ``False``,
    we use viscous stress :math:`\boldsymbol{\tau}_\nu = \nu_h \nabla \bar{\textbf{u}}`.

    If option :attr:`.ModelOptions.use_grad_depth_viscosity_term` is ``True``, we also include
    the term

    .. math::
        \boldsymbol{\tau}_{\nabla H} = - \frac{\nu_h \nabla(H)}{H} \cdot ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T )

    as a source term.
    """
    name = 'HorizontalViscosity'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        total_h = self.get_total_depth(eta_old)
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        nu = fields_old.get('viscosity_h')
        if nu is not None:

            if self.options.use_grad_div_viscosity_term:
                stress = nu*2.*sym(grad(uv))
            else:
                stress = nu*grad(uv)

            f = -I * inner(div(stress), adj) * self.dx

            if self.options.use_grad_depth_viscosity_term:
                f += -I * inner(dot(grad(total_h)/total_h, stress), adj) * self.dx

            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        i = avg(I)
        return None  # FIXME


class CoriolisResidual(ShallowWaterMomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{u}}`
    """
    name = 'Coriolis'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        coriolis = fields_old.get('coriolis')
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if coriolis is not None:
            f = I * inner(coriolis * as_vector((-uv[1], uv[0])), adj) * self.dx
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class WindStressResidual(ShallowWaterMomentumTerm):
    r"""
    Wind stress term, :math:`-\tau_w/(H \rho_0)`

    Here :math:`\tau_w` is a user-defined wind stress :class:`Function`.
    """
    name = 'WindStress'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        wind_stress = fields_old.get('wind_stress')
        total_h = self.get_total_depth(eta_old)
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if wind_stress is not None:
            f = I * inner(wind_stress / total_h / rho_0, adj) * self.dx
            return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class AtmosphericPressureResidual(ShallowWaterMomentumTerm):
    r"""
    Atmospheric pressure term, :math:`\nabla (p_a / \rho_0)`

    Here :math:`p_a` is a user-defined atmospheric pressure :class:`Function`.
    """
    name = 'AtmosphericPressure'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        atmospheric_pressure = fields_old.get('atmospheric_pressure')
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if atmospheric_pressure is not None:
            f = I * inner(grad(atmospheric_pressure) / rho_0, adj) * self.dx
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class QuadraticDragResidual(ShallowWaterMomentumTerm):
    r"""
    Quadratic Manning bottom friction term
    :math:`C_D \| \bar{\textbf{u}} \| \bar{\textbf{u}}`

    where the drag term is computed with the Manning formula

    .. math::
        C_D = g \frac{\mu^2}{H^{1/3}}

    if the Manning coefficient :math:`\mu` is defined (see field :attr:`manning_drag_coefficient`).
    Otherwise :math:`C_D` is taken as a constant (see field :attr:`quadratic_drag_coefficient`).
    """
    name = 'QuadraticDrag'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        total_h = self.get_total_depth(eta_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

            f = I * C_D * sqrt(dot(uv_old, uv_old)) * inner(uv, adj) / total_h * self.dx
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class LinearDragResidual(ShallowWaterMomentumTerm):
    r"""
    Linear friction term, :math:`C \bar{\textbf{u}}`

    Here :math:`C` is a user-defined drag coefficient.
    """
    name = 'LinearDrag'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if linear_drag_coefficient is not None:
            bottom_fri = linear_drag_coefficient*uv
            f = I * inner(bottom_fri, adj) * self.dx
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class BottomDrag3DResidual(ShallowWaterMomentumTerm):
    r"""
    Bottom drag term consistent with the 3D mode,
    :math:`C_D \| \textbf{u}_b \| \textbf{u}_b`

    Here :math:`\textbf{u}_b` is the bottom velocity used in the 3D mode, and
    :math:`C_D` the corresponding bottom drag.
    These fields are computed in the 3D model.
    """
    name = 'BottomDrag3D'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        total_h = self.get_total_depth(eta_old)
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            bot_friction = stress
            f = I * inner(bot_friction, adj) * self.dx
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class TurbineDragResidual(ShallowWaterMomentumTerm):
    r"""
    Turbine drag parameterisation implemented through quadratic drag term
    :math:`c_t \| \bar{\textbf{u}} \| \bar{\textbf{u}}`

    where the turbine drag :math:`c_t` is related to the turbine thrust coefficient
    :math:`C_T`, the turbine diameter :math:`A_T`, and the turbine density :math:`d`
    (n/o turbines per unit area), by:

    .. math::
        c_t = (C_T A_T d)/2

    """
    name = 'TurbineDrag'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        if self.options.tidal_turbine_farms != {}:
            total_h = self.get_total_depth(eta_old)
            adj = adjoint.split()[0]
            I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f = 0
            for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
                density = farm_options.turbine_density
                C_T = farm_options.turbine_options.thrust_coefficient
                A_T = pi * (farm_options.turbine_options.diameter / 2.) ** 2
                C_D = (C_T * A_T * density) / 2.
                unorm = sqrt(dot(uv_old, uv_old))
                f += I * C_D * unorm * inner(uv, adj) / total_h * self.dx(subdomain_id)
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class MomentumSourceResidual(ShallowWaterMomentumTerm):
    r"""
    Generic source term :math:`\boldsymbol{\tau}` in the shallow water momentum equation.
    """
    name = 'MomentumSource'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        momentum_source = fields_old.get('momentum_source')
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        if momentum_source is not None:
            f = I * inner(momentum_source, adj) * self.dx
            return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class ContinuitySourceResidual(ShallowWaterContinuityTerm):
    r"""
    Generic source term :math:`S` in the depth-averaged continuity equation.
    """
    name = 'ContinuitySource'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        volume_source = fields_old.get('volume_source')
        adj = adjoint.split()[1]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        if volume_source is not None:
            f = I * volume_source * adj * self.dx
            return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class BathymetryDisplacementMassResidual(ShallowWaterMomentumTerm):
    r"""
    Bathmetry mass displacement term, :math:`\partial \eta / \partial t + \partial \tilde{h} / \partial t`.
    """
    name = 'BathymetryDisplacementMass'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        adj = adjoint.split()[0]
        I = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if self.options.use_wetting_and_drying:
            f = I * inner(self.wd_bathymetry_displacement(eta), adj) * self.dx
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return None


class BaseShallowWaterResidual(Equation):
    """
    Abstract base class for ShallowWaterResidual.
    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        super(BaseShallowWaterResidual, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientResidual(*args), 'implicit')
        self.add_term(HorizontalAdvectionResidual(*args), 'explicit')
        self.add_term(HorizontalViscosityResidual(*args), 'explicit')
        self.add_term(CoriolisResidual(*args), 'explicit')
        self.add_term(WindStressResidual(*args), 'source')
        self.add_term(AtmosphericPressureResidual(*args), 'source')
        self.add_term(QuadraticDragResidual(*args), 'explicit')
        self.add_term(LinearDragResidual(*args), 'explicit')
        self.add_term(BottomDrag3DResidual(*args), 'source')
        self.add_term(TurbineDragResidual(*args), 'implicit')
        self.add_term(MomentumSourceResidual(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(HUDivResidual(*args), 'implicit')
        self.add_term(ContinuitySourceResidual(*args), 'source')

    def cell_residual_uv_eta(self, label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_cell(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
            if r is not None:
                f += r

        return f

    def edge_residual_uv_eta(self, label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_edge(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
            if r is not None:
                f += r
        return f


class ShallowWaterResidual(BaseShallowWaterResidual):
    """
    Residual for 2D depth-averaged shallow water equations in non-conservative form.
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
        super(ShallowWaterResidual, self).__init__(function_space, bathymetry, options)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry, options)

        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, options)
        self.bathymetry_displacement_mass_residual = BathymetryDisplacementMassResidual(eta_test, eta_space, u_space, bathymetry, options)

        self.options = options

    def mass_term(self, solution):
        f, g = split(solution)
        if self.options.use_wetting_and_drying:
            f += -self.bathymetry_displacement_mass_residual.residual_cell(solution)
        return np.array([f, g])

    def cell_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint=None):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        f = self.cell_residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
        return assemble(f)

    def edge_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint=None):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        f = self.edge_residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
        return assemble(f)
