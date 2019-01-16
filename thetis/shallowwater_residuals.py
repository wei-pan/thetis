r"""
Strong residual for depth averaged shallow water equations
"""
# TODO: More documentation
from __future__ import absolute_import
from .utility import *
from .equation import Equation
from .shallowwater_eq import ShallowWaterMomentumTerm, ShallowWaterContinuityTerm


class ExternalPressureGradientResidual(ShallowWaterMomentumTerm):
    r"""
    External pressure gradient term, :math:`g \nabla \eta`
    """
    name = 'ExternalPressureGradient'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):

        f = g_grav * grad(eta)

        return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        raise NotImplementedError  # FIXME


class HUDivResidual(ShallowWaterContinuityTerm):
    r"""
    Divergence term, :math:`\nabla \cdot (H \bar{\textbf{u}})`
    """
    name = 'HUDiv'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        total_h = self.get_total_depth(eta_old)

        f = div(total_h*uv)

        return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        raise NotImplementedError  # FIXME


class HorizontalAdvectionResidual(ShallowWaterMomentumTerm):
    r"""
    Advection of momentum term, :math:`\bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}}`
    """
    name = 'HorizontalAdvection'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):

        if self.options.use_nonlinear_equations:
            f = dot(uv_old, nabla_grad(uv))

            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        raise NotImplementedError  # FIXME


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

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        total_h = self.get_total_depth(eta_old)

        nu = fields_old.get('viscosity_h')
        if nu is not None:

            if self.options.use_grad_div_viscosity_term:
                stress = nu*2.*sym(grad(uv))
            else:
                stress = nu*grad(uv)

            f = -div(stress)

            if self.options.use_grad_depth_viscosity_term:
                f += -dot(grad(total_h)/total_h, stress)

            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        raise NotImplementedError  # FIXME


class CoriolisResidual(ShallowWaterMomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{u}}`
    """
    name = 'Coriolis'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        coriolis = fields_old.get('coriolis')
        if coriolis is not None:
            f = coriolis * as_vector((-uv[1], uv[0]))
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


class WindStressResidual(ShallowWaterMomentumTerm):
    r"""
    Wind stress term, :math:`-\tau_w/(H \rho_0)`

    Here :math:`\tau_w` is a user-defined wind stress :class:`Function`.
    """
    name = 'WindStress'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        wind_stress = fields_old.get('wind_stress')
        total_h = self.get_total_depth(eta_old)
        if wind_stress is not None:
            f = wind_stress / total_h / rho_0
            return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


class AtmosphericPressureResidual(ShallowWaterMomentumTerm):
    r"""
    Atmospheric pressure term, :math:`\nabla (p_a / \rho_0)`

    Here :math:`p_a` is a user-defined atmospheric pressure :class:`Function`.
    """
    name = 'AtmosphericPressure'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        atmospheric_pressure = fields_old.get('atmospheric_pressure')
        if atmospheric_pressure is not None:
            f = grad(atmospheric_pressure) / rho_0
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
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

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        total_h = self.get_total_depth(eta_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

            f = C_D * sqrt(dot(uv_old, uv_old)) * uv / total_h
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


class LinearDragResidual(ShallowWaterMomentumTerm):
    r"""
    Linear friction term, :math:`C \bar{\textbf{u}}`

    Here :math:`C` is a user-defined drag coefficient.
    """
    name = 'LinearDrag'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        if linear_drag_coefficient is not None:
            bottom_fri = linear_drag_coefficient*uv
            f = bottom_fri
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
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

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        total_h = self.get_total_depth(eta_old)
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            bot_friction = stress
            f = bot_friction
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
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

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        total_h = self.get_total_depth(eta_old)
        f = 0
        mesh = uv.function_space().mesh()
        p0_test = TestFunction(FunctionSpace(mesh, "DG", 0))

        if self.options.tidal_turbine_farms != {}:
            for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
                indicator = assemble(p0_test/CellVolume(mesh) * dx(subdomain_id))
                density = farm_options.turbine_density
                C_T = farm_options.turbine_options.thrust_coefficient
                A_T = pi * (farm_options.turbine_options.diameter / 2.) ** 2
                C_D = (C_T * A_T * density) / 2.
                unorm = sqrt(dot(uv_old, uv_old))
                f += C_D * unorm * indicator * uv / total_h
            return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


class MomentumSourceResidual(ShallowWaterMomentumTerm):
    r"""
    Generic source term :math:`\boldsymbol{\tau}` in the shallow water momentum equation.
    """
    name = 'MomentumSource'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        momentum_source = fields_old.get('momentum_source')

        if momentum_source is not None:
            f = momentum_source
            return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


class ContinuitySourceResidual(ShallowWaterContinuityTerm):
    r"""
    Generic source term :math:`S` in the depth-averaged continuity equation.
    """
    name = 'ContinuitySource'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        volume_source = fields_old.get('volume_source')

        if volume_source is not None:
            f = volume_source
            return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


class BathymetryDisplacementMassResidual(ShallowWaterMomentumTerm):
    r"""
    Bathmetry mass displacement term, :math:`\partial \eta / \partial t + \partial \tilde{h} / \partial t`.
    """
    name = 'BathymetryDisplacementMass'

    def residual_cell(self, solution):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        if self.options.use_wetting_and_drying:
            f = self.wd_bathymetry_displacement(eta)
        return -f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        return None


# TODO: ShallowWaterResidual to combine these terms
