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
        f = [0, 0, 0] if adjoint is None else 0
        if adjoint is None:
            f[0] -= g_grav*eta.dx(0)
            f[1] -= g_grav*eta.dx(1)
        else:
            adj = adjoint.split()[0]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f -= g_grav * i * inner(grad(eta), adj) * self.dx

        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0

        # TODO: explicit estimators
        total_h = self.get_total_depth(eta_old)
        head = eta
        grad_eta_by_parts = self.eta_is_dg
        i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        if adjoint is None:
            if grad_eta_by_parts:
                if uv is not None:
                    head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
                else:
                    head_star = avg(head)
                loc = i * self.normal[0]
                f[0] += g_grav*head_star*(loc('+') + loc('-')) * self.dS
                loc = i * self.normal[1]
                f[1] += g_grav*head_star*(loc('+') + loc('-')) * self.dS
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    if funcs is not None:
                        eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                        # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                        un_jump = inner(uv - uv_ext, self.normal)
                        eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                        f[0] += i * g_grav*(eta_rie-eta)*self.normal[0]*ds_bnd
                        f[1] += i * g_grav*(eta_rie-eta)*self.normal[1]*ds_bnd
                    if funcs is None or 'symm' in funcs:
                        # assume land boundary
                        # impermeability implies external un=0
                        un_jump = inner(uv, self.normal)
                        head_rie = head + sqrt(total_h/g_grav)*un_jump
                        f[0] += i * g_grav*head_rie*self.normal[0]*ds_bnd
                        f[1] += i * g_grav*head_rie*self.normal[1]*ds_bnd
        else:
            adj = adjoint.split()[0]
            if grad_eta_by_parts:
                if uv is not None:
                    head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
                else:
                    head_star = avg(head)
                loc = i * dot(adj, self.normal)
                f += g_grav*head_star*(loc('+') + loc('-')) * self.dS

                # Extra terms from second integration by parts
                loc = -i * g_grav*head*dot(adj, self.normal)
                f += (loc('+') + loc('-')) * self.dS + loc*ds(degree=self.quad_degree)

                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    if funcs is not None:
                        eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                        # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                        un_jump = inner(uv - uv_ext, self.normal)
                        eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                        f += i * g_grav*(eta_rie-eta)*dot(adj, self.normal)*ds_bnd
                    if funcs is None or 'symm' in funcs:
                        # assume land boundary
                        # impermeability implies external un=0
                        un_jump = inner(uv, self.normal)
                        head_rie = head + sqrt(total_h/g_grav)*un_jump
                        f += i * g_grav*head_rie*dot(adj, self.normal)*ds_bnd
            return -f


class HUDivResidual(ShallowWaterContinuityTerm):
    r"""
    Divergence term, :math:`\nabla \cdot (H \bar{\textbf{u}})`
    """
    name = 'HUDiv'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        total_h = self.get_total_depth(eta_old)

        f = [0, 0, 0] if adjoint is None else 0
        if adjoint is None:
            f[2] -= div(total_h*uv)
        else:
            adj = adjoint.split()[1]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f -= i * div(total_h*uv) * adj * self.dx

        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0
        # TODO: explicit estimators
        i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        total_h = self.get_total_depth(eta_old)

        hu_by_parts = self.u_continuity in ['dg', 'hdiv']

        if adjoint is None:
            raise NotImplementedError
        else:
            adj = adjoint.split()[1]
            if hu_by_parts:
                if self.eta_is_dg:
                    h = avg(total_h)
                    uv_rie = avg(uv) + sqrt(g_grav/h)*jump(eta, self.normal)
                    hu_star = h*uv_rie
                    loc = i * self.normal * adj
                    f += dot(hu_star, loc('+') + loc('-')) * self.dS

                    # Extra terms from second integration by parts
                    loc = -i * dot(total_h*uv, self.normal) * adj
                    f += (loc('+') + loc('-')) * self.dS + loc * ds(degree=self.quad_degree)

                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    if funcs is not None:
                        eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                        eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                        # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                        total_h_ext = self.get_total_depth(eta_ext_old)
                        h_av = 0.5*(total_h + total_h_ext)
                        eta_jump = eta - eta_ext
                        un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                        un_jump = inner(uv_old - uv_ext_old, self.normal)
                        eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                        h_rie = self.bathymetry + eta_rie
                        f += i * h_rie * un_rie * adj * ds_bnd
            return -f


class HorizontalAdvectionResidual(ShallowWaterMomentumTerm):
    r"""
    Advection of momentum term, :math:`\bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}}`
    """
    name = 'HorizontalAdvection'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0
        if not self.options.use_nonlinear_equations:
            return f

        if adjoint is None:
            ff = dot(uv_old, nabla_grad(uv))
            f[0] -= ff[0]
            f[1] -= ff[1]
        else:
            adj = adjoint.split()[0]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f -= i * inner(dot(uv_old, nabla_grad(uv)), adj) * self.dx

        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0
        # TODO: explicit estimators
        i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        if not self.options.use_nonlinear_equations:
            return -f

        if adjoint is None:
            raise NotImplementedError
        else:
            adj = adjoint.split()[0]
            if self.u_continuity in ['dg', 'hdiv']:
                un_av = dot(avg(uv_old), self.normal('-'))
                uv_up = avg(uv)
                loc = i*adj
                f += jump(uv_old, self.normal) * dot(uv_up, loc('+') + loc('-')) * self.dS

                # Extra terms from second integration by parts
                #loc = -i * dot(uv, adj) * dot(uv_old, self.normal)
                #loc = -i * dot(uv, self.normal) * dot(uv_old, adj)
                loc = -i*inner(outer(uv, self.normal), outer(uv_old, adj))
                f += (loc('+') + loc('-')) * self.dS + loc * ds(degree=self.quad_degree)

                if self.options.use_lax_friedrichs_velocity:
                    uv_lax_friedrichs = fields_old.get('lax_friedrichs_velocity_scaling_factor')
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    loc = i*adj
                    f += gamma*dot(loc('+') + loc('-'), jump(uv))*self.dS
                    for bnd_marker in self.boundary_markers:
                        funcs = bnd_conditions.get(bnd_marker)
                        ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                        if funcs is None:
                            # impose impermeability with mirror velocity
                            n = self.normal
                            uv_ext = uv - 2*dot(uv, n)*n
                            gamma = 0.5*abs(dot(uv_old, n))*uv_lax_friedrichs
                            f += i*gamma*dot(adj, uv-uv_ext)*ds_bnd
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    eta_jump = eta_old - eta_ext_old
                    total_h = self.get_total_depth(eta_old)
                    un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                    uv_av = 0.5*(uv_ext + uv)
                    f += i * dot(uv_av, adj) * un_rie * ds_bnd
            return -f



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

        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        if self.options.use_grad_div_viscosity_term:
            stress = nu*2.*sym(grad(uv))
        else:
            stress = nu*grad(uv)


        f = [0, 0, 0] if adjoint is None else 0
        if adjoint is None:
            ff = -div(stress)
            if self.options.use_grad_depth_viscosity_term:
                ff += -dot(grad(total_h)/total_h, stress)
            f[0] -= ff[0]
            f[1] -= ff[1]
        else:
            adj = adjoint.split()[0]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

            f -= -i * inner(div(stress), adj) * self.dx

            if self.options.use_grad_depth_viscosity_term:
                f -= -i * inner(dot(grad(total_h)/total_h, stress), adj) * self.dx

        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0
        # TODO: explicit estimators
        i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

        total_h = self.get_total_depth(eta_old)
        n = self.normal
        h = self.cellsize

        nu = fields_old.get('viscosity_h')
        if nu is None:
            return f

        if self.options.use_grad_div_viscosity_term:
            stress = nu*2.*sym(grad(uv))
            stress_jump = avg(nu)*2.*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        if adjoint is None:
            raise NotImplementedError
        else:
            adj = adjoint.split()[0]
            if self.u_continuity in ['dg', 'hdiv']:
                p = self.u_space.ufl_element().degree()
                alpha = 5.*p*(p+1)
                if p == 0:
                    alpha = 1.5
                loc = i * outer(adj, n)
                f += alpha/avg(h)*inner(loc('+') + loc('-'), stress_jump) * self.dS
                loc = i * grad(adj)
                f -= 0.5 * inner(loc('+') + loc('-'), stress_jump) * self.dS
                loc = i * outer(adj, n)
                f -= inner(loc('+') + loc('-'), avg(stress)) * self.dS

                # Extra terms from second integration by parts
                loc = i * inner(stress, outer(adj, n))
                f += (loc('+') + loc('-')) * self.dS + loc * ds(degree=self.quad_degree)

                # Dirichlet bcs only for DG
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    if funcs is not None:
                        if 'un' in funcs:
                            delta_uv = (dot(uv, n) - funcs['un'])*n
                        else:
                            eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                            if uv_ext is uv:
                                continue
                            delta_uv = uv - uv_ext
                        if self.options.use_grad_div_viscosity_term:
                            stress_jump = nu*2.*sym(outer(delta_uv, n))
                        else:
                            stress_jump = nu*outer(delta_uv, n)
                        f += (
                              i * alpha/h*inner(outer(adj, n), stress_jump)*ds_bnd
                              -i * inner(grad(adj), stress_jump)*ds_bnd
                              -i * inner(outer(adj, n), stress)*ds_bnd
                        )

            return -f


class CoriolisResidual(ShallowWaterMomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{u}}`
    """
    name = 'Coriolis'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        coriolis = fields_old.get('coriolis')
        f = [0, 0, 0] if adjoint is None else 0
        if coriolis is not None:
            if adjoint is None:
                f[0] -= -coriolis*uv[1]
                f[1] -= coriolis*uv[0]
            else:
                adj = adjoint.split()[0]
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f -= i * inner(coriolis * as_vector((-uv[1], uv[0])), adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


class WindStressResidual(ShallowWaterMomentumTerm):
    r"""
    Wind stress term, :math:`-\tau_w/(H \rho_0)`

    Here :math:`\tau_w` is a user-defined wind stress :class:`Function`.
    """
    name = 'WindStress'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        wind_stress = fields_old.get('wind_stress')
        total_h = self.get_total_depth(eta_old)
        f = [0, 0, 0] if adjoint is None else 0
        if wind_stress is not None:
            if adjoint is None:
                f[0] += wind_stress[0] / total_h / rho_0
                f[1] += wind_stress[1] / total_h / rho_0
            else:
                adj = adjoint.split()[0]
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f += i * inner(wind_stress / total_h / rho_0, adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


class AtmosphericPressureResidual(ShallowWaterMomentumTerm):
    r"""
    Atmospheric pressure term, :math:`\nabla (p_a / \rho_0)`

    Here :math:`p_a` is a user-defined atmospheric pressure :class:`Function`.
    """
    name = 'AtmosphericPressure'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        atmospheric_pressure = fields_old.get('atmospheric_pressure')
        f = [0, 0, 0] if adjoint is None else 0
        if atmospheric_pressure is not None:
            if adjoint is None:
                f[0] -= atmospheric_pressure.dx(0) / rho_0
                f[1] -= atmospheric_pressure.dx(1) / rho_0
            else:
                adj = adjoint.split()[0]
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f -= i * inner(grad(atmospheric_pressure) / rho_0, adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


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
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

        f = [0, 0, 0] if adjoint is None else 0
        if  C_D is not None:
            if adjoint is None:
                f[0] -= C_D * sqrt(dot(uv_old, uv_old)) * uv[0] / total_h
                f[1] -= C_D * sqrt(dot(uv_old, uv_old)) * uv[1] / total_h
            else:
                adj = adjoint.split()[0]
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f -= i * C_D * sqrt(dot(uv_old, uv_old)) * inner(uv, adj) / total_h * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


class LinearDragResidual(ShallowWaterMomentumTerm):
    r"""
    Linear friction term, :math:`C \bar{\textbf{u}}`

    Here :math:`C` is a user-defined drag coefficient.
    """
    name = 'LinearDrag'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = [0, 0, 0] if adjoint is None else 0
        if linear_drag_coefficient is not None:
            bottom_fri = linear_drag_coefficient*uv
            if adjoint is None:
                f -= bottom_fri
            else:
                adj = adjoint.split()[0]
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f -= i * inner(bottom_fri, adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


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
        f = [0, 0, 0] if adjoint is None else 0

        total_h = self.get_total_depth(eta_old)
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            bot_friction = stress
        else:
            return f

        if adjoint is None:
            f[0] -= bot_friction[0]
            f[1] -= bot_friction[1]
        else:
            adj = adjoint.split()[0]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f -= i * inner(bot_friction, adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


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
        i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
        if adjoint is not None:
            adj = adjoint.split()[0]

        f = [0, 0, 0] if adjoint is None else 0
        if self.options.tidal_turbine_farms != {}:
            total_h = self.get_total_depth(eta_old)
            for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
                density = farm_options.turbine_density
                C_T = farm_options.turbine_options.thrust_coefficient
                A_T = pi * (farm_options.turbine_options.diameter / 2.) ** 2
                C_D = (C_T * A_T * density) / 2.
                unorm = sqrt(dot(uv_old, uv_old))
                if adjoint is None:
                    indicator = assemble(i*self.dx(subdomain_id))
                    f[0] -= indicator * C_D * unorm * uv[0] / total_h
                    f[1] -= indicator * C_D * unorm * uv[1] / total_h
                else:
                    f -= i * C_D * unorm * inner(uv, adj) / total_h * self.dx(subdomain_id)
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


class MomentumSourceResidual(ShallowWaterMomentumTerm):
    r"""
    Generic source term :math:`\boldsymbol{\tau}` in the shallow water momentum equation.
    """
    name = 'MomentumSource'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0

        momentum_source = fields_old.get('momentum_source')
        if momentum_source is None:
            return f

        if adjoint is None:
            f[0] += momentum_source[0]
            f[1] += momentum_source[1]
        else:
            adj = adjoint.split()[0]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f += i * inner(momentum_source, adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


class ContinuitySourceResidual(ShallowWaterContinuityTerm):
    r"""
    Generic source term :math:`S` in the depth-averaged continuity equation.
    """
    name = 'ContinuitySource'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0

        volume_source = fields_old.get('volume_source')
        if volume_source is None:
            return f

        f = [0, 0, 0] if adjoint is None else 0
        if adjoint is None:
            f[2] += volume_source
        else:
            adj = adjoint.split()[1]
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            f += i * volume_source * adj * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


class BathymetryDisplacementMassResidual(ShallowWaterMomentumTerm):
    r"""
    Bathmetry mass displacement term, :math:`\partial \eta / \partial t + \partial \tilde{h} / \partial t`.
    """
    name = 'BathymetryDisplacementMass'

    def residual_cell(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0
        if self.options.use_wetting_and_drying:
            if adjoint is None:
                f[2] -= self.wd_bathymetry_displacement(eta)
            else:
                adj = adjoint.split()[0]
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f -= i * inner(self.wd_bathymetry_displacement(eta), adj) * self.dx
        return f

    def residual_edge(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        return [0, 0, 0] if adjoint is None else 0


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
        f = [0, 0, 0] if adjoint is None else 0
        for term in self.select_terms(label):
            r = term.residual_cell(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
            if adjoint is None:
                f[0] += r[0]
                f[1] += r[1]
                f[2] += r[2]
            else:
                f += r

        return f

    def edge_residual_uv_eta(self, label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint=None):
        f = [0, 0, 0] if adjoint is None else 0
        for term in self.select_terms(label):
            r = term.residual_edge(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
            if adjoint is None:
                f[0] += r[0]
                f[1] += r[1]
                f[2] += r[2]
            else:
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
        self.mesh = function_space.mesh()

    def mass_term(self, solution, adjoint):
        f, g = split(solution)
        if self.options.use_wetting_and_drying:
            f += -self.bathymetry_displacement_mass_residual.residual_cell(solution)
        if adjoint is None:
            return [f[0], f[1], g]
        else:
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            return i * inner(solution, adjoint) * dx

    def cell_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint=None):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        f = self.cell_residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
        return f

    def edge_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint=None):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        flux_terms = self.edge_residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions, adjoint)
        
        # Solve auxiliary problems to get traces on a particular element
        P0 = FunctionSpace(self.mesh, "DG", 0)
        res_trial = TrialFunction(P0)
        i = TestFunction(P0)
        mass_term = i*res_trial*dx
        if adjoint is None:
            res = [Function(P0), Function(P0), Function(P0)]
            for j in range(3):
                solve(mass_term == flux_terms[j], res[j])
        else:
            res = Function(P0)
            solve(mass_term == flux_terms, res)        

        return res
