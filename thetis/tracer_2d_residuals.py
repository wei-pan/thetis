r"""
Strong residual for 2D tracer equation.
"""
# TODO: More documentation
from __future__ import absolute_import
from .utility import *
from .equation import Equation
from .tracer_eq_2d import TracerTerm

__all__ = [
    'HorizontalAdvectionResidual',
    'HorizontalDiffusionResidual',
    'SourceResidual',
    'TracerResidual2D'
]


class HorizontalAdvectionResidual(TracerTerm):
    r"""
    Advection of tracer term, :math:`\bar{\textbf{u}} \cdot \nabla T`
    """
    name = 'HorizontalAdvection'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        if fields_old.get('uv_2d') is not None:
            uv = fields_old['uv_2d']
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

            f = 0
            if adjoint is None:
                f = dot(uv, grad(solution))
            else:
                f += i * dot(uv, grad(solution)) * adjoint * self.dx

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        if fields_old.get('uv_2d') is not None:
            if adjoint is None:
                raise NotImplementedError  # TODO
            else:
                f = 0
                if self.horizontal_dg:
                    uv = fields_old['uv_2d']
                    i = TestFunction(FunctionSpace(self.mesh, "DG", 0))

                    uv_av = avg(uv)
                    un_av = (uv_av[0]*self.normal('-')[0]
                             + uv_av[1]*self.normal('-')[1])
                    s = 0.5*(sign(un_av) + 1.0)
                    c_up = solution('-')*s + solution('+')*(1-s)

                    # Interface term
                    loc = i * dot(uv, self.normal) * adjoint
                    f += c_up * (loc('+') + loc('-')) * self.dS

                    # Term resulting from second integration by parts
                    loc = -i * dot(uv, self.normal) * solution * adjoint
                    f += (loc('+') + loc('-')) * self.dS + loc * ds(degree=self.quad_degree)

                    # TODO: Lax-Friedrichs

                    if bnd_conditions is not None:
                        for bnd_marker in self.boundary_markers:
                            funcs = bnd_conditions.get(bnd_marker)
                            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                            c_in = solution
                            if funcs is not None:
                                c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                                uv_av = 0.5*(uv + uv_ext)
                                un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                                s = 0.5*(sign(un_av) + 1.0)
                                c_up = (c_in-c_ext)*(1-s)  # c_in-c_ext is boundary residual
                                f += i * c_up*(uv_av[0]*self.normal[0]
                                               + uv_av[1]*self.normal[1]) * adjoint * ds_bnd
                return -f


class HorizontalDiffusionResidual(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
    """
    name = 'HorizontalDiffusion'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        if fields_old.get('diffusivity_h') is not None:
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                     [0, diffusivity_h, ]])
            diff_flux = dot(diff_tensor, grad(solution))

            f = 0
            if adjoint is None:
                f = -div(diff_flux)
            else:
                f += -i * div(diff_flux) * adjoint * self.dx

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        if adjoint is None:
            raise NotImplementedError  # TODO
        else:
            f = 0
            if self.horizontal_dg:
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                diffusivity_h = fields_old['diffusivity_h']
                diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                             [0, diffusivity_h, ]])

                degree_h = self.function_space.ufl_element().degree()
                sigma = 5.0*degree_h*(degree_h + 1)/self.cellsize
                if degree_h == 0:
                    sigma = 1.5 / self.cellsize
                alpha = avg(sigma)

                loc = i * self.normal * adjoint
                f += alpha*inner(dot(avg(diff_tensor), jump(solution, self.normal)),
                                 loc('+') + loc('-')) * self.dS
                f += -inner(jump(dot(diff_tensor, grad(solution))),
                            loc('+') + loc('-')) * self.dS
                loc = i * dot(diff_tensor, grad(adjoint))
                f += -0.5 *  inner(loc('+') + loc('-'),
                                   jump(solution, self.normal)) * self.dS

                loc = i * dot(dot(diff_tensor, grad(solution)), self.normal) * adjoint
                f += (loc('+') + loc('-')) * self.dS + loc * ds(degree=self.quad_degree)

            return -f


class SourceResidual(TracerTerm):
    """
    Generic source term
    """
    name = 'Source'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        source = fields_old.get('source')
        f = 0
        if source is not None:
            if adjoint is None:
                f = source
            else:
                i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
                f += i * source * adjoint * self.dx
        return f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        return 0


class TracerResidual2D(Equation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        """
        super(TracerResidual2D, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs)
        self.add_term(HorizontalAdvectionResidual(*args), 'explicit')
        self.add_term(HorizontalDiffusionResidual(*args), 'explicit')
        self.add_term(SourceResidual(*args), 'source')

    def mass_term(self, solution, adjoint):
        if adjoint is None:
            return solution
        else:
            i = TestFunction(FunctionSpace(self.mesh, "DG", 0))
            return i * inner(solution, adjoint) * dx

    def cell_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_cell(solution, solution_old, fields, fields_old, bnd_conditions, adjoint)
            if r is not None:
                f += r
        return f

    def edge_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint):
        flux_terms = 0
        for term in self.select_terms(label):
            r = term.residual_edge(solution, solution_old, fields, fields_old, bnd_conditions, adjoint)
            if r is not None:
                flux_terms += r

        # Solve an auxiliary problem to get traces on a particular element
        P0 = FunctionSpace(self.mesh, "DG", 0)
        res = TrialFunction(P0)
        i = TestFunction(P0)
        mass_term = i*res*dx
        res = Function(P0)
        solve(mass_term == flux_terms, res) 

        return i * res * dx
