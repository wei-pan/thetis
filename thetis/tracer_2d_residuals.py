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
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)

            if adjoint is None:
                f = dot(uv, grad(solution))
            else:
                f = I * dot(uv, grad(solution)) * adjoint * self.dx

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        if fields_old.get('uv_2d') is not None:
            if adjoint is None:
                raise NotImplementedError  # TODO
            else:
                uv = fields_old['uv_2d']
                P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
                I = TestFunction(P0)
                i = avg(I)

                uv_av = avg(uv)
                un_av = (uv_av[0]*self.normal('-')[0]
                         + uv_av[1]*self.normal('-')[1])
                s = 0.5*(sign(un_av) + 1.0)
                c_up = solution('-')*s + solution('+')*(1-s)

                # Interface term
                f = i * c_up*(jump(uv, self.normal)) * adjoint('+') * self.dS
                # TODO: Make symmetric (+ and - are arbitrary)

                # Term resulting from reverse integration by parts
                f += -i * dot(uv('+'), self.normal('+')) * solution('+') * adjoint('+') * self.dS
                # TODO: Make symmetric (+ and - are arbitrary)

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
                            f += I * c_up*(uv_av[0]*self.normal[0]
                                           + uv_av[1]*self.normal[1]) * adjoint * ds_bnd
            # TODO: Check works with CG space
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
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                     [0, diffusivity_h, ]])
            diff_flux = dot(diff_tensor, grad(solution))

            if adjoint is None:
                f = -div(diff_flux)
            else:
                f = -I * div(diff_flux) * adjoint * self.dx

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        if adjoint is None:
            raise NotImplementedError  # TODO
        else:
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)
            i = avg(I)
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                         [0, diffusivity_h, ]])

            f = I * inner(dot(diff_tensor, grad(solution)), self.normal) * adjoint * ds

            degree_h = self.function_space.ufl_element().degree()
            sigma = 5.0*degree_h*(degree_h + 1)/self.cellsize
            if degree_h == 0:
                sigma = 1.5 / self.cellsize
            alpha = avg(sigma)
            ds_interior = self.dS

            f += i * alpha*inner(dot(avg(diff_tensor), jump(solution, self.normal)),
                                self.normal('+'))*adjoint('+')*ds_interior
            # TODO: Make symmetric (+ and - are arbitrary)
            f += -0.5 * i * inner(dot(diff_tensor, grad(adjoint('+'))),
                                  jump(solution, self.normal))*ds_interior
            # TODO: Make symmetric (+ and - are arbitrary)
            f += 0.5 * i * inner(jump(dot(diff_tensor, grad(solution))),
                            self.normal('+'))*adjoint('+')*ds_interior
            # TODO: Make symmetric (+ and - are arbitrary)

            # TODO: Check works with CG space
            return -f


class SourceResidual(TracerTerm):
    """
    Generic source term
    """
    name = 'Source'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        source = fields_old.get('source')
        if source is not None:
            if adjoint is None:
                f = source
            else:
                P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
                I = TestFunction(P0)
                f = I * source * adjoint * self.dx
            return f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None, adjoint=None):
        return None


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

    def mass_term(self, solution):
        return solution

    def cell_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_cell(solution, solution_old, fields, fields_old, bnd_conditions, adjoint)
            if r is not None:
                f += r
        if adjoint is None:
            return f
        else:
            return assemble(f)

    def edge_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, adjoint):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_edge(solution, solution_old, fields, fields_old, bnd_conditions, adjoint)
            if r is not None:
                f += r
        if adjoint is None:
            return f
        else:
            return assemble(f)
