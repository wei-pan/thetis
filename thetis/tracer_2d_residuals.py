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

    def residual_cell(self, solution, solution_old, adjoint, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is not None:
            uv = fields_old['uv_2d']
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)
            f = I * dot(uv, grad(solution)) * adjoint * self.dx

            return -f

    def residual_edge(self, solution, solution_old, adjoint, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is not None:
            uv = fields_old['uv_2d']
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)
            i = avg(I)

            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)

            f = i * c_up*(jump(uv, self.normal)) * adjoint('+') * self.dS

            # TODO: Lax-Friedrichs

            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    c_in = solution
                    if funcs is None:
                        f += I * c_in * (uv[0]*self.normal[0]
                                         + uv[1]*self.normal[1]) * adjoint * ds_bnd
                    else:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        f += I * c_up*(uv_av[0]*self.normal[0]
                                       + uv_av[1]*self.normal[1]) * adjoint * ds_bnd
            return f


class HorizontalDiffusionResidual(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
    """
    name = 'HorizontalDiffusion'

    def residual_cell(self, solution, solution_old, adjoint, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is not None:
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                     [0, diffusivity_h, ]])
            diff_flux = dot(diff_tensor, grad(solution))

            f = -I * div(diff_flux) * adjoint * self.dx

            return -f

    def residual_edge(self, solution, solution_old, adjoint, fields, fields_old, bnd_conditions=None):
        #P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
        #I = TestFunction(P0)
        #i = avg(I)
        return None  # FIXME


class SourceResidual(TracerTerm):
    """
    Generic source term
    """
    name = 'Source'

    def residual_cell(self, solution, solution_old, adjoint, fields, fields_old, bnd_conditions=None):
        source = fields_old.get('source')
        if source is not None:
            P0 = FunctionSpace(solution.function_space().mesh(), "DG", 0)
            I = TestFunction(P0)
            return I * source * adjoint * self.dx

    def residual_edge(self, solution, solution_old, adjoint, fields, fields_old, bnd_conditions=None):
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

    def cell_residual(self, label, solution, solution_old, adjoint, fields, fields_old, bnd_conditions, tag=''):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_cell(solution, solution_old, adjoint, fields, fields_old, bnd_conditions)
            if r is not None:
                f += r
        return 0 if f == 0 else assemble(f)

    def edge_residual(self, label, solution, solution_old, adjoint, fields, fields_old, bnd_conditions, tag=''):
        f = 0
        for term in self.select_terms(label):
            r = term.residual_edge(solution, solution_old, adjoint, fields, fields_old, bnd_conditions)
            if r is not None:
                f += r
        return 0 if f == 0 else assemble(f)
