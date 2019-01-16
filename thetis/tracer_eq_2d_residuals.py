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
    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is not None:
            uv = fields_old['uv_2d']
            f = div(solution*uv)

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # FIXME

class HorizontalDiffusionResidual(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    """
    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is not None:
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                     [0, diffusivity_h, ]])
            diff_flux = dot(diff_tensor, grad(solution))

            f = -div(diff_flux)

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # FIXME

class SourceResidual(TracerTerm):
    """
    Generic source term
    """
    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        source = fields_old.get('source')
        if source is not None:
            return source

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
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

    def cell_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        f = 0
        print("\nCell residual norm contributions:")
        for term in self.select_terms(label):
            r = term.residual_cell(solution, solution_old, fields, fields_old, bnd_conditions)
            if r is not None:
                f += r
                print("    {name:30s} {norm:.4e}".format(name=term.name, norm=norm(r)))
        return f

    def edge_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        f = 0
        print("\nEdge residual norm contributions:")
        for term in self.select_terms(label):
            r = term.residual_edge(solution, solution_old, fields, fields_old, bnd_conditions)
            if r is not None:
                f += r
                print("    {name:30s} {norm:.4e}".format(name=term.name, norm=norm(r)))
        return f

