r"""
Strong residual for 3D tracer equation
"""
# TODO: More documentation
from __future__ import absolute_import
from .utility import *
from .equation import Equation
from .tracer_eq import TracerTerm

__all__ = [
    'HorizontalAdvectionResidual',
    'VerticalAdvectionResidual',
    'HorizontalDiffusionResidual',
    'VerticalDiffusionResidual',
    'SourceResidual',
    'TracerResidual'
]


class HorizontalAdvectionResidual(TracerTerm):
    r"""
    Horizontal advection term :math:`\nabla_h \cdot (\textbf{u} T)`
    """
    name = 'HorizontalAdvection'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_3d') is not None:
            uv = fields_old['uv_3d']
            f = div(solution*uv)

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # FIXME


class VerticalAdvectionResidual(TracerTerm):
    r"""
    Vertical advection term :math:`\partial (w T)/(\partial z)`
    """
    name = 'VerticalAdvection'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        w = fields_old.get('w')
        if w is not None:
            w_mesh = fields_old.get('w_mesh')
            vertvelo = w[2]
            if w_mesh is not None:
                vertvelo = w[2] - w_mesh

            return Dx(solution * vertvelo, 2)

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # FIXME


class HorizontalDiffusionResidual(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`
    """
    name = 'HorizontalDiffusion'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is not None:
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, 0],
                                     [0, diffusivity_h, 0],
                                     [0, 0, 0]])
            diff_flux = dot(diff_tensor, grad(solution))

            f = -div(diff_flux)

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # FIXME


class VerticalDiffusionResidual(TracerTerm):
    r"""
    Vertical diffusion term :math:`-\frac{\partial}{\partial z} \Big(\mu \frac{T}{\partial z}\Big)`
    """
    name = 'VerticalDiffusion'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_v') is not None:

            diffusivity_v = fields_old['diffusivity_v']

            f = -Dx(diffusivity_v * Dx(solution, 2), 2)

            return -f

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # FIXME


class SourceResidual(TracerTerm):
    """
    Generic source term
    """
    name = 'Source'

    def residual_cell(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        source = fields_old.get('source')
        if source is not None:
            return source

    def residual_edge(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        return None


class TracerResidual(Equation):
    """
    3D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True, use_lax_friedrichs=True):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(TracerResidual, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, use_symmetric_surf_bnd, use_lax_friedrichs)
        self.add_term(HorizontalAdvectionResidual(*args), 'explicit')
        self.add_term(VerticalAdvectionResidual(*args), 'explicit')
        self.add_term(HorizontalDiffusionResidual(*args), 'explicit')
        self.add_term(VerticalDiffusionResidual(*args), 'explicit')
        self.add_term(SourceResidual(*args), 'source')

    def mass_term(self, solution):
        return solution

    def cell_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, tag=''):
        f = 0
        print("\n{:}Cell residual norm contributions:".format(tag))
        for term in self.select_terms(label):
            r = term.residual_cell(solution, solution_old, fields, fields_old, bnd_conditions)
            if r is not None:
                f += r
                print("    {name:30s} {norm:.4e}".format(name=term.name, norm=norm(r)))
        return f

    def edge_residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions, tag=''):
        f = 0
        print("\n{:}Edge residual norm contributions:".format(tag))
        for term in self.select_terms(label):
            r = term.residual_edge(solution, solution_old, fields, fields_old, bnd_conditions)
            if r is not None:
                f += r
                print("    {name:30s} {norm:.4e}".format(name=term.name, norm=norm(r)))
        return f
