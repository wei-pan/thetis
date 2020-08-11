"""
Module for 3D barotropic/baroclinic non-hydrostatic solver
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_nh
from . import fluid_slide
from . import granular_cf
from . import momentum_nh
from . import tracer_nh
from . import sediment_nh
from . import coupled_timeintegrator_nh
from . import turbulence_nh
from .. import timeintegrator
from .. import rungekutta
from . import limiter_nh
import time as time_mod
from mpi4py import MPI
from .. import exporter
import weakref
from ..field_defs import field_metadata
from ..options import ModelOptions3d
from .. import callback
from ..log import *
from collections import OrderedDict

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']

class FlowSolver(FrozenClass):
    """
    Main object for 3D solver

    **Example**

    Create 2D mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create a 3D model with 6 uniform levels, and set some options
    (see :class:`.ModelOptions3d`)

    .. code-block:: python

        solver_obj = solver_nh.FlowSolver(mesh2d, bathymetry_2d, n_layers=6)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'SSPRK22'
        options.timestepper_options.use_automatic_timestep = False
        options.solve_salinity = False
        options.solve_temperature = False
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(2.0*exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 options=None, extrude_options=None, mesh_ls=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: 2D :class:`Function`
        :arg int n_layers: Number of layers in the vertical direction.
            Elements are distributed uniformly over the vertical.
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions3d` instance
        """
        self._initialized = False

        self.bathymetry_cg_2d = bathymetry_2d

        self.mesh2d = mesh2d
        # independent landslide mesh for granular flow
        self.mesh_ls = self.mesh2d
        if mesh_ls is not None:
            self.mesh_ls = mesh_ls
        """2D :class`Mesh`"""
        if extrude_options is None:
            extrude_options = {}
        self.mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)#extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d, **extrude_options)
        self.horizontal_domain_is_2d = self.mesh2d.geometric_dimension() == 2
        if self.horizontal_domain_is_2d:
            self.vert_ind = 2
        else:
            self.vert_ind = 1

        self.normal_2d = FacetNormal(self.mesh2d)
        self.normal = FacetNormal(self.mesh)
        self.boundary_markers = self.mesh.exterior_facets.unique_markers
        self.n_layers = n_layers
        """3D :class`Mesh`"""
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.mesh_ls.boundary_len = bnd_len
        self.mesh.boundary_len = bnd_len

        # override default options
        self.options = ModelOptions3d()
        """
        Dictionary of all options. A :class:`.ModelOptions3d` object.
        """
        if options is not None:
            self.options.update(options)

        self.dt = self.options.timestep
        """Time step"""
        self.dt_2d = self.options.timestep_2d
        """Time of the 2D solver"""
        self.M_modesplit = None
        """Mode split ratio (int)"""

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

        self.bnd_functions = {'shallow_water': {},
                              'landslide_motion': {},
                              'momentum': {},
                              'salt': {},
                              'temp': {},
                              'sediment': {},
                              }

        self.callbacks = callback.CallbackManager()
        """
        :class:`.CallbackManager` object that stores all callbacks
        """

        self.fields = FieldDict()
        """
        :class:`.FieldDict` that holds all functions needed by the solver
        object
        """

        self.function_spaces = AttrDict()
        """
        :class:`.AttrDict` that holds all function spaces needed by the
        solver object
        """

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self._simulation_continued = False
        self._isfrozen = True

    def compute_dx_factor(self):
        """
        Computes normalized distance between nodes in the horizontal direction

        The factor depends on the finite element space and its polynomial
        degree. It is used to compute maximal stable time steps.
        """
        p = self.options.polynomial_degree
        if self.options.element_family == 'rt-dg':
            # velocity space is essentially p+1
            p = self.options.polynomial_degree + 1
        # assuming DG basis functions on triangles
        l_r = p**2/3.0 + 7.0/6.0*p + 1.0
        factor = 0.5*0.25/l_r
        return factor

    def compute_dz_factor(self):
        """
        Computes a normalized distance between nodes in the vertical direction

        The factor depends on the finite element space and its polynomial
        degree. It is used to compute maximal stable time steps.
        """
        p = self.options.polynomial_degree
        # assuming DG basis functions in an interval
        l_r = 1.0/max(p, 1)
        factor = 0.5*0.25/l_r
        return factor

    def compute_dt_2d(self, u_scale):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :arg u_scale: User provided maximum advective velocity scale
        :type u_scale: float or :class:`Constant`
        """
        csize = self.fields.h_elem_size_2d
        bath = self.fields.bathymetry_2d
        fs = bath.function_space()
        bath_pos = Function(fs, name='bathymetry')
        bath_pos.assign(bath)
        min_depth = 0.05
        bath_pos.dat.data[bath_pos.dat.data < min_depth] = min_depth
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        solution = Function(fs)
        g = physical_constants['g_grav']
        u = (sqrt(g * bath_pos) + u_scale)
        a = inner(test, trial) * dx
        l = inner(test, csize / u) * dx
        solve(a == l, solution)
        dt = float(solution.dat.data.min())
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        dt *= self.compute_dx_factor()
        return dt

    def compute_dt_h_advection(self, u_scale):
        r"""
        Computes maximum explicit time step for horizontal advection

        .. math :: \Delta t = \frac{\Delta x}{U_{scale}}

        where :math:`U_{scale}` is estimated horizontal advective velocity.

        :arg u_scale: User provided maximum horizontal velocity scale
        :type u_scale: float or :class:`Constant`
        """
        u = u_scale
        if isinstance(u_scale, FiredrakeConstant):
            u = u_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/10.0/(self.options.polynomial_degree + 1)*min_dx/u
        min_dx *= self.compute_dx_factor()
        dt = min_dx/u
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_v_advection(self, w_scale):
        r"""
        Computes maximum explicit time step for vertical advection

        .. math :: \Delta t = \frac{\Delta z}{W_{scale}}

        where :math:`W_{scale}` is estimated vertical advective velocity.

        :arg w_scale: User provided maximum vertical velocity scale
        :type w_scale: float or :class:`Constant`
        """
        w = w_scale
        if isinstance(w_scale, FiredrakeConstant):
            w = w_scale.dat.data[0]
        min_dz = self.fields.v_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/1.5/(self.options.polynomial_degree + 1)*min_dz/w
        min_dz *= self.compute_dz_factor()
        dt = min_dz/w
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_diffusion(self, nu_scale):
        r"""
        Computes maximum explicit time step for horizontal diffusion.

        .. math :: \Delta t = \alpha \frac{(\Delta x)^2}{\nu_{scale}}

        where :math:`\nu_{scale}` is estimated diffusivity scale.
        """
        nu = nu_scale
        if isinstance(nu_scale, FiredrakeConstant):
            nu = nu_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        factor = 2.0
        if self.options.timestepper_type == 'LeapFrog':
            factor = 1.2
        min_dx *= factor*self.compute_dx_factor()
        dt = (min_dx)**2/nu
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_mesh_stats(self):
        """
        Computes number of elements, nodes etc and prints to sdtout
        """
        nnodes = self.function_spaces.P1_2d.dim()
        ntriangles = int(self.function_spaces.P1DG_2d.dim()/3)
        nlayers = self.mesh.topology.layers - 1
        nprisms = ntriangles*nlayers
        dofs_per_elem = len(self.function_spaces.H.finat_element.entity_dofs())
        ntracer_dofs = dofs_per_elem*nprisms
        min_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.min(), MPI.MIN)
        max_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.max(), MPI.MAX)
        min_v_size = self.comm.allreduce(self.fields.v_elem_size_3d.dat.data.min(), MPI.MIN)
        max_v_size = self.comm.allreduce(self.fields.v_elem_size_3d.dat.data.max(), MPI.MAX)

        print_output('2D mesh: {:} nodes, {:} triangles'.format(nnodes, ntriangles))
        print_output('3D mesh: {:} layers, {:} prisms'.format(nlayers, nprisms))
        print_output('Horizontal element size: {:.2f} ... {:.2f} m'.format(min_h_size, max_h_size))
        print_output('Vertical element size: {:.3f} ... {:.3f} m'.format(min_v_size, max_v_size))
        print_output('Element family: {:}, degree: {:}'.format(self.options.element_family, self.options.polynomial_degree))
        print_output('Number of tracer DOFs: {:}'.format(ntracer_dofs))
        print_output('Number of cores: {:}'.format(self.comm.size))
        print_output('Tracer DOFs per core: ~{:.1f}'.format(float(ntracer_dofs)/self.comm.size))

    def set_time_step(self):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions3d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions3d.timestep`.

        Once the time step is determined, will adjust it to be an integer
        fraction of export interval ``options.simulation_export_time``.
        """
        automatic_timestep = (hasattr(self.options.timestepper_options, 'use_automatic_timestep') and
                              self.options.timestepper_options.use_automatic_timestep)

        cfl2d = self.timestepper.cfl_coeff_2d
        cfl3d = self.timestepper.cfl_coeff_3d
        max_dt_swe = self.compute_dt_2d(self.options.horizontal_velocity_scale)
        max_dt_hadv = self.compute_dt_h_advection(self.options.horizontal_velocity_scale)
        max_dt_vadv = self.compute_dt_v_advection(self.options.vertical_velocity_scale)
        max_dt_diff = self.compute_dt_diffusion(self.options.horizontal_viscosity_scale)
        print_output('  - dt 2d swe: {:}'.format(max_dt_swe))
        print_output('  - dt h. advection: {:}'.format(max_dt_hadv))
        print_output('  - dt v. advection: {:}'.format(max_dt_vadv))
        print_output('  - dt viscosity: {:}'.format(max_dt_diff))
        max_dt_2d = cfl2d*max_dt_swe
        max_dt_3d = cfl3d*min(max_dt_hadv, max_dt_vadv, max_dt_diff)
        print_output('  - CFL adjusted dt: 2D: {:} 3D: {:}'.format(max_dt_2d, max_dt_3d))
        if not automatic_timestep:
            print_output('  - User defined dt: 2D: {:} 3D: {:}'.format(self.options.timestep_2d, self.options.timestep))
        self.dt = self.options.timestep
        self.dt_2d = self.options.timestep_2d
        if automatic_timestep:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            assert self.options.timestep_2d is not None
            assert self.options.timestep_2d > 0.0

        if self.dt_mode == 'split':
            if automatic_timestep:
                self.dt = max_dt_3d
                self.dt_2d = max_dt_2d
            # compute mode split ratio and force it to be integer
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        elif self.dt_mode == '2d':
            if automatic_timestep:
                self.dt = min(max_dt_2d, max_dt_3d)
            self.dt_2d = self.dt
            self.M_modesplit = 1
        elif self.dt_mode == '3d':
            if automatic_timestep:
                self.dt = max_dt_3d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        print_output('  - chosen dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        # fit dt to export time
        m_exp = int(np.ceil(self.options.simulation_export_time/self.dt))
        self.dt = float(self.options.simulation_export_time)/m_exp
        if self.dt_mode == 'split':
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        else:
            self.dt_2d = self.dt

        print_output('  - adjusted dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        print_output('dt = {0:f}'.format(self.dt))
        if self.dt_mode == 'split':
            print_output('2D dt = {0:f} {1:d}'.format(self.dt_2d, self.M_modesplit))
        sys.stdout.flush()

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0 = get_functionspace(self.mesh, 'DG', 0, 'DG', 0, name='P0')
        self.function_spaces.P1 = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1')
        self.function_spaces.P2 = get_functionspace(self.mesh, 'CG', 2, 'CG', 2, name='P2')
        self.function_spaces.P1v = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1v', vector=True)
        self.function_spaces.P1DG = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DG')
        self.function_spaces.P1DGv = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DGv', vector=True)

        # function spaces for (u,v) and w
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U = get_functionspace(self.mesh, 'RT', self.options.polynomial_degree+1, 'DG', self.options.polynomial_degree, name='U', hdiv=True)
            self.function_spaces.W = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'CG', self.options.polynomial_degree+1, name='W', hdiv=True)
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='U', vector=True)
            self.function_spaces.W = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='W', vector=True)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))

        self.function_spaces.Uint = self.function_spaces.U  # vertical integral of uv
        # tracers
        self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='H')
       # self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', 0, name='H')
        self.function_spaces.turb_space = self.function_spaces.P0

        # 2D spaces
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 2, name='P2_2d')
        self.function_spaces.P1v_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1v_2d', vector=True)
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DGv_2d', vector=True)
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'RT', self.options.polynomial_degree+1, name='U_2d')
        elif self.options.element_family == 'dg-dg':
            if self.horizontal_domain_is_2d:
                self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d', vector=True)
            else:
                self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
        self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d], name='V_2d')

        # define function spaces for baroclinic head and internal pressure gradient
        if self.options.use_quadratic_pressure:
            self.function_spaces.P2DGxP2 = get_functionspace(self.mesh, 'DG', 2, 'CG', 2, name='P2DGxP2')
            self.function_spaces.P2DG_2d = get_functionspace(self.mesh2d, 'DG', 2, name='P2DG_2d')
            if self.options.element_family == 'dg-dg':
                self.function_spaces.P2DGxP1DGv = get_functionspace(self.mesh, 'DG', 2, 'DG', 1, name='P2DGxP1DGv', vector=True, dim=2)
                self.function_spaces.H_bhead = self.function_spaces.P2DGxP2
                self.function_spaces.H_bhead_2d = self.function_spaces.P2DG_2d
                self.function_spaces.U_int_pg = self.function_spaces.P2DGxP1DGv
            elif self.options.element_family == 'rt-dg':
                self.function_spaces.H_bhead = self.function_spaces.P2DGxP2
                self.function_spaces.H_bhead_2d = self.function_spaces.P2DG_2d
                self.function_spaces.U_int_pg = self.function_spaces.U
        else:
            self.function_spaces.P1DGxP2 = get_functionspace(self.mesh, 'DG', 1, 'CG', 2, name='P1DGxP2')
            self.function_spaces.H_bhead = self.function_spaces.P1DGxP2
            self.function_spaces.H_bhead_2d = self.function_spaces.P1DG_2d
            self.function_spaces.U_int_pg = self.function_spaces.U

        # function spaces for granular landslide
        self.function_spaces.H_ls = get_functionspace(self.mesh_ls, 'DG', self.options.polynomial_degree)
        self.function_spaces.U_ls = get_functionspace(self.mesh_ls, 'DG', self.options.polynomial_degree, vector=True)
        self.function_spaces.V_ls = MixedFunctionSpace([self.function_spaces.H_ls, self.function_spaces.H_ls, self.function_spaces.H_ls])
        self.function_spaces.P1_ls = get_functionspace(self.mesh_ls, 'CG', 1)

        self._isfrozen = True

    def create_fields(self):
        """
        Creates all fields
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # mesh velocity etc fields must be in the same space as 3D coordinates
        coord_is_dg = element_continuity(self.mesh2d.coordinates.function_space().ufl_element()).horizontal == 'dg'
        if coord_is_dg:
            coord_fs = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
            coord_fs_2d = self.function_spaces.P1DG_2d
        else:
            coord_fs = self.function_spaces.P1
            coord_fs_2d = self.function_spaces.P1_2d

        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)
        # correct treatment of the split 2d functions
        uv_2d, eta2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = eta2d
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_2d = Function(self.function_spaces.P1v_2d)
            self.fields.z_bottom_2d = Function(coord_fs_2d)
            self.fields.bottom_drag_2d = Function(coord_fs_2d)

        self.fields.elev_3d = Function(self.function_spaces.H)
        self.fields.elev_cg_3d = Function(coord_fs)
        self.fields.elev_cg_2d = Function(coord_fs_2d)
        self.fields.uv_3d = Function(self.function_spaces.U)
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_3d = Function(self.function_spaces.P1v)
            self.fields.bottom_drag_3d = Function(coord_fs)
        self.fields.bathymetry_2d = Function(coord_fs_2d)
        self.fields.bathymetry_3d = Function(coord_fs)
        # z coordinate in the strecthed mesh
        self.fields.z_coord_3d = Function(coord_fs)
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref_3d = Function(coord_fs)

        # sigma coordinate
        self.sigma_coord = Function(coord_fs).project(self.mesh.coordinates[2]) # WPan added.
        self.z_coord_3d_old = Function(coord_fs)

        self.fields.uv_dav_3d = Function(self.function_spaces.U)
        self.fields.uv_dav_2d = Function(self.function_spaces.U_2d)
        self.fields.split_residual_2d = Function(self.function_spaces.U_2d)
        self.fields.uv_mag_3d = Function(self.function_spaces.P0)
        self.fields.uv_p1_3d = Function(self.function_spaces.P1v)
        self.fields.w_3d = Function(self.function_spaces.W)
        self.fields.hcc_metric_3d = Function(self.function_spaces.P1DG, name='mesh consistency')
        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d = Function(coord_fs)
            self.fields.w_mesh_surf_3d = Function(coord_fs)
            self.fields.w_mesh_surf_2d = Function(coord_fs_2d)
        if self.options.solve_salinity:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
        if self.options.solve_temperature:
            self.fields.temp_3d = Function(self.function_spaces.H, name='Temperature')
        if self.options.use_implicit_vertical_diffusion and self.options.use_parabolic_viscosity:
            self.fields.parab_visc_3d = Function(self.function_spaces.P1)
        if self.options.use_baroclinic_formulation:
            if self.options.use_quadratic_density:
                self.fields.density_3d = Function(self.function_spaces.P2DGxP2, name='Density')
            else:
                self.fields.density_3d = Function(self.function_spaces.H, name='Density')
            self.fields.baroc_head_3d = Function(self.function_spaces.H_bhead)
            self.fields.int_pg_3d = Function(self.function_spaces.U_int_pg, name='int_pg_3d')
        else:
            self.fields.density_3d = Function(self.function_spaces.H, name='Density').assign(self.options.rho_fluid) # WPan added.
        if self.options.coriolis_frequency is not None:
            if isinstance(self.options.coriolis_frequency, FiredrakeConstant):
                self.fields.coriolis_3d = self.options.coriolis_frequency
            else:
                self.fields.coriolis_3d = Function(self.function_spaces.P1)
                ExpandFunctionTo3d(self.options.coriolis_frequency, self.fields.coriolis_3d).solve()
        if self.options.wind_stress is not None:
            if isinstance(self.options.wind_stress, FiredrakeFunction):
                assert self.options.wind_stress.function_space().mesh().geometric_dimension() == 3, \
                    'wind stress field must be a 3D function'
                self.fields.wind_stress_3d = self.options.wind_stress
            elif isinstance(self.options.wind_stress, FiredrakeConstant):
                self.fields.wind_stress_3d = self.options.wind_stress
            else:
                raise Exception('Unsupported wind stress type: {:}'.format(type(self.options.wind_stress)))
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_3d(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.max_h_diff = Function(self.function_spaces.P1)
        if self.options.use_smagorinsky_viscosity:
            self.fields.smag_visc_3d = Function(self.function_spaces.P1)
        if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
            self.tracer_limiter = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.H)
        else:
            self.tracer_limiter = None
        if (self.options.use_limiter_for_velocity
                and self.options.polynomial_degree > 0
                and self.options.element_family == 'dg-dg'):
            self.uv_limiter = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.U)
        else:
            self.uv_limiter = None
        if self.options.use_turbulence:
            if self.options.turbulence_model_type == 'gls':
                # NOTE tke and psi should be in H as tracers ??
                self.fields.tke_3d = Function(self.function_spaces.turb_space)
                self.fields.psi_3d = Function(self.function_spaces.turb_space)
                # NOTE other turb. quantities should share the same nodes ??
                self.fields.eps_3d = Function(self.function_spaces.turb_space)
                self.fields.len_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
                # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
                self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
                self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
                self.turbulence_model = turbulence_nh.GenericLengthScaleModel(
                    weakref.proxy(self),
                    self.fields.tke_3d,
                    self.fields.psi_3d,
                    self.fields.uv_3d,
                    self.fields.get('density_3d'),
                    self.fields.len_3d,
                    self.fields.eps_3d,
                    self.fields.eddy_diff_3d,
                    self.fields.eddy_visc_3d,
                    self.fields.buoy_freq_3d,
                    self.fields.shear_freq_3d,
                    options=self.options.turbulence_model_options)
            elif self.options.turbulence_model_type == 'pacanowski':
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
                self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
                self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
                self.turbulence_model = turbulence_nh.PacanowskiPhilanderModel(
                    weakref.proxy(self),
                    self.fields.uv_3d,
                    self.fields.get('density_3d'),
                    self.fields.eddy_diff_3d,
                    self.fields.eddy_visc_3d,
                    self.fields.buoy_freq_3d,
                    self.fields.shear_freq_3d,
                    options=self.options.turbulence_model_options)
            else:
                raise Exception('Unsupported turbulence model: {:}'.format(self.options.turbulence_model))
        else:
            self.turbulence_model = None
        # copute total viscosity/diffusivity
        self.tot_h_visc = SumFunction()
        self.tot_h_visc.add(self.options.horizontal_viscosity)
        self.tot_h_visc.add(self.fields.get('smag_visc_3d'))
        self.tot_v_visc = SumFunction()
        self.tot_v_visc.add(self.options.vertical_viscosity)
        self.tot_v_visc.add(self.fields.get('eddy_visc_3d'))
        self.tot_v_visc.add(self.fields.get('parab_visc_3d'))
        self.tot_h_diff = SumFunction()
        self.tot_h_diff.add(self.options.horizontal_diffusivity)
        self.tot_v_diff = SumFunction()
        self.tot_v_diff.add(self.options.vertical_diffusivity)
        self.tot_v_diff.add(self.fields.get('eddy_diff_3d'))

        self.create_functions() # WPan added.

        self._isfrozen = True

    def create_functions(self):
        """
        Creates extra functions, including fields
        """
        self.bathymetry_dg_old = Function(self.function_spaces.H_2d)
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.bathymetry_cg_2d)
        self.bathymetry_3d_dg = Function(self.function_spaces.H)
        self.bathymetry_init = Function(self.function_spaces.H_2d).assign(self.bathymetry_dg)
        self.elev_2d_old = Function(self.function_spaces.H_2d)
        self.elev_2d_mid = Function(self.function_spaces.H_2d)
        self.elev_2d_tmp = Function(self.function_spaces.H_2d)
        self.elev_3d_old = Function(self.function_spaces.H)
        self.elev_3d_mid = Function(self.function_spaces.H)

        self.uv_2d_dg = Function(self.function_spaces.P1DGv_2d)
        self.uv_2d_old = Function(self.function_spaces.U_2d)
        self.uv_2d_mid = Function(self.function_spaces.U_2d)
        self.uv_3d_old = Function(self.function_spaces.U)
        self.uv_3d_mid = Function(self.function_spaces.U)
        self.uv_3d_tmp = Function(self.function_spaces.U)
        self.uv_dav_3d_mid = Function(self.function_spaces.U)
        self.uv_dav_2d_mid = Function(self.function_spaces.U_2d)
        self.w_3d_old = Function(self.function_spaces.W)
        self.w_3d_mid = Function(self.function_spaces.W)

        self.w_surface = Function(self.function_spaces.H_2d)
        self.w_interface = Function(self.function_spaces.H_2d)
        self.fields.w_nh = Function(self.function_spaces.H_2d)
        self.fields.q_3d = Function(FunctionSpace(self.mesh, 'CG', self.options.polynomial_degree+1))
        self.q_3d_old = Function(self.fields.q_3d.function_space())
        self.fields.q_2d = Function(self.function_spaces.P2_2d)
        self.q_2d_mid = Function(self.fields.q_2d.function_space())
        self.q_2d_old = Function(self.function_spaces.P2_2d)
        self.fields.ext_pg_3d = Function(self.function_spaces.U_int_pg)

        self.fields.uv_delta = Function(self.function_spaces.U_2d)
        self.fields.uv_delta_2 = Function(self.function_spaces.U_2d)
        self.fields.uv_delta_3 = Function(self.function_spaces.U_2d)
        self.fields.uv_nh = Function(self.function_spaces.U_2d)
        self.uv_av_1 = Function(self.function_spaces.U_2d)
        self.uv_av_2 = Function(self.function_spaces.U_2d)
        self.uv_av_3 = Function(self.function_spaces.U_2d)
        self.w_01 = Function(self.function_spaces.H_2d)
        self.w_12 = Function(self.function_spaces.H_2d)
        self.w_23 = Function(self.function_spaces.H_2d)
        self.function_spaces.q_mixed_two_layers = MixedFunctionSpace([self.function_spaces.P2_2d, self.function_spaces.P2_2d])
        self.function_spaces.q_mixed_three_layers = MixedFunctionSpace([self.function_spaces.P2_2d, self.function_spaces.P2_2d, self.function_spaces.P2_2d])
        self.q_mixed_two_layers = Function(self.function_spaces.q_mixed_two_layers)
        self.q_mixed_three_layers = Function(self.function_spaces.q_mixed_three_layers)
        self.q_0 = Function(self.function_spaces.P2_2d)
        self.q_1 = Function(self.function_spaces.P2_2d)
        self.q_2 = Function(self.function_spaces.P2_2d)
        self.fields.elev_nh = Function(self.function_spaces.H_bhead)
        self.solution_2d_old = Function(self.function_spaces.V_2d)
        self.solution_2d_tmp = Function(self.function_spaces.V_2d)

        for k in range(self.n_layers):
            setattr(self, 'uv_av_' + str(k+1), Function(self.function_spaces.U_2d))
            #self.__dict__['uv_av_' + str(k+1)] = Function(self.function_spaces.U_2d)
            setattr(self, 'w_' + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'w_' + str(k) + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'q_' + str(k+1), Function(self.function_spaces.P2_2d))
            if k == 0:
                list_fs = [self.function_spaces.P2_2d]
                setattr(self, 'w_' + str(k), Function(self.function_spaces.H_2d))
                setattr(self, 'q_' + str(k), Function(self.function_spaces.P2_2d))
                setattr(self, 'q_' + str(self.n_layers+1), Function(self.function_spaces.P2_2d))
            else:
                list_fs.append(self.function_spaces.P2_2d)
        self.function_spaces.q_mixed_n_layers = MixedFunctionSpace(list_fs)
        self.q_mixed_n_layers = Function(self.function_spaces.q_mixed_n_layers)

        self.fields.omega = Function(FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1))

        # IBM test, temporarily fail to implement
        coord_fs = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        self.xyz_coord = Function(coord_fs).project(self.mesh.coordinates)
        self.interface_detector = Function(self.function_spaces.P1)
        xyz = self.xyz_coord.dat.data
        det = self.interface_detector.dat.data
        assert xyz.shape[0] == det.shape[0]
        for i, x in enumerate(xyz):
            det[i] = 1.
            if abs(x[0] - 250) <= 0.2 and abs(x[1] - (-5)) <= 3.2:
                det[i] = 0.
            if abs(x[0] - 750) <= 0.2 and abs(x[1] - (-5)) <= 3.2:
                det[i] = 0.
            if abs(x[1] - (-8)) <= 0.2 and abs(x[0] - 500) <= 250.2:
                det[i] = 0.
            if abs(x[1] - (-2)) <= 0.2 and abs(x[0] - 500) <= 250.2:
                det[i] = 0.

        # for sediment transport
        if self.options.solve_sediment:
            self.fields.c_3d = Function(self.function_spaces.H)

        # rigid slide
        if self.options.slide_is_rigid:
            self.fields.h_ls = Function(self.function_spaces.H_ls)
            self.h_ls_old = Function(self.function_spaces.H_ls)
        # fluid slide
        if self.options.slide_is_viscous_fluid:
            self.fields.solution_ls = Function(self.function_spaces.V_2d)
            self.fields.uv_ls, self.fields.elev_ls = self.fields.solution_ls.split()
            self.solution_ls_old = Function(self.function_spaces.V_2d)
            self.solution_ls_tmp = Function(self.function_spaces.V_2d)
        # granular flow
        if self.options.flow_is_granular:
            self.fields.solution_ls = Function(self.function_spaces.V_ls)
            self.solution_ls_old = Function(self.function_spaces.V_ls)
            self.solution_ls_mid = Function(self.function_spaces.V_ls)
            self.solution_ls_tmp = Function(self.function_spaces.V_ls)
            self.fields.h_ls, self.fields.hu_ls, self.fields.hv_ls = self.fields.solution_ls.split()
            self.h_ls_old, self.hu_ls_old, self.hv_ls_old = self.solution_ls_old.split()
            self.bathymetry_ls = Function(self.function_spaces.H_ls)
            self.phi_i = Function(self.function_spaces.P1_ls).assign(self.options.phi_i)
            self.phi_b = Function(self.function_spaces.P1_ls).assign(self.options.phi_b)
            self.kap = Function(self.function_spaces.P1_ls)
            self.uv_div_ls = Function(self.function_spaces.P1_ls)
            self.strain_rate_ls = Function(self.function_spaces.P1_ls)
            self.grad_p_ls = Function(self.function_spaces.U_ls)
            self.grad_p = Function(self.function_spaces.U_2d)
            self.slope = Function(self.function_spaces.H_ls).interpolate(self.options.bed_slope[2])
            self.h_2d_ls = Function(self.function_spaces.P1_ls)
            self.h_2d_cg = Function(self.function_spaces.P1_2d)

        self.landslide = self.options.slide_is_rigid or self.options.slide_is_viscous_fluid or self.options.flow_is_granular
        self.fields.slide_source_2d = Function(self.function_spaces.H_2d)
        self.fields.slide_source_3d = Function(self.function_spaces.H)




    def create_equations(self):
        """
        Creates all dynamic equations and time integrators
        """
        if 'uv_3d' not in self.fields:
            self.create_fields()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        self.eq_operator_split = shallowwater_nh.ModeSplit2DEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)
        self.eq_operator_split.bnd_functions = self.bnd_functions['shallow_water']

        self.eq_sw_nh = shallowwater_nh.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)

        self.eq_sw_mom = shallowwater_nh.ShallowWaterMomentumEquation(
            TestFunction(self.function_spaces.U_2d),
            self.function_spaces.U_2d,
            self.function_spaces.H_2d,
            self.bathymetry_dg,
            self.options)

        # only elevation gradient term
        self.eq_mom_2d = shallowwater_nh.MomentumEquation2D(
            TestFunction(self.function_spaces.U_2d),
            self.function_spaces.U_2d,
            self.function_spaces.H_2d,
            self.bathymetry_dg,
            self.options)

        self.eq_free_surface = shallowwater_nh.FreeSurfaceEquation(
            TestFunction(self.function_spaces.H_2d),
            self.function_spaces.H_2d,
            self.function_spaces.U_2d,
            self.bathymetry_dg,
            self.options)

        # treat landslide as a viscous fluid
        if self.options.slide_is_viscous_fluid:
            self.eq_ls = fluid_slide.FluidSlideEquations(
            self.fields.solution_ls.function_space(),
            self.bathymetry_ls,
            self.options)

        if self.options.flow_is_granular:
            self.eq_ls = granular_cf.GranularEquations(
                self.fields.solution_ls.function_space(),
                self.bathymetry_ls,
                self.options
            )
            self.eq_ls.bnd_functions = self.bnd_functions['landslide_motion']

        if self.options.use_wetting_and_drying:
            self.wd_modification = wetting_and_drying_modification(self.function_spaces.H_2d)
            self.wd_modification_ls = wetting_and_drying_modification(self.function_spaces.H_ls)

        self.copy_bath_to_3d = ExpandFunctionTo3d(self.bathymetry_dg, self.bathymetry_3d_dg)
        self.copy_bath_to_3d.solve()
        # landslide
        if self.landslide:
            self.copy_slide_source_to_3d = ExpandFunctionTo3d(self.fields.slide_source_2d, self.fields.slide_source_3d)
        self.limiter_h = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.H_2d)
        self.limiter_u = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.U_2d)


        ##################################
        # sediment transport equation
        if self.options.solve_sediment:
            assert (not self.options.solve_salinity) and (not self.options.solve_temperature), \
                   'Sediment transport equation is being solved... \
                    Temporarily it is not supported to solve other tracers simultaneously.'
        if self.options.solve_sediment:
            self.eq_sediment = sediment_nh.SedimentEquation(self.fields.c_3d.function_space(),
                                                            bathymetry=self.fields.bathymetry_3d,
                                                            v_elem_size=self.fields.v_elem_size_3d,
                                                            h_elem_size=self.fields.h_elem_size_3d,
                                                            use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                            use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            if self.options.use_implicit_vertical_diffusion:
                self.eq_sediment_vdff = sediment_nh.SedimentEquation(self.fields.c_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        ##################################

        expl_bottom_friction = self.options.use_bottom_friction and not self.options.use_implicit_vertical_diffusion
        self.eq_momentum = momentum_nh.MomentumEquation(self.fields.uv_3d.function_space(),
                                                        bathymetry=self.fields.bathymetry_3d,
                                                        v_elem_size=self.fields.v_elem_size_3d,
                                                        h_elem_size=self.fields.h_elem_size_3d,
                                                        use_nonlinear_equations=self.options.use_nonlinear_equations,
                                                        use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                        use_bottom_friction=expl_bottom_friction)

        if self.options.use_implicit_vertical_diffusion:
            self.eq_vertmomentum = momentum_nh.MomentumEquation(self.fields.uv_3d.function_space(),
                                                                bathymetry=self.fields.bathymetry_3d,
                                                                v_elem_size=self.fields.v_elem_size_3d,
                                                                h_elem_size=self.fields.h_elem_size_3d,
                                                                use_nonlinear_equations=False, # i.e. advection terms neglected
                                                                use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                                use_bottom_friction=self.options.use_bottom_friction)
        if self.options.solve_salinity:
            self.eq_salt = tracer_nh.TracerEquation(self.fields.salt_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            if self.options.use_implicit_vertical_diffusion:
                self.eq_salt_vdff = tracer_nh.TracerEquation(self.fields.salt_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        if self.options.solve_temperature:
            self.eq_temp = tracer_nh.TracerEquation(self.fields.temp_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            if self.options.use_implicit_vertical_diffusion:
                self.eq_temp_vdff = tracer_nh.TracerEquation(self.fields.temp_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solve_sediment:
            self.eq_sediment.bnd_functions = self.bnd_functions['sediment']
        if self.options.solve_salinity:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.solve_temperature:
            self.eq_temp.bnd_functions = self.bnd_functions['temp']
        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if self.options.use_turbulence_advection:
                # explicit advection equations
                self.eq_tke_adv = tracer_nh.TracerEquation(self.fields.tke_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
                self.eq_psi_adv = tracer_nh.TracerEquation(self.fields.psi_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence_nh.TKEEquation(self.fields.tke_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)
            self.eq_psi_diff = turbulence_nh.PsiEquation(self.fields.psi_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)

        # ----- Time integrators
        self.dt_mode = '3d'  # 'split'|'2d'|'3d' use constant 2d/3d dt, or split
        if self.options.timestepper_type == 'LeapFrog':
            raise Exception('Not surpport this time integrator: '+str(self.options.timestepper_type))
            self.timestepper = coupled_timeintegrator_nh.CoupledLeapFrogAM3(weakref.proxy(self))
        elif self.options.timestepper_type == 'SSPRK22':
            self.timestepper = coupled_timeintegrator_nh.CoupledTwoStageRK(weakref.proxy(self))
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))

        # ----- File exporters
        # create export_managers and store in a list
        self.exporters = OrderedDict()
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.output_directory,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            hdf5_dir = os.path.join(self.options.output_directory, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        # ----- Operators
        #tot_uv_3d = self.fields.uv_3d + self.fields.uv_dav_3d
        tot_uv_3d = self.fields.uv_3d # modified for operator-splitting method used # TODO note here
        self.w_solver = VerticalVelocitySolver(self.fields.w_3d,
                                               tot_uv_3d,
                                               self.fields.bathymetry_3d,
                                               self.eq_momentum.bnd_functions)
        if self.horizontal_domain_is_2d:
            zero_bnd_value = Constant((0.0, 0.0, 0.0))
        else:
            zero_bnd_value = Constant((0.0, 0.0))
        self.uv_averager = VerticalIntegrator(self.fields.uv_3d,
                                              self.fields.uv_dav_3d,
                                              bottom_to_top=True,
                                              bnd_value=zero_bnd_value,
                                              average=True,
                                              bathymetry=self.fields.bathymetry_3d,
                                              elevation=self.fields.elev_cg_3d)
        if self.options.use_baroclinic_formulation:
            if self.options.solve_salinity:
                s = self.fields.salt_3d
            else:
                s = self.options.constant_salinity
            if self.options.solve_temperature:
                t = self.fields.temp_3d
            else:
                t = self.options.constant_temperature
            if self.options.equation_of_state_type == 'linear':
                eos_options = self.options.equation_of_state_options
                self.equation_of_state = LinearEquationOfState(eos_options.rho_ref,
                                                               eos_options.alpha,
                                                               eos_options.beta,
                                                               eos_options.th_ref,
                                                               eos_options.s_ref)
            else:
                self.equation_of_state = JackettEquationOfState()
            if self.options.solve_sediment:
                self.density_solver = DensitySolverSediment(self.fields.c_3d, self.fields.density_3d, 
                                                    self.options.rho_slide, self.options.rho_fluid)
            elif self.options.use_quadratic_density:
                self.density_solver = DensitySolverWeak(s, t, self.fields.density_3d,
                                                        self.equation_of_state)
            else:
                self.density_solver = DensitySolver(s, t, self.fields.density_3d,
                                                    self.equation_of_state)
            self.rho_integrator = VerticalIntegrator(self.fields.density_3d,
                                                     self.fields.baroc_head_3d,
                                                     bottom_to_top=False,
                                                     average=False,
                                                     bathymetry=self.fields.bathymetry_3d,
                                                     elevation=self.fields.elev_cg_3d)
            self.int_pg_calculator = momentum_nh.InternalPressureGradientCalculator(
                self.fields, self.options,
                self.bnd_functions['momentum'],
                solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)
        self.extract_surf_dav_uv = SubFunctionExtractor(self.fields.uv_dav_3d,
                                                        self.fields.uv_dav_2d,
                                                        boundary='top', elem_facet='top',
                                                        elem_height=self.fields.v_elem_size_2d)
        self.copy_elev_to_3d = ExpandFunctionTo3d(self.fields.elev_2d, self.fields.elev_3d)
        self.copy_elev_cg_to_3d = ExpandFunctionTo3d(self.fields.elev_cg_2d, self.fields.elev_cg_3d) # seems ok to delete?
        self.copy_uv_dav_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                                                           elem_height=self.fields.v_elem_size_3d)
        self.copy_uv_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                                       elem_height=self.fields.v_elem_size_3d)
        self.uv_mag_solver = VelocityMagnitudeSolver(self.fields.uv_mag_3d, u=self.fields.uv_3d)
        if self.options.use_bottom_friction:
            self.extract_uv_bottom = SubFunctionExtractor(self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                                                          boundary='bottom', elem_facet='average',
                                                          elem_height=self.fields.v_elem_size_2d)
            self.extract_z_bottom = SubFunctionExtractor(self.fields.z_coord_3d, self.fields.z_bottom_2d,
                                                         boundary='bottom', elem_facet='average',
                                                         elem_height=self.fields.v_elem_size_2d)
            if self.options.use_parabolic_viscosity:
                self.copy_uv_bottom_to_3d = ExpandFunctionTo3d(self.fields.uv_bottom_2d,
                                                               self.fields.uv_bottom_3d,
                                                               elem_height=self.fields.v_elem_size_3d)
                self.copy_bottom_drag_to_3d = ExpandFunctionTo3d(self.fields.bottom_drag_2d,
                                                                 self.fields.bottom_drag_3d,
                                                                 elem_height=self.fields.v_elem_size_3d)
        self.mesh_updater = ALEMeshUpdater(self)

        if self.options.use_smagorinsky_viscosity:
            self.smagorinsky_diff_solver = SmagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                                                self.options.smagorinsky_coefficient, self.fields.h_elem_size_3d,
                                                                self.fields.max_h_diff,
                                                                weak_form=self.options.polynomial_degree == 0)
        if self.options.use_parabolic_viscosity:
            self.parabolic_viscosity_solver = ParabolicViscosity(self.fields.uv_bottom_3d,
                                                                 self.fields.bottom_drag_3d,
                                                                 self.fields.bathymetry_3d,
                                                                 self.fields.parab_visc_3d)
        self.uv_p1_projector = Projector(self.fields.uv_3d, self.fields.uv_p1_3d)
        self.elev_3d_to_cg_projector = Projector(self.fields.elev_3d, self.fields.elev_cg_3d)
        self.elev_2d_to_cg_projector = Projector(self.fields.elev_2d, self.fields.elev_cg_2d)

        # ----- set initial values
        self.fields.bathymetry_2d.project(self.bathymetry_cg_2d)
        ExpandFunctionTo3d(self.fields.bathymetry_2d, self.fields.bathymetry_3d).solve()
        self.mesh_updater.initialize()
        self.compute_mesh_stats()
        self.set_time_step()
        self.timestepper.set_dt(self.dt, self.dt_2d)
        # compute maximal diffusivity for explicit schemes
        degree_h, degree_v = self.function_spaces.H.ufl_element().degree()
        max_diff_alpha = 1.0/60.0/max((degree_h*(degree_h + 1)), 1.0)  # FIXME depends on element type and order
        self.fields.max_h_diff.assign(max_diff_alpha/self.dt * self.fields.h_elem_size_3d**2)
        d = self.fields.max_h_diff.dat.data
        print_output('max h diff {:} - {:}'.format(d.min(), d.max()))

        self.next_export_t = self.simulation_time + self.options.simulation_export_time
        self._initialized = True
        self._isfrozen = True

    def assign_initial_conditions(self, elev=None, salt=None, temp=None,
                                  uv_2d=None, uv_3d=None, tke=None, psi=None,
                                  elev_slide=None, uv_slide=None, 
                                  h_ls=None, uv_ls=None, sedi=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar 2D :class:`Function`, :class:`Constant`, or an expression
        :kwarg salt: Initial condition for salinity field
        :type salt: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg temp: Initial condition for temperature field
        :type temp: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_2d: Initial condition for depth averaged velocity
        :type uv_2d: vector valued 2D :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_3d: Initial condition for horizontal velocity
        :type uv_3d: vector valued 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg tke: Initial condition for turbulent kinetic energy field
        :type tke: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg psi: Initial condition for turbulence generic lenght scale field
        :type psi: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.create_equations()
        if elev is not None:
            self.fields.elev_2d.project(elev)
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            if uv_3d is None:
                ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_3d,
                                   elem_height=self.fields.v_elem_size_3d).solve()
        # fluid slide
        if self.options.slide_is_viscous_fluid:
            if elev_ls is not None:
                self.fields.solution_ls.sub(1).project(elev_ls)
            if uv_ls is not None:
                self.fields.solution_ls.sub(0).project(uv_ls)
        # granular slide
        if self.options.flow_is_granular:
            if h_ls is not None:
                self.fields.h_ls.project(h_ls)
            if uv_ls is not None:
                self.fields.solution_ls.sub(1).project(self.fields.h_ls*uv_ls[0])
                self.fields.solution_ls.sub(2).project(self.fields.h_ls*uv_ls[1])

        if sedi is not None and self.options.solve_sediment:
            self.fields.c_3d.project(sedi)

        if uv_3d is not None:
            self.fields.uv_3d.project(uv_3d)
        if salt is not None and self.options.solve_salinity:
            self.fields.salt_3d.project(salt)
        if temp is not None and self.options.solve_temperature:
            self.fields.temp_3d.project(temp)
        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if tke is not None:
                self.fields.tke_3d.project(tke)
            if psi is not None:
                self.fields.psi_3d.project(psi)
            self.turbulence_model.initialize()

        if self.options.use_ale_moving_mesh:
            self.timestepper._update_3d_elevation()
            self.timestepper._update_moving_mesh()
        self.timestepper.initialize()
        # update all diagnostic variables
        self.timestepper._update_all_dependencies(self.simulation_time, 
                                                  do_2d_coupling=False,
                                                  do_vert_diffusion=False,
                                                  do_ale_update=True,
                                                  do_stab_params=True,
                                                  do_turbulence=False)
        if self.options.use_turbulence:
            self.turbulence_model.initialize()

    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg str eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export', index=self.i_export)
        # set uv to total uv instead of deviation from depth average
        # TODO find a cleaner way of doing this ...
        #self.fields.uv_3d += self.fields.uv_dav_3d
        for e in self.exporters.values():
            e.export()
        # restore uv_3d
        #self.fields.uv_3d -= self.fields.uv_dav_3d

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces :meth:`.assign_initial_conditions` in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format.  The required
        state variables are: elev_2d, uv_2d, uv_3d, salt_3d, temp_3d, tke_3d,
        psi_3d

        Currently hdf5 field import only works for the same number of MPI
        processes.

        :arg int i_export: export index to load
        :kwarg string outputdir: (optional) directory where files are read from.
            By default ``options.output_directory``.
        :kwarg float t: simulation time. Overrides the time stamp stored in the
            hdf5 files.
        :kwarg int iteration: Overrides the iteration count in the hdf5 files.
        """
        if not self._initialized:
            self.create_equations()
        if outputdir is None:
            outputdir = self.options.output_directory
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d', 'uv_3d',
                        'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        e.exporters['uv_3d'].load(i_export, self.fields.uv_3d)
        # NOTE remove mean from uv_3d
        self.timestepper._remove_depth_average_from_uv_3d()
        salt = temp = tke = psi = None
        if self.options.solve_salinity:
            salt = self.fields.salt_3d
            e.exporters['salt_3d'].load(i_export, salt)
        if self.options.solve_temperature:
            temp = self.fields.temp_3d
            e.exporters['temp_3d'].load(i_export, temp)
        if self.options.use_turbulence:
            if 'tke_3d' in self.fields:
                tke = self.fields.tke_3d
                e.exporters['tke_3d'].load(i_export, tke)
            if 'psi_3d' in self.fields:
                psi = self.fields.psi_3d
                e.exporters['psi_3d'].load(i_export, psi)
        self.assign_initial_conditions(elev=self.fields.elev_2d,
                                       uv_2d=self.fields.uv_2d,
                                       uv_3d=self.fields.uv_3d,
                                       salt=salt, temp=temp,
                                       tke=tke, psi=psi,
                                       )

        # time stepper bookkeeping for export time step
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.simulation_export_time
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.output_directory
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.simulation_export_time
        for e in self.exporters.values():
            e.set_next_export_ix(self.i_export + offset)

        self._simulation_continued = True

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        norm_h = norm(self.fields.elev_2d)
        norm_u = norm(self.fields.uv_3d)

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def solve_poisson_eq(self, q, uv_3d, w_3d, A=None, B=None, C=None, multi_layers=True):
        """
        Solve Poisson equation in two modes controlled by parameter `multi_layers'.

        Generic forms:
           2D: `div(grad(q)) + inner(A, grad(q)) + B*q = C`
           3d: `div(grad(q)) = C*(div(uv_3d) + Dx(w_3d, 2))`

        :arg A, B and C: Known functions, constants or expressions
        :type A: vector, B: scalar, C: scalar (3D: div terms). Valued :class:`Function`, `Constant`, or an expression
        :arg q: Non-hydrostatic pressure to be solved and output
        :type q: scalar function 3D or 2D :class:`Function`
        """
        q_test = TestFunction(q.function_space())
        normal = FacetNormal(q.function_space().mesh())
        boundary_markers = q.function_space().mesh().exterior_facets.unique_markers
        horizontal_is_dg = element_continuity(q.function_space().ufl_element()).horizontal in ['dg', 'hdiv']

        if not multi_layers:
            # weak forms
            f = (-dot(grad(q), grad(q_test)) + B*q*q_test)*dx - C*q_test*dx - q*div(A*q_test)*dx
            if horizontal_is_dg:
                f += inner(grad(avg(q)), jump(q_test, normal))*dS + avg(q)*jump(q_test, inner(A, normal))*dS
            # boundary conditions
            for bnd_marker in boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                #q_open_bc = self.q_bnd.assign(0.)
                if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                    # Neumann boundary condition => inner(grad(q), normal)=0.
                    f += (q*inner(A, normal))*q_test*ds_bnd

            prob = NonlinearVariationalProblem(f, q)
            solver = NonlinearVariationalSolver(prob,
                                            solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu', #'bjacobi', 'lu'
                                                               },
                                            options_prefix='poisson_solver')
            solver.solve()

            return q

    def set_sponge_damping(self, length, sponge_start_point, alpha=10., sponge_is_2d=True):
        """
        Set damping terms to reduce the reflection on solid boundaries.
        """
        pi = 4*np.arctan(1.)
        if length == [0., 0.]:
            return None
        if sponge_is_2d:
            damping_coeff = Function(self.function_spaces.P1_2d)
        else:
            damping_coeff = Function(self.function_spaces.P1)
        damp_vector = damping_coeff.dat.data[:]
        mesh = damping_coeff.ufl_domain()
        xvector = mesh.coordinates.dat.data[:, 0]
        yvector = mesh.coordinates.dat.data[:, 1]
        assert xvector.shape[0] == damp_vector.shape[0]
        assert yvector.shape[0] == damp_vector.shape[0]
        if xvector.max() <= sponge_start_point[0] + length[0]:
            length[0] = xvector.max() - sponge_start_point[0]
        if yvector.max() <= sponge_start_point[1] + length[1]:
            length[1] = yvector.max() - sponge_start_point[1]

        if length[0] > 0.:
            for i, x in enumerate(xvector):
                x = (x - sponge_start_point[0])/length[0]
                if x > 0 and x < 0.5:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
                elif x > 0.5 and x < 1.:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
                else:
                    damp_vector[i] = 0.
        if length[1] > 0.:
            for i, y in enumerate(yvector):
                x = (y - sponge_start_point[1])/length[1]
                if x > 0 and x < 0.5:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
                elif x > 0.5 and x < 1.:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
                else:
                    damp_vector[i] = 0.

        return damping_coeff

    def set_vertical_2d(self):
        """
        Set zero in y- direction, forming artificial vertical two dimensional.
        """
        if self.horizontal_domain_is_2d:
            uv_2d, elev_2d = self.fields.solution_2d.split()
            self.uv_2d_dg.project(uv_2d)
            self.uv_2d_dg.sub(1).assign(0.)
            uv_2d.project(self.uv_2d_dg)
            # landslide
            if self.options.slide_is_viscous_fluid:
                uv_ls, elev_ls = self.fields.solution_ls.split()
                self.uv_2d_dg.project(uv_ls)
                self.uv_2d_dg.sub(1).assign(0.)
                uv_ls.project(self.uv_2d_dg)

    def slide_shape(self, simulation_time):
        """
        Specific slide shape function for rigid landslide generated tsunami modelling.
        """
        L = 215.E3
        B = self.mesh2d.coordinates.sub(1).dat.data.max()
        S = 7.5E3
        hmax = 144.
        Umax = 35.
        T = self.options.t_landslide
        Ta = 0.5*T
        Tc = 0.
        Td = T - Ta - Tc
        R = 150.E3
        Ra = 75.E3
        Rc = Umax*Tc
        Rd = R - Ra - Rc
        phi = 0.
        x0 = 1112.5E3
        y0 = 0.5*B
        if simulation_time < Ta:
            s = Ra*(1. - cos(Umax/Ra*simulation_time))
        elif simulation_time >= Ta and simulation_time < (Ta + Tc):
            s = Ra + Umax*(simulation_time - Ta)
        elif simulation_time >= (Ta + Tc) and simulation_time < T:
            s = Ra + Rc + Rd*(sin(Umax/Rd*(simulation_time - Ta - Tc)))
        else:
            s = R
        xs = x0 + s*cos(phi)
        ys = y0 + s*sin(phi)
        # calculate slide shape below
        hs = Function(self.function_spaces.P1_2d)
        xy_vector = self.mesh2d.coordinates.dat.data
        hs_vector = hs.dat.data
        assert xy_vector.shape[0] == hs_vector.shape[0]
        for i, xy in enumerate(xy_vector):
            x = (xy[0] - xs)*cos(phi) + (xy[1] - ys)*sin(phi)
            y = -(xy[0] - xs)*sin(phi) + (xy[1] - ys)*cos(phi)
            if x < -(L+S) and x > -(L+2.*S):
                hs_vector[i] = hmax*exp(-(2.*(x+S+L)/S)**4 - (2.*y/B)**4)
            elif x < -S and x >= -(L+S):
                hs_vector[i] = hmax*exp(-(2.*y/B)**4)
            elif x < 0. and x >= -S:
                hs_vector[i] = hmax*exp(-(2.*(x+S)/S)**4 - (2.*y/B)**4)
            else:
                hs_vector[i] = 0.

            is_block = True # i.e. block slide
            if is_block:
                if x < -(L+S) and x > -(L+2.*S):
                   hs_vector[i] = hmax*exp(-(2.*(x+S+L)/S)**4)
                elif x < -S and x >= -(L+S):
                   hs_vector[i] = hmax
                elif x < 0. and x >= -S:
                   hs_vector[i] = hmax*exp(-(2.*(x+S)/S)**4)
                else:
                   hs_vector[i] = 0.

        return hs

    def iterate(self, update_forcings=None, update_forcings3d=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.

        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            of the 2D system (if any).
        :kwarg update_forcings_3d: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            of the 3D equations (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        if not self._initialized:
            self.create_equations()

        self.options.check_salinity_conservation &= self.options.solve_salinity
        self.options.check_salinity_overshoot &= self.options.solve_salinity
        self.options.check_temperature_conservation &= self.options.solve_temperature
        self.options.check_temperature_overshoot &= self.options.solve_temperature
        self.options.check_volume_conservation_3d &= self.options.use_ale_moving_mesh
        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0
        self.options.use_limiter_for_velocity &= self.options.polynomial_degree > 0
        self.options.use_limiter_for_velocity &= self.options.element_family == 'dg-dg'

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_volume_conservation_3d:
            c = callback.VolumeConservation3DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salinity_conservation:
            c = callback.TracerMassConservationCallback('salt_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salinity_overshoot:
            c = callback.TracerOvershootCallBack('salt_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temperature_conservation:
            c = callback.TracerMassConservationCallback('temp_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temperature_overshoot:
            c = callback.TracerOvershootCallBack('temp_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self._simulation_continued:
            # set all callbacks to append mode
            for m in self.callbacks:
                for k in self.callbacks[m]:
                    self.callbacks[m][k].set_write_mode('append')

        # split solution to facilitate the following
        uv_2d_old, elev_2d_old = self.solution_2d_old.split()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        uta_old, eta_old = split(self.solution_2d_old) # note: not '.split()'
        uta, eta = split(self.fields.solution_2d)

        # viscous fluid landslide
        if self.options.slide_is_viscous_fluid:
            uv_ls_old, elev_ls_old = self.solution_ls_old.split()
            uv_ls, elev_ls = self.fields.solution_ls.split()
            uta_ls_old, eta_ls_old = split(self.solution_ls_old) # note: not '.split()'
            uta_ls, eta_ls = split(self.fields.solution_ls)

        # trial and test functions used to update
        uv_tri = TrialFunction(self.function_spaces.U_2d)
        uv_test = TestFunction(self.function_spaces.U_2d)
        w_tri = TrialFunction(self.function_spaces.H_2d)
        w_test = TestFunction(self.function_spaces.H_2d)
        uta_test, eta_test = TestFunctions(self.fields.solution_2d.function_space())

        # for 3d velocities
        tri_uv_3d = TrialFunction(self.function_spaces.U)
        test_uv_3d = TestFunction(self.function_spaces.U)
        tri_w_3d = TrialFunction(self.function_spaces.W)
        test_w_3d = TestFunction(self.function_spaces.W)
        tri_h_3d = TrialFunction(self.function_spaces.H)
        test_h_3d = TestFunction(self.function_spaces.H)

        # update sigma mesh
        h_3d = self.fields.elev_3d + self.bathymetry_3d_dg
        alpha = self.options.depth_wd_interface
        if self.options.use_wetting_and_drying:
            h_total = 2 * alpha**2 / (2 * alpha + abs(h_3d)) + 0.5 * (abs(h_3d) + h_3d)
        else:
            h_total = h_3d
        z_in_sigma = self.sigma_coord*h_total + ( - 1.)*self.bathymetry_3d_dg
        self.fields.z_coord_3d.project(z_in_sigma)
        self.mesh.coordinates.dat.data[:, 2] = self.fields.z_coord_3d.dat.data[:]
        self.fields.z_coord_ref_3d.assign(self.fields.z_coord_3d)
        self.mesh_updater.update_elem_height()
        self.mesh.clear_spatial_index()

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        if self.options.set_vertical_2d: # True for 1D case
            self.set_vertical_2d()

        if True:
            # ----- Self-defined time integrator for layer-integrated NH solver
            fields_2d = {
                    'linear_drag_coefficient': self.options.linear_drag_coefficient,
                    'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
                    'manning_drag_coefficient': self.options.manning_drag_coefficient,
                    'viscosity_h': self.options.horizontal_viscosity,
                    'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                    'coriolis': self.options.coriolis_frequency,
                    'wind_stress': self.options.wind_stress,
                    'atmospheric_pressure': self.options.atmospheric_pressure,
                    'momentum_source': self.options.momentum_source_2d,
                    'volume_source': self.options.volume_source_2d,
                    'w_nh': self.fields.w_nh,
                    'uv_nh': self.fields.uv_nh,
                    'eta': self.fields.elev_2d,
                    'uv': self.fields.uv_2d,
                    'bathymetry_init': self.fields.bathymetry_2d,
                    'slide_viscosity': self.options.slide_viscosity,
                    'slide_source': self.fields.slide_source_2d,
                    'ext_pressure': self.fields.q_2d + self.options.rho_fluid*g_grav*(self.bathymetry_dg + self.fields.elev_2d),
                    'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_start, alpha=10., sponge_is_2d=True),}

            solver_parameters = {'snes_type': 'newtonls', # ksponly, newtonls
                                 'ksp_type': 'gmres', # gmres, preonly
                                 'pc_type': 'fieldsplit'}

            fields_with_nh_terms = {'nonhydrostatic_pressure': self.fields.q_2d}
            fields_layer_integrated = {'uv_delta_for_depth_integrated': self.fields.uv_delta}
            fields_layer_difference = {'uv_2d_for_layer_difference': self.fields.uv_2d}

            # timestepper for operator splitting in 3D NH solver
            timestepper_operator_splitting = timeintegrator.CrankNicolson(self.eq_operator_split, self.fields.solution_2d,
                                                              fields_2d, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)
            timestepper_operator_splitting_explicit = timeintegrator.CrankNicolson(self.eq_operator_split, self.fields.solution_2d,
                                                              fields_2d, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)
            timestepper_operator_splitting_implicit = timeintegrator.CrankNicolson(self.eq_operator_split, self.fields.solution_2d,
                                                              fields_2d, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=1.0)
            # timestepper for depth-integrated NH solver
            timestepper_depth_integrated = timeintegrator.CrankNicolson(self.eq_sw_nh, self.fields.solution_2d,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)
            # timestepper for two-layer NH solvers
            fields_layer_integrated.update(fields_2d)
            timestepper_layer_integrated = timeintegrator.CrankNicolson(self.eq_sw_nh, self.fields.solution_2d,
                                                              fields_layer_integrated, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe,
                                                              semi_implicit=False,
                                                              theta=0.5)
            fields_layer_difference.update(fields_2d)
            timestepper_layer_difference = timeintegrator.CrankNicolson(self.eq_sw_mom, self.fields.uv_delta,
                                                              fields_layer_difference, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit,
                                                              semi_implicit=False,
                                                              theta=0.5)
            # timestepper for free surface equation
            timestepper_free_surface = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_mid,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)
            timestepper_free_surface_explicit = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_mid,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)
            timestepper_free_surface_implicit = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_mid,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=1.0)
            # timestepper for only elevation gradient term
            timestepper_mom_2d = timeintegrator.CrankNicolson(self.eq_mom_2d, self.uv_2d_mid,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)

        if self.options.flow_is_granular:
            fields_ls = {
                'phi_i': self.phi_i,
                'phi_b': self.phi_b,
                #'kap': self.kap,
                'uv_div': self.uv_div_ls,
                'strain_rate': self.strain_rate_ls,
                'fluid_pressure_gradient': self.grad_p_ls,
                'h_2d': self.h_2d_ls,
                }

            dt_ls = self.dt / self.options.n_dt
            # solver for granular landslide motion
            a_ls = self.eq_ls.mass_term(self.eq_ls.trial)
            l_ls = (self.eq_ls.mass_term(self.fields.solution_ls) + Constant(dt_ls)*
                    self.eq_ls.residual('all', self.fields.solution_ls, self.fields.solution_ls,
                                        fields_ls, fields_ls, self.bnd_functions['landslide_motion'])
                   )
            prob_ls = LinearVariationalProblem(a_ls, l_ls, self.solution_ls_tmp)
            solver_ls = LinearVariationalSolver(prob_ls, solver_parameters=self.options.timestepper_options.solver_parameters_granular_explicit)
            # solver for div(velocity)
            h_ls, hu_ls, hv_ls = self.fields.solution_ls.split()
            u_ls = conditional(h_ls <= 0, zero(hu_ls.ufl_shape), hu_ls/h_ls)
            v_ls = conditional(h_ls <= 0, zero(hv_ls.ufl_shape), hv_ls/h_ls)
            tri_div = TrialFunction(self.uv_div_ls.function_space())
            test_div = TestFunction(self.uv_div_ls.function_space())
            a_div = tri_div*test_div*dx
            l_div = (Dx(u_ls, 0) + Dx(v_ls, 1))*test_div*dx
            prob_div = LinearVariationalProblem(a_div, l_div, self.uv_div_ls)
            solver_div = LinearVariationalSolver(prob_div)
            # solver for strain rate
            l_sr = 0.5*(Dx(u_ls, 1) + Dx(v_ls, 0))*test_div*dx
            prob_sr = LinearVariationalProblem(a_div, l_sr, self.strain_rate_ls)
            solver_sr = LinearVariationalSolver(prob_sr)
            # solver for fluid pressure at slide surface
            h_2d = self.bathymetry_dg + self.fields.elev_2d
            tri_pf = TrialFunction(self.grad_p.function_space())
            test_pf = TestFunction(self.grad_p.function_space())
            a_pf = dot(tri_pf, test_pf)*dx
            l_pf = dot(conditional(h_2d <= 0, zero(self.grad_p.ufl_shape), 
                       grad(self.options.rho_fluid*physical_constants['g_grav']*h_2d + self.fields.q_2d)), test_pf)*dx
            prob_pf = LinearVariationalProblem(a_pf, l_pf, self.grad_p)
            solver_pf = LinearVariationalSolver(prob_pf)

        if self.options.slide_is_viscous_fluid:
            theta = 0.5
            solution_if_semi = self.fields.solution_ls
            F_fluid_ls = (self.eq_ls.mass_term(self.fields.solution_ls) - self.eq_ls.mass_term(self.solution_ls_old) - self.dt*(
                            theta*self.eq_ls.residual('all', self.fields.solution_ls, solution_if_semi, fields, fields, self.bnd_functions['landslide_motion']) + 
                            (1-theta)*self.eq_ls.residual('all', self.solution_ls_old, self.solution_ls_old, fields, fields, self.bnd_functions['landslide_motion'])) - 
                            self.dt*self.eq_ls.add_external_surface_term(uv_2d_old, elev_2d_old, uv_2d_old, elev_ls_old, fields, self.bnd_functions['shallow_water']) 
                           )
            prob_fluid_ls = NonlinearVariationalProblem(F_fluid_ls, self.fields.solution_ls)
            solver_fluid_ls = NonlinearVariationalSolver(prob_fluid_ls,
                                                           solver_parameters=solver_parameters)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            if self.options.timestepper_type == 'SSPRK22':
                n_stages = 2
                coeff = [[0., 1.], [1./2., 1./2.]]

            self.uv_3d_old.assign(self.fields.uv_3d)
            self.w_3d_old.assign(self.fields.w_3d)
            self.uv_2d_old.assign(self.fields.uv_2d)
            self.elev_3d_old.assign(self.fields.elev_3d)
            self.elev_2d_old.assign(self.fields.elev_2d)
            self.elev_2d_mid.assign(self.fields.elev_2d)
            self.q_2d_old.assign(self.fields.q_2d)
            self.solution_2d_old.assign(self.fields.solution_2d)
            self.bathymetry_dg_old.assign(self.bathymetry_dg)
            self.z_coord_3d_old.assign(self.fields.z_coord_3d)

            if self.options.slide_is_rigid:
                self.h_ls_old.assign(self.fields.h_ls)
            if self.options.flow_is_granular:
                self.solution_ls_old.assign(self.fields.solution_ls)
                self.solution_ls_mid.assign(self.fields.solution_ls)

            h_2d_array = self.fields.elev_2d.dat.data + self.bathymetry_dg.dat.data
            h_3d_array = self.fields.elev_3d.dat.data + self.bathymetry_3d_dg.dat.data

            couple_granular_and_wave_in_ssprk = False
            if (not couple_granular_and_wave_in_ssprk):
                if self.options.flow_is_granular:
                    if not self.options.lamda == 0.:
                        self.h_2d_cg.project(self.bathymetry_dg + self.fields.elev_2d)
                        self.h_2d_ls.dat.data[:] = self.h_2d_cg.dat.data[:]
                    for i in range(self.options.n_dt):
                        # solve fluid pressure on slide
                       # self.extract_bot_q.solve()
                        self.bathymetry_dg.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                       # solver_pf.solve()
                       # self.grad_p_ls.dat.data[:] = self.grad_p.dat.data[:]

                        self.solution_ls_mid.assign(self.fields.solution_ls)
                        for i_stage in range(n_stages):
                            solver_ls.solve()
                            self.fields.solution_ls.assign(coeff[i_stage][0]*self.solution_ls_mid + coeff[i_stage][1]*self.solution_ls_tmp)
                            if self.options.use_wetting_and_drying:
                                limiter_start_time = 0.
                                limiter_end_time = self.options.simulation_end_time - t_epsilon
                                use_limiter = self.options.use_limiter_for_granular and self.simulation_time >= limiter_start_time and self.simulation_time <= limiter_end_time
                                self.wd_modification_ls.apply(self.fields.solution_ls, self.options.wetting_and_drying_threshold, use_limiter)
                            solver_div.solve()

                if self.landslide:
                    # update landslide motion source
                    if update_forcings is not None:
                        update_forcings(self.simulation_time + self.dt)

                    ind_wet_2d = np.where(h_2d_array[:] > 0)[0]
                    if self.simulation_time >= 0.:
                        self.fields.slide_source_2d.assign(0.)
                        self.fields.slide_source_2d.dat.data[ind_wet_2d] = (self.fields.h_ls.dat.data[ind_wet_2d] 
                                                                            - self.h_ls_old.dat.data[ind_wet_2d])/self.dt/self.slope.dat.data.min()
                    # copy slide source to 3d
                    self.copy_slide_source_to_3d.solve()
                    self.bathymetry_dg.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()

                # update 3d bathymetry
                self.copy_bath_to_3d.solve()



            if self.options.thin_film: 
                # this thin-film wetting-drying scheme relies on the high resolution at wetting-drying front
                # has been benchmarked by Thacker test, but needs further work by orther tests, WPan
                H = self.bathymetry_dg.dat.data + elev_2d.dat.data
                visu_space = exporter.get_visu_space(uv_2d.function_space())
                tmp_proj_func = visu_space.get_work_function()
                tmp_proj_func.project(uv_2d)
                visu_space.restore_work_function(tmp_proj_func)
                UV = tmp_proj_func.dat.data
                ind = np.where(H[:] <= self.options.wetting_and_drying_threshold)[0]
                H[ind] = self.options.wetting_and_drying_threshold
                elev_2d.dat.data[ind] = (H - self.bathymetry_dg.dat.data)[ind] # note not appllies to landslide

            else:
                H_min = (self.bathymetry_dg.dat.data + self.fields.elev_2d.dat.data).min()
                if self.options.slide_is_viscous_fluid and self.simulation_time <= self.options.t_landslide:
                    H_min = (self.bathymetry_ls.dat.data + self.fields.elev_ls.dat.data).min()
              #  H = self.bathymetry_dg + self.fields.elev_2d
              #  self.bathymetry_wd.project(self.bathymetry_dg + 2*self.options.depth_wd_interface**2/(2*self.options.depth_wd_interface + abs(H)) + 0.5 * (abs(H) - H))
              #  ExpandFunctionTo3d(self.bathymetry_wd, self.fields.bathymetry_3d).solve()

            # ----- Construct depth-integrated landslide solver
            if self.options.slide_is_viscous_fluid:
                if update_forcings is not None:
                    update_forcings(self.simulation_time + self.dt)
                if self.simulation_time <= self.options.t_landslide:
                    if self.options.slide_is_viscous_fluid:
                        solver_fluid_ls.solve()
                        elev_fluid_ls = elev_ls + self.eq_ls.water_height_displacement(elev_ls)
                        self.bathymetry_dg.project(-elev_fluid_ls)
                else:
                    print_output('Landslide motion has been stopped, and waves continue propagating!')

            hydrostatic_solver_2d = False
            hydrostatic_solver_3d = False
            conventional_3d_NH_solver = True
            one_layer_NH_solver = False
            reduced_two_layer_NH_solver = False
            coupled_two_layer_NH_solver = False
            coupled_three_layer_NH_solver = False
            arbitrary_multi_layer_NH_solver = False
            # use n_layers to control
            if self.n_layers == 1:
                one_layer_NH_solver = False
                arbitrary_multi_layer_NH_solver = True
            elif self.n_layers == 2:
                reduced_two_layer_NH_solver = False
                coupled_two_layer_NH_solver = False
                arbitrary_multi_layer_NH_solver = True
            else:
                coupled_three_layer_NH_solver = False
                arbitrary_multi_layer_NH_solver = True
                arbitrary_multi_layer_NH_solver_variant_form = True

            # --- Hydrostatic solver ---
            if hydrostatic_solver_2d:
                if self.options.slide_is_viscous_fluid:
                    if self.simulation_time <= t_epsilon:
                        timestepper_depth_integrated.F += -self.dt*self.eq_sw_nh.add_landslide_term(uv_ls, elev_ls, fields, self.bathymetry_ls, self.bnd_functions['landslide_motion'])
                        timestepper_depth_integrated.update_solver()
                    if self.simulation_time == self.options.t_landslide:
                        timestepper_depth_integrated.F += self.dt*self.eq_sw_nh.add_landslide_term(uv_ls, elev_ls, fields, self.bathymetry_ls, self.bnd_functions['landslide_motion'])
                        timestepper_depth_integrated.update_solver()

                timestepper_depth_integrated.advance(self.simulation_time, update_forcings)

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()

            elif hydrostatic_solver_3d:
                #self.timestepper.advance(self.simulation_time,
                #                         update_forcings, update_forcings3d)
                self.bathymetry_cg_2d.project(self.bathymetry_dg)
                ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d).solve()
                n_stages = 2
                if True:
                    for i_stage in range(n_stages):
                        ## 2D advance
                        if i_stage == 1 and self.options.update_free_surface and self.options.solve_separate_elevation_gradient:
                            self.timestepper.store_elevation(i_stage - 1)
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.copy_uv_dav_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                            timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                            #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                            # compute mesh velocity
                            self.timestepper.compute_mesh_velocity(i_stage - 1)

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        if self.options.update_free_surface:
                            self.copy_elev_to_3d.solve()
                            if self.options.use_ale_moving_mesh:
                                self.mesh_updater.update_mesh_coordinates()

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.c_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        last_stage = i_stage == n_stages - 1

                        if last_stage:
                            ## compute final prognostic variables
                            # correct uv_3d
                            if self.options.update_free_surface and self.options.solve_separate_elevation_gradient:
                                self.copy_uv_to_uv_dav_3d.solve()
                                self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                if self.options.solve_sediment:
                                    with timed_stage('impl_sediment_vdiff'):
                                        self.timestepper.timesteppers.sediment_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            ## compute final diagnostic fields
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = 0. # TODO
                            self.w_solver.solve()
                            # update parametrizations
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = self.fields.w_3d.dat.data[:, self.vert_ind] # TODO
                        else:
                            ## update variables that explict solvers depend on
                            # correct uv_3d
                          #  self.copy_uv_to_uv_dav_3d.solve()
                          #  self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = 0. # TODO
                            self.w_solver.solve()
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = self.fields.w_3d.dat.data[:, self.vert_ind] # TODO

            # --- Non-hydrostatic solver ---
            # based on operator-splitting method used in Telemac3D
            # Jankowski, J.A., 1999. A non-hydrostatic model for free surface flows.
            # Two-stage second-order nonlinear Strong Stability-Preserving (SSP) Runge-Kutta scheme
            # Gottlieb et al., 2001. doi: https://doi.org/10.1137/S003614450036757X
            elif conventional_3d_NH_solver:
                if self.horizontal_domain_is_2d:
                    vert_ind = 2
                else:
                    vert_ind = 1

              #  if self.landslide:
               #     if update_forcings is not None:
                #        update_forcings(self.simulation_time)
                 #   self.bathymetry_cg_2d.project(self.bathymetry_dg)
                  #  if self.simulation_time <= t_epsilon:
                   #     bath_2d_to_3d = ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d)
                    #    slide_source_2d_to_3d = ExpandFunctionTo3d(self.fields.slide_source_2d, self.fields.slide_source_3d)
                    #bath_2d_to_3d.solve()

                h_3d = self.fields.elev_3d + self.bathymetry_3d_dg

                if self.simulation_time <= t_epsilon:
                    assert not self.options.use_pressure_correction, \
                        'Pressure correction method is temporarily implemented in only sigma model.'
                    # solver for the Poisson equation
                    q_3d = self.fields.q_3d
                    fs_q = q_3d.function_space()
                    q_is_dg = element_continuity(fs_q.ufl_element()).horizontal == 'dg'
                    uv_3d = self.fields.uv_3d
                    w_3d = self.fields.w_3d
                    Const = physical_constants['rho0']/self.dt
                   # if self.fields.density_3d is not None:
                   #     Const += self.fields.density_3d/self.dt
                    trial_q = TrialFunction(fs_q)
                    test_q = TestFunction(fs_q)

                    # nabla^2-term is integrated by parts
                    a_q = dot(grad(test_q), grad(trial_q)) * dx #+ test_q*inner(grad(q), normal)*ds_surf
                    l_q = Const * dot(grad(test_q), uv_3d) * dx
                    if self.landslide:
                        l_q += -Const*self.fields.slide_source_3d*self.normal[vert_ind]*test_q*ds_bottom

                    if q_is_dg:
                        degree_h, degree_v = fs_q.ufl_element().degree()
                        if self.horizontal_domain_is_2d:
                            elemsize = (self.fields.h_elem_size_3d*(self.normal[0]**2 + self.normal[1]**2) +
                                        self.fields.v_elem_size_3d*self.normal[2]**2)
                        else:
                            elemsize = (self.fields.h_elem_size_3d*self.normal[0]**2 +
                                        self.fields.v_elem_size_3d*self.normal[1]**2)
                        sigma = 5.0*degree_h*(degree_h + 1)/elemsize
                        if degree_h == 0:
                            sigma = 1.5/elemsize
                        alpha_q = avg(sigma)
                        # Nitsche proved that if gamma_q is taken as η/h, where
                        # h is the element size and η is a sufficiently large constant, then the discrete solution
                        # converges to the exact solution with optimal order in H1 and L2.
                        gamma_q = 2*sigma*Const # 1E10

                        a_q += - dot(avg(grad(test_q)), jump(trial_q, self.normal))*(dS_v + dS_h) \
                               - dot(jump(test_q, self.normal), avg(grad(trial_q)))*(dS_v + dS_h) \
                               + alpha_q*dot(jump(test_q, self.normal), jump(trial_q, self.normal))*(dS_v + dS_h)

                        incompressibility_flux_type = 'central'
                        if incompressibility_flux_type == 'central':
                            u_flux = avg(uv_3d)
                        elif incompressibility_flux_type == 'upwind':
                            switch = conditional(
                                gt(abs(dot(uv_3d, self.normal))('+'), 0.0), 1.0, 0.0
                            )
                            u_flux = switch * uv_3d('+') + (1 - switch) * uv_3d('-')

                        l_q += -Const * dot(u_flux, self.normal('+')) * jump(test_q) * (dS_v + dS_h)
                       # l_q += -Const * dot(uv_3d, self.normal) * test_q * ds_surf
                       # l_q = -Const * div(uv_3d) * test_q * dx
                        # zero Dirichlet top boundary
                        q0 = Constant(0.)
                        a_q += - dot(grad(test_q), trial_q*self.normal)*ds_surf \
                               - dot(test_q*self.normal, grad(trial_q))*ds_surf \
                               + gamma_q*test_q*trial_q*ds_surf
                        l_q += -q0*dot(grad(test_q), self.normal)*ds_surf + gamma_q*q0*test_q*ds_surf

                    # boundary conditions: to refer to the top and bottom use "top" and "bottom"
                    # for other boundaries use the normal numbers (ids) from the horizontal mesh
                    # (UnitSquareMesh automatically defines 1,2,3, and 4)
                    bc_top = DirichletBC(fs_q, 0., "top")
                    bcs = [bc_top]
                    if not self.options.update_free_surface:
                        bcs = []
                   # bcs.append(DirichletBC(fs_q, 0., 1)) # TODO delete after testing turbidity current case
                    for bnd_marker in self.boundary_markers:
                        func = self.bnd_functions['shallow_water'].get(bnd_marker)
                        ds_bnd = ds_v(int(bnd_marker))
                        if func is not None: #TODO set more general and accurate conditional statement
                            bc = DirichletBC(fs_q, 0., int(bnd_marker))
                            bcs.append(bc)
                            if q_is_dg:
                                a_q += - dot(grad(test_q), trial_q*self.normal)*ds_bnd \
                                       - dot(test_q*self.normal, grad(trial_q))*ds_bnd \
                                       + gamma_q*test_q*trial_q*ds_bnd
                                l_q += -q0*dot(grad(test_q), self.normal)*ds_bnd + gamma_q*q0*test_q*ds_bnd
                            l_q += -Const * dot(uv_3d, self.normal) * test_q * ds_bnd
                        else:
                            pass#l_q += -Const * dot(uv_3d, self.normal) * test_q * ds_bnd

                    # you can add Dirichlet bcs to other boundaries if needed
                    # any boundary that is not specified gets the natural zero Neumann bc
                    prob_q = LinearVariationalProblem(a_q, l_q, q_3d, bcs=bcs)
                    if q_is_dg:
                        prob_q = LinearVariationalProblem(a_q, l_q, q_3d)
                    solver_q = LinearVariationalSolver(prob_q,
                                                    solver_parameters={'snes_type': 'ksponly',#'newtonls''ksponly', final: 'ksponly'
                                                                       'ksp_type': 'gmres',#'gmres''preonly',              'gmres'
                                                                       'pc_type': 'gamg'},#'ilu''gamg',                     'ilu'
                                                    options_prefix='poisson_solver')

                    # solver for updating uv_3d
                    a_u = dot(tri_uv_3d, test_uv_3d)*dx
                    l_u = dot(uv_3d - self.dt/physical_constants['rho0']*grad(q_3d), test_uv_3d)*dx
                    prob_u = LinearVariationalProblem(a_u, l_u, uv_3d)
                    solver_u = LinearVariationalSolver(prob_u)

                n_stages = 2
                if self.options.use_operator_splitting:
                    for i_stage in range(n_stages):
                        ## 2D advance
                        if i_stage == 1 and self.options.update_free_surface and self.options.solve_separate_elevation_gradient:
                           # self.timestepper.store_elevation(i_stage - 1)
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.copy_uv_dav_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                            timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                            if self.limiter_h is not None:
                                self.limiter_h.apply(self.fields.elev_2d)
                            if self.limiter_u is not None:
                                self.limiter_u.apply(self.fields.uv_2d)
                            #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                            # compute mesh velocity
                           # self.timestepper.compute_mesh_velocity(i_stage - 1)
                            self.copy_elev_to_3d.solve()

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.c_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        # wetting and drying treatment
                        if self.options.use_wetting_and_drying:
                            ind_dry_3d = np.where(h_3d_array[:] <= 0)[0]
                            self.fields.uv_3d.dat.data[ind_dry_3d] = [0, 0, 0]

                        last_stage = i_stage == n_stages - 1

                        if last_stage:
                            ## compute final prognostic variables
                            # correct uv_3d
                            if self.options.update_free_surface and self.options.solve_separate_elevation_gradient:
                                self.copy_uv_to_uv_dav_3d.solve()
                                self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                if self.options.solve_sediment:
                                    with timed_stage('impl_sediment_vdiff'):
                                        self.timestepper.timesteppers.sediment_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            ## compute final diagnostic fields
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update parametrizations
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()
                        else:
                            ## update variables that explict solvers depend on
                            # correct uv_3d
                          #  self.copy_uv_to_uv_dav_3d.solve()
                          #  self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()

                        if last_stage:
                           # if self.landslide:
                           #     slide_source_2d_to_3d.solve()
                            # solve q_3d
                            solver_q.solve()
                           # solve(a_q==l_q, q_3d)
                            # update uv_3d
                            solver_u.solve()

                            # wetting and drying treatment
                            if self.options.use_wetting_and_drying:
                                ind_dry_3d = np.where(h_3d_array[:] <= 0)[0]
                                self.fields.uv_3d.dat.data[ind_dry_3d] = [0, 0, 0]

                            # update water level elev_2d
                            if self.options.update_free_surface:
                                # update final depth-averaged uv_2d
                                self.uv_averager.solve()
                                self.extract_surf_dav_uv.solve()
                                self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                                self.elev_2d_mid.assign(self.elev_2d_old)
                                timestepper_free_surface.advance(self.simulation_time, update_forcings)
                                self.fields.elev_2d.assign(self.elev_2d_mid)
                                ## update mesh
                                self.copy_elev_to_3d.solve()
                               # if self.options.use_ale_moving_mesh:
                               #     self.mesh_updater.update_mesh_coordinates()
                                if self.limiter_h is not None:
                                    self.limiter_h.apply(self.fields.elev_2d)
                                if self.limiter_u is not None:
                                    self.limiter_u.apply(self.fields.uv_2d)
                            ## update mesh
                            self.fields.z_coord_3d.project(z_in_sigma)
                            self.mesh.coordinates.dat.data[:, 2] = self.fields.z_coord_3d.dat.data[:]
                            self.mesh_updater.update_elem_height()
                            self.mesh.clear_spatial_index()
                           # self.fields.w_mesh_3d.project((self.fields.z_coord_3d - self.z_coord_3d_old)/self.dt)

                else: # ssprk in NHWAVE TODO change something about ALE due to nh pressure updating free surface
                    for i_stage in range(n_stages):
                        advancing_elev_implicitly = False
                        # 2d advance
                        if self.options.solve_separate_elevation_gradient and self.options.update_free_surface:
                           # self.uv_averager.solve()
                           # self.extract_surf_dav_uv.solve()
                            self.copy_uv_dav_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                           # self.timestepper.store_elevation(i_stage)
                            if i_stage == 0 or (not advancing_elev_implicitly):
                               # self.timestepper.store_elevation(i_stage)
                                self.uv_2d_mid.assign(self.fields.uv_dav_2d)
                                timestepper_mom_2d.advance(self.simulation_time, update_forcings)
                                self.fields.uv_2d.assign(self.uv_2d_mid)
                               # timestepper_operator_splitting_explicit.advance(self.simulation_time, update_forcings)
                               # self.timestepper.compute_mesh_velocity(i_stage)
                            else:
                               # self.uv_2d_mid.assign(self.fields.uv_dav_2d)
                               # timestepper_mom_2d.advance(self.simulation_time, update_forcings)
                               # self.fields.uv_2d.assign(self.uv_2d_mid)
                               # self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                                self.fields.elev_2d.assign(self.elev_2d_old)
                                timestepper_operator_splitting_explicit.advance(self.simulation_time, update_forcings)
                           # self.timestepper.compute_mesh_velocity(i_stage)

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage_nh(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        if self.options.update_free_surface and i_stage == 1:
                            self.copy_elev_to_3d.solve()
                            if self.options.use_ale_moving_mesh:
                                self.mesh_updater.update_mesh_coordinates()

                        ## 3D advance in old mesh
         #               solver_mom_hori_ssprk.solve()
          #              self.fields.uv_3d.assign(self.uv_3d_mid)
           #             if self.options.use_limiter_for_velocity:
            #                self.uv_limiter.apply(self.fields.uv_3d)

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.c_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage_nh(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)
                            self.uv_limiter.apply(self.fields.w_3d)

                        # couple 2d (elevation gradient) into 3d
                        if self.options.solve_separate_elevation_gradient and self.options.update_free_surface:
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.assign(self.fields.uv_3d + (self.fields.uv_dav_3d - self.uv_dav_3d_mid))

                        if self.landslide:
                            slide_source_2d_to_3d.solve()
                        # solve q_3d
                        solver_q.solve()
                       # solve(a_q==l_q, q_3d)
                        # update uv_3d
                        solver_u.solve()

                       # l_pg_u = -self.dt/physical_constants['rho0']*(Dx(q_3d, 0)*test_uv_3d[0] + Dx(q_3d, 1)*test_uv_3d[1])*dx
                       # l_pg_w = -self.dt/physical_constants['rho0']*dot(Dx(q_3d, 2), test_uv_3d[2])*dx
                        self.timestepper.timesteppers.mom_expl.solve_pg_nh(i_stage)

                        if i_stage == 1:
                           # self.fields.uv_3d.assign(0.5*(self.uv_3d_old + self.fields.uv_3d))
                           # self.fields.w_3d.assign(0.5*(self.w_3d_old + self.fields.w_3d))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                if self.options.solve_sediment:
                                    with timed_stage('impl_sediment_vdiff'):
                                        self.timestepper.timesteppers.sediment_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()

                        # update free surface elevation
                        if self.options.update_free_surface:
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.elev_2d_mid.assign(self.elev_2d_old)

                            self.timestepper.store_elevation(i_stage)
                            if i_stage == 0:
                                timestepper_free_surface_implicit.advance(self.simulation_time, update_forcings)
                            else:
                                timestepper_free_surface_implicit.advance(self.simulation_time, update_forcings)
                            self.fields.elev_2d.assign(self.elev_2d_mid)
                        #    self.timestepper.compute_mesh_velocity(i_stage)
                            self.copy_elev_to_3d.solve()
                            if i_stage == 1:
                                # compute mesh velocity
                               # self.timestepper.compute_mesh_velocity(0)
                               # self.fields.w_mesh_3d.assign(0.)
                                ## update mesh
                                if self.options.use_ale_moving_mesh:
                                    self.mesh_updater.update_mesh_coordinates()
                            else:
                                self.timestepper.compute_mesh_velocity(0)

            elif one_layer_NH_solver:
                pressure_projection = True
                theta = 0.5
                theta_nh = 1.0
                par = 0.5 # approximation parameter for NH terms
                d = self.bathymetry_dg
                h_old = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(self.elev_2d_old)
                h_mid = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(self.elev_2d_mid)
                if self.simulation_time <= t_epsilon:
                    uta_old, eta_old = split(self.solution_2d_old) # note: not '.split()'
                    uta, eta = split(self.fields.solution_2d)
                    fields_with_nh_terms.update(fields)
                    timestepper_depth_integrated.F += (-self.dt*(theta_nh*self.eq_sw_nh.add_nonhydrostatic_term(uta, eta, fields_with_nh_terms, self.bnd_functions['shallow_water']) +
                                                       (1 - theta_nh)*self.eq_sw_nh.add_nonhydrostatic_term(uta_old, eta_old, fields_with_nh_terms, self.bnd_functions['shallow_water']))
                                                      )
                    prob_nh = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                    solver_nh = NonlinearVariationalSolver(prob_nh,
                                                           solver_parameters=solver_parameters)
                if pressure_projection:
                    timestepper_depth_integrated.advance(self.simulation_time, update_forcings)
                else:
                    timestepper_depth_integrated.solution_old.assign(self.fields.solution_2d)
                    if update_forcings is not None:
                        update_forcings(self.simulation_time + self.dt)
                    solver_nh.solve()
                self.uv_2d_mid.assign(self.fields.uv_2d)
                self.elev_2d_mid.assign(self.fields.elev_2d)
                # formulate coefficients
                if pressure_projection:
                    A = theta*grad(self.elev_2d_mid - d)/h_mid + (1. - theta)*grad(self.elev_2d_old - d)/h_old
                    B = div(A) - 2./(par*h_mid*h_mid)
                    C = (div(self.uv_2d_mid) + (self.w_surface + inner(2.*self.uv_2d_mid - self.uv_2d_old, grad(d)))/h_mid)/(par*self.dt)
                    if self.landslide:
                        C = (div(self.uv_2d_mid) + (self.w_surface + inner(2.*self.uv_2d_mid - self.uv_2d_old, grad(d)) - self.fields.slide_source_2d)/h_mid)/(par*self.dt)
                else:
                    A = theta*grad(self.elev_2d_mid - d)/h_mid + (1. - theta)*grad(self.elev_2d_old - d)/h_old
                    B = div(A) - 2./(par*h_mid*h_mid)
                    C = (div(self.uv_2d_mid) + (self.w_surface + 2.*self.dt*self.q_2d_old/h_mid +
                         inner(2.*self.uv_2d_mid - self.uv_2d_old, grad(d)))/h_mid)/(par*self.dt)
                    if self.landslide:
                        C = (div(self.uv_2d_mid) + (self.w_surface + 2.*self.dt*self.q_2d_old/h_mid +
                             inner(2.*self.uv_2d_mid - self.uv_2d_old, grad(d)) - self.fields.slide_source_2d)/h_mid)/(par*self.dt)
                q_2d = self.fields.q_2d
                self.solve_poisson_eq(q_2d, self.fields.uv_3d, self.fields.w_3d, A=A, B=B, C=C, multi_layers=False)
                # update uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = inner(self.uv_2d_mid - par*self.dt*(grad(q_2d) + A*q_2d), uv_test)*dx
                solve(a == l, self.fields.uv_2d)
                # update w_surf
                a = w_tri*w_test*dx
                l = (self.w_surface + 2.*self.dt*q_2d/h_mid + inner(self.uv_2d_mid - self.uv_2d_old, grad(d)))*w_test*dx
                if not pressure_projection:
                    l = (self.w_surface + 2.*self.dt*self.q_2d_old/h_mid + 2.*self.dt/h_mid*q_2d + \
                         inner(self.uv_2d_mid - self.uv_2d_old, grad(d)))*w_test*dx
                solve(a == l, self.w_surface)
                if not pressure_projection:
                    q_2d.project(self.q_2d_old + q_2d)
                solver_nh.solve()
                if self.options.set_vertical_2d:
                    self.set_vertical_2d()
  
            elif reduced_two_layer_NH_solver:
                timestepper_layer_integrated.advance(self.simulation_time, update_forcings)
                timestepper_layer_difference.advance(self.simulation_time, update_forcings)
                self.uv_av_2.assign(uv_2d)
                self.uv_av_1.assign(self.fields.uv_delta)
                self.elev_2d_mid.assign(elev_2d)
                if len(self.options.alpha_nh) >= 1:
                    alpha = self.options.alpha_nh[0]
                else:
                    alpha = 0.15
                beta = self.options.beta_nh
                depth = self.bathymetry_dg
                h_mid = self.elev_2d_mid + depth
                term_a = (1. + alpha*beta)/2.
                term_b = (1. + alpha*beta)/2.*grad(h_mid)/h_mid - beta*grad(depth)/h_mid
                term_c = (2.*alpha + alpha*beta - 1.)/2.
                term_d = (alpha*beta - 2.*alpha - 1.)/2.*grad(h_mid)/h_mid + (2. - beta)*grad(depth)/h_mid
                alpha_1 = 1./(1. - alpha)
                alpha_2 = -alpha_1
                alpha_3 = (-2.*alpha**2 + 2.*alpha - 1.)/(2.*alpha)
                alpha_4 = (2.*alpha - 1.)/(2.*alpha)
                # coefficients for Poisson equation 'div(grad(q)) + inner(A, grad(q)) + B*q = C'
                C = (h_mid*(1.5*div(self.uv_av_2) + 0.5*div(self.uv_av_1)) + inner(grad(h_mid), 1.5*self.uv_av_2 + 0.5*self.uv_av_1) + 
                    self.w_interface - inner(alpha_1*self.uv_av_2 + alpha_2*self.uv_av_1, grad(self.elev_2d_mid)) -
                    inner(alpha_3*self.uv_av_2 + alpha_4*self.uv_av_1, grad(h_mid)))/(h_mid*(1.5*term_a + 0.5*term_c)*self.dt)
                A = (h_mid*(1.5*term_b + 0.5*term_d) + grad(h_mid)*(1.5*term_a + 0.5*term_c) - 
                     ((alpha_1*term_a + alpha_2*term_c)*grad(self.elev_2d_mid) +
                      (alpha_3*term_a + alpha_4*term_c)*grad(h_mid)))/(h_mid*(1.5*term_a + 0.5*term_c))
                B = (h_mid*(1.5*div(term_b) + 0.5*div(term_d)) + inner(grad(h_mid), 1.5*term_b + 0.5*term_d) - 2./((1. - alpha)*h_mid) - 
                     (inner(alpha_1*term_b + alpha_2*term_d, grad(self.elev_2d_mid)) + 
                      inner(alpha_3*term_b + alpha_4*term_d, grad(h_mid))))/(h_mid*(1.5*term_a + 0.5*term_c))
                self.solve_poisson_eq(self.fields.q_2d, self.fields.uv_3d, self.fields.w_3d, A=A, B=B, C=C, multi_layers=False)

                # update uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = inner(self.uv_av_2 - self.dt*(term_a*grad(self.fields.q_2d) + term_b*self.fields.q_2d), uv_test)*dx
                solve(a == l, uv_2d)
                # update uv_delta
                a = inner(uv_tri, uv_test)*dx
                l = inner(self.uv_av_1 - self.dt*(term_c*grad(self.fields.q_2d) + term_d*self.fields.q_2d), uv_test)*dx
                solve(a == l, self.fields.uv_delta)
                # update w_nh
                a = w_tri*w_test*dx
                l = (self.w_interface + 2.*self.dt*self.fields.q_2d/((1. - alpha)*h_mid))*w_test*dx
                solve(a == l, self.w_interface)

                if self.simulation_time <= t_epsilon:
                    timestepper_layer_integrated.F += (self.dt*inner(term_a*grad(self.fields.q_2d) + 
                                                                     term_b*self.fields.q_2d, uta_test)*dx
                                                      )
                    prob_layer_int = NonlinearVariationalProblem(timestepper_layer_integrated.F, self.fields.solution_2d)
                    solver_layer_int = NonlinearVariationalSolver(prob_layer_int,
                                                                  solver_parameters=solver_parameters)

                    timestepper_layer_difference.F += (self.dt*inner(term_c*grad(self.fields.q_2d) + 
                                                                     term_d*self.fields.q_2d, uv_test)*dx
                                                      )
                    prob_layer_dif = NonlinearVariationalProblem(timestepper_layer_difference.F, self.fields.uv_delta)
                    solver_layer_dif = NonlinearVariationalSolver(prob_layer_dif,
                                                                  solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)
                solver_layer_int.solve()
                solver_layer_dif.solve()

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()
                    self.uv_2d_dg.project(self.fields.uv_delta)
                    self.uv_2d_dg.sub(1).assign(0.)
                    self.fields.uv_delta.project(self.uv_2d_dg)

            elif coupled_two_layer_NH_solver:
                timestepper_layer_integrated.advance(self.simulation_time, update_forcings)
                timestepper_layer_difference.advance(self.simulation_time, update_forcings)
                self.uv_av_2.assign(uv_2d)
                self.uv_av_1.assign(self.fields.uv_delta)
                self.elev_2d_mid.assign(elev_2d)
                if len(self.options.alpha_nh) >= 1:
                    alpha = self.options.alpha_nh[0]
                else:
                    alpha = 0.5
                depth = self.bathymetry_dg
                elev_mid = self.elev_2d_mid
                h_mid = elev_mid + depth
                # common coefficients
                term_a = 1./2.
                term_b = 1./2.*grad(h_mid)/h_mid
                term_c = (2.*alpha - 1.)/2.
                term_d = -(2.*alpha + 1.)/2.*grad(h_mid)/h_mid + 2.*grad(depth)/h_mid
                term_e = alpha/2.
                term_f = alpha/2.*grad(h_mid)/h_mid - grad(depth)/h_mid
                # coefficients for 1st Poisson equation of mixed formulations
                m = 1.5
                n = 0.5
                alpha_1 = 1./(1. - alpha)
                alpha_2 = -alpha_1
                alpha_3 = (-2.*alpha**2 + 2.*alpha - 1.)/(2.*alpha)
                alpha_4 = (2.*alpha - 1.)/(2.*alpha)
                A_1 = h_mid*self.dt*(m + n)*term_e
                B_1 = h_mid*self.dt*(m*term_a + n*term_c)
                C_1 = h_mid*self.dt*(m + n)*term_f + grad(h_mid)*self.dt*(m + n)*term_e - self.dt*(
                      (alpha_1 + alpha_2)*term_e*grad(elev_mid) + (alpha_3 + alpha_4)*term_e*grad(h_mid))
                D_1 = h_mid*self.dt*(m*term_b + n*term_d) + grad(h_mid)*self.dt*(m*term_a + n*term_c) - self.dt*(
                      (alpha_1*term_a + alpha_2*term_c)*grad(elev_mid) + (alpha_3*term_a + alpha_4*term_c)*grad(h_mid))
                E_1 = h_mid*self.dt*(m + n)*div(term_f) + (m + n)*self.dt*inner(term_f, grad(h_mid)) - self.dt*(
                      inner((alpha_1 + alpha_2)*term_f, grad(elev_mid)) + inner((alpha_3 + alpha_4)*term_f, grad(h_mid)))
                F_1 = h_mid*self.dt*div(m*term_b + n*term_d) + self.dt*inner(m*term_b + n*term_d, grad(h_mid)) - 2.*self.dt/((1. - alpha)*h_mid) - \
                      self.dt*(inner(alpha_1*term_b + alpha_2*term_d, grad(elev_mid)) + inner(alpha_3*term_b + alpha_4*term_d, grad(h_mid)))
                G_1 = div((m*self.uv_av_2 + n*self.uv_av_1)*h_mid) - (inner(alpha_1*self.uv_av_2 + alpha_2*self.uv_av_1, grad(elev_mid)) + 
                      inner(alpha_3*self.uv_av_2 + alpha_4*self.uv_av_1, grad(h_mid))) + self.w_12
                # coefficients for 2nd Poisson equation of mixed formulations
                m = 0.5
                n = 0.5
                alpha_1 = (2.*alpha - 1.)/(1. - alpha)
                alpha_2 = -1./(1. - alpha)
                alpha_3 = (2.*alpha**2 - 6.*alpha + 3.)/(2.*(1. - alpha))
                alpha_4 = (3. - 2.*alpha)/(2.*(1. - alpha))
                A_2 = h_mid*self.dt*(m + n)*term_e
                B_2 = h_mid*self.dt*(m*term_a + n*term_c)
                C_2 = h_mid*self.dt*(m + n)*term_f + grad(h_mid)*self.dt*(m + n)*term_e - self.dt*(
                      (alpha_1 + alpha_2)*term_e*grad(elev_mid) + (alpha_3 + alpha_4)*term_e*grad(h_mid))
                D_2 = h_mid*self.dt*(m*term_b + n*term_d) + grad(h_mid)*self.dt*(m*term_a + n*term_c) - self.dt*(
                      (alpha_1*term_a + alpha_2*term_c)*grad(elev_mid) + (alpha_3*term_a + alpha_4*term_c)*grad(h_mid))
                E_2 = h_mid*self.dt*(m + n)*div(term_f) + (m + n)*self.dt*inner(term_f, grad(h_mid)) - 2.*self.dt/(alpha*h_mid) - \
                      self.dt*(inner((alpha_1 + alpha_2)*term_f, grad(elev_mid)) + inner((alpha_3 + alpha_4)*term_f, grad(h_mid)))
                F_2 = h_mid*self.dt*div(m*term_b + n*term_d) + self.dt*inner(m*term_b + n*term_d, grad(h_mid)) + 2.*self.dt/(alpha*h_mid) - \
                      self.dt*(inner(alpha_1*term_b + alpha_2*term_d, grad(elev_mid)) + inner(alpha_3*term_b + alpha_4*term_d, grad(h_mid)))
                G_2 = div((m*self.uv_av_2 + n*self.uv_av_1)*h_mid) - (inner(alpha_1*self.uv_av_2 + alpha_2*self.uv_av_1, grad(elev_mid)) + 
                      inner(alpha_3*self.uv_av_2 + alpha_4*self.uv_av_1, grad(h_mid))) + self.w_01

                # build the solver for the mixed Poisson equations
                if self.simulation_time <= t_epsilon:
                    q_0, q_1 = split(self.q_mixed_two_layers)
                    q_test_0, q_test_1 = TestFunctions(self.function_spaces.q_mixed_two_layers)
                    f = 0
                    f += (-A_1*dot(grad(q_0), grad(q_test_0))*dx - A_2*dot(grad(q_0), grad(q_test_1))*dx - 
                          B_1*dot(grad(q_1), grad(q_test_0))*dx - B_2*dot(grad(q_1), grad(q_test_1))*dx - 
                          q_0*div(C_1*q_test_0)*dx - q_0*div(C_2*q_test_1)*dx - 
                          q_1*div(D_1*q_test_0)*dx - q_1*div(D_2*q_test_1)*dx + 
                          (E_1*q_0*q_test_0 + E_2*q_0*q_test_1)*dx + 
                          (F_1*q_1*q_test_0 + F_2*q_1*q_test_1)*dx -
                          (G_1*q_test_0 + G_2*q_test_1)*dx
                         )
                    if self.landslide:
                        f += 2.*self.fields.slide_source_2d*(q_test_0 + q_test_1)*dx
                    for bnd_marker in self.boundary_markers:
                        func = self.bnd_functions['shallow_water'].get(bnd_marker)
                        ds_bnd = ds(int(bnd_marker))
                        if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                            f += q_0*inner(C_1, self.normal_2d)*q_test_0*ds_bnd + q_0*inner(C_2, self.normal_2d)*q_test_1*ds_bnd + \
                                 q_1*inner(D_1, self.normal_2d)*q_test_0*ds_bnd + q_1*inner(D_2, self.normal_2d)*q_test_1*ds_bnd

                    prob = NonlinearVariationalProblem(f, self.q_mixed_two_layers)
                    solver = NonlinearVariationalSolver(prob,
                                                        solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu'})
                solver.solve()
                self.q_0, self.q_1 = self.q_mixed_two_layers.split()
                self.fields.q_2d.assign(self.q_0)

                # update uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = inner(self.uv_av_2 - self.dt*(term_e*grad(self.q_0) + term_f*self.q_0 + 
                    term_a*grad(self.q_1) + term_b*self.q_1), uv_test)*dx
                solve(a == l, uv_2d)
                self.uv_2d_mid.assign(uv_2d)
                # update uv_delta
                a = inner(uv_tri, uv_test)*dx
                l = inner(self.uv_av_1 - self.dt*(term_e*grad(self.q_0) + term_f*self.q_0 + 
                    term_c*grad(self.q_1) + term_d*self.q_1), uv_test)*dx
                solve(a == l, self.fields.uv_delta)
                # update w_12
                a = w_tri*w_test*dx
                l = (self.w_12 + 2.*self.dt*self.q_1/((1. - alpha)*h_mid))*w_test*dx
                solve(a == l, self.w_12)
                # update w_01
                a = w_tri*w_test*dx
                l = (self.w_01 + 2.*self.dt*(self.q_0 - self.q_1)/(alpha*h_mid))*w_test*dx
                solve(a == l, self.w_01)

                if self.simulation_time <= t_epsilon:
                    timestepper_layer_integrated.F += (self.dt*inner(term_e*grad(self.q_0) + term_a*grad(self.q_1) + 
                                                                     term_f*self.q_0 + term_b*self.q_1, uta_test)*dx
                                                      )
                    prob_layer_int = NonlinearVariationalProblem(timestepper_layer_integrated.F, self.fields.solution_2d)
                    solver_layer_int = NonlinearVariationalSolver(prob_layer_int,
                                                                  solver_parameters=solver_parameters)

                    timestepper_layer_difference.F += (self.dt*inner(term_e*grad(self.q_0) + term_c*grad(self.q_1) + 
                                                                     term_f*self.q_0 + term_d*self.q_1, uv_test)*dx
                                                      )
                    prob_layer_dif = NonlinearVariationalProblem(timestepper_layer_difference.F, self.fields.uv_delta)
                    solver_layer_dif = NonlinearVariationalSolver(prob_layer_dif,
                                                                  solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)
                solver_layer_int.solve()
                #solver_layer_dif.solve()
                uv_2d.assign(self.uv_2d_mid)

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()
                    self.uv_2d_dg.project(self.fields.uv_delta)
                    self.uv_2d_dg.sub(1).assign(0.)
                    self.fields.uv_delta.project(self.uv_2d_dg)

            elif coupled_three_layer_NH_solver: # i.e. standard three-layer NH model
                if self.simulation_time <= t_epsilon:               
                    timestepper_three_layer_difference_1 = timeintegrator.CrankNicolson(self.eq_sw_mom, self.fields.uv_delta,
                                                              fields, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit,
                                                              semi_implicit=False,
                                                              theta=0.5) 
                    timestepper_three_layer_difference_2 = timeintegrator.CrankNicolson(self.eq_sw_mom, self.fields.uv_delta_2,
                                                              fields, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit,
                                                              semi_implicit=False,
                                                              theta=0.5)
                timestepper_depth_integrated.advance(self.simulation_time, update_forcings)
                self.uv_2d_mid.assign(self.fields.uv_2d)
                timestepper_three_layer_difference_1.advance(self.simulation_time, update_forcings)
                timestepper_three_layer_difference_2.advance(self.simulation_time, update_forcings)
                du_1 = self.fields.uv_delta # i.e. uv of layer 1
                du_2 = self.fields.uv_delta_2 # i.e. uv of layer 2
                du_3 = self.fields.uv_delta_3 # i.e. uv of layer 3
                self.elev_2d_mid.assign(elev_2d)
                if len(self.options.alpha_nh) >= 2:
                    alpha_1 = self.options.alpha_nh[0]
                    alpha_2 = self.options.beta_nh[1]
                else:
                    alpha_1 = 1./3.
                    alpha_2 = 1./3.
                alpha_3 = 1. - alpha_1 - alpha_2
                du_3.project(1./alpha_3*(uv_2d - (alpha_1*du_1 + alpha_2*du_2)))
                h_mid = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(),self.bathymetry_dg, self.options).get_total_depth(self.elev_2d_mid)
                h_1 = alpha_1*h_mid
                h_2 = alpha_2*h_mid
                h_3 = alpha_3*h_mid
                # z-coordinate of layer interface
                z_0 = -self.bathymetry_dg
                z_1 = z_0 + h_1
                z_2 = z_1 + h_2
                z_3 = z_2 + h_3
                z_01 = 0.5*(z_0 + z_1)
                z_12 = 0.5*(z_1 + z_2)
                z_23 = 0.5*(z_2 + z_3)

                # build the solver for the mixed Poisson equations
                if self.simulation_time <= t_epsilon:
                    q_0, q_1, q_2 = split(self.q_mixed_three_layers)
                    q_test_0, q_test_1, q_test_2 = TestFunctions(self.function_spaces.q_mixed_three_layers)
                    # first continuity equation: div(h_1*u_1) + w_01 = dot(u_z0, grad(z_0)) + dot(u_z1, grad(z_1))
                    # weak form of div(h_1*u_1)
                    f1 = div(h_1*du_1)*q_test_0*dx + 0.5*alpha_1*self.dt*h_mid*dot(grad(q_0+q_1), grad(q_test_0))*dx + \
                                                     self.dt*(q_0-q_1)*dot(grad(z_01), grad(q_test_0))*dx

                    # weak form of w_01
                    f1 += (self.w_01 + 2.*self.dt*(q_0 - q_1)/h_1)*q_test_0*dx
                    # weak form of RHS terms
                    grad_1_layer1 = grad(z_0 + z_01)
                    grad_2_layer1 = grad(0.5*h_1)
                    f1 -= (dot(grad_1_layer1, du_1) + dot(grad_2_layer1, du_2))*q_test_0*dx - \
                          self.dt*(-1./h_1*div(grad_1_layer1*(0.5*h_1)*q_test_0)*(q_0+q_1) - 1./h_2*div(grad_2_layer1*(0.5*h_2)*q_test_0)*(q_1+q_2))*dx - \
                          self.dt*(1./h_1*dot(grad_1_layer1, grad(z_01))*(q_0-q_1) + 1./h_2*dot(grad_2_layer1, grad(z_12))*(q_1-q_2))*q_test_0*dx

                    # second continuity equation: div(2*h_1*u_1 + h_2*u_2) + w_12 = dot(u_z1, grad(z_1)) + dot(u_z2, grad(z_2))
                    # weak form of div(2*h_1*u_1 + h_2*u_2)
                    f2 = 2*(div(h_1*du_1)*q_test_1*dx + 0.5*alpha_1*self.dt*h_mid*dot(grad(q_0+q_1), grad(q_test_1))*dx + \
                                                     self.dt*(q_0-q_1)*dot(grad(z_01), grad(q_test_1))*dx) + \
                         div(h_2*du_2)*q_test_1*dx + 0.5*alpha_2*self.dt*h_mid*dot(grad(q_1+q_2), grad(q_test_1))*dx + \
                                                     self.dt*(q_1-q_2)*dot(grad(z_12), grad(q_test_1))*dx

                    # weak form of w_12
                    f2 += (self.w_12 + 2.*self.dt*(q_1 - q_2)/h_2)*q_test_1*dx
                    # weak form of RHS terms
                    grad_1_layer2 = grad(0.5*z_1)
                    grad_2_layer2 = grad(z_12)
                    grad_3_layer2 = grad(0.5*z_2)
                    f2 -= (dot(grad_1_layer2, du_1) + dot(grad_2_layer2, du_2) + dot(grad_3_layer2, du_3))*q_test_1*dx - \
                          self.dt*(-1./h_1*div(grad_1_layer2*(0.5*h_1)*q_test_1)*(q_0+q_1) - 1./h_2*div(grad_2_layer2*(0.5*h_2)*q_test_1)*(q_1+q_2) - 
                                   1./h_3*div(grad_3_layer2*(0.5*h_3)*q_test_1)*(q_2))*dx - \
                          self.dt*(1./h_1*dot(grad_1_layer2, grad(z_01))*(q_0-q_1) + 1./h_2*dot(grad_2_layer2, grad(z_12))*(q_1-q_2) + 
                                   1./h_3*dot(grad_3_layer2, grad(z_23))*(q_2))*q_test_1*dx

                    # third continuity equation: div(2*h_1*u_1 + 2*h_2*u_2 + h_3*u_3) + w_23 = dot(u_z2, grad(z_2)) + dot(u_z3, grad(z_3))
                    # weak form of div(2*h_1*u_1 + 2*h_2*u_2 + h_3*u_3)
                    f3 = 2*(div(h_1*du_1)*q_test_2*dx + 0.5*alpha_1*self.dt*h_mid*dot(grad(q_0+q_1), grad(q_test_2))*dx + \
                                                     self.dt*(q_0-q_1)*dot(grad(z_01), grad(q_test_2))*dx) + \
                         2*(div(h_2*du_2)*q_test_2*dx + 0.5*alpha_2*self.dt*h_mid*dot(grad(q_1+q_2), grad(q_test_2))*dx + \
                                                     self.dt*(q_1-q_2)*dot(grad(z_12), grad(q_test_2))*dx) + \
                         div(h_3*du_3)*q_test_2*dx + 0.5*alpha_3*self.dt*h_mid*dot(grad(q_2), grad(q_test_2))*dx + \
                                                     self.dt*(q_2)*dot(grad(z_23), grad(q_test_2))*dx

                    # weak form of w_23
                    f3 += (self.w_23 + 2.*self.dt*(q_2)/h_3)*q_test_2*dx
                    # weak form of RHS terms
                    grad_2_layer3 = grad(-0.5*h_3)
                    grad_3_layer3 = grad(z_3 + z_23)
                    f3 -= (dot(grad_2_layer3, du_2) + dot(grad_3_layer3, du_3))*q_test_0*dx - \
                          self.dt*(-1./h_2*div(grad_2_layer3*(0.5*h_2)*q_test_2)*(q_1+q_2) - 1./h_3*div(grad_3_layer3*(0.5*h_3)*q_test_2)*(q_2))*dx - \
                          self.dt*(1./h_2*dot(grad_2_layer3, grad(z_12))*(q_1-q_2) + 1./h_3*dot(grad_3_layer3, grad(z_23))*(q_2))*q_test_2*dx

                    f = f1 + f2 + f3

                    for bnd_marker in self.boundary_markers:
                        func = self.bnd_functions['shallow_water'].get(bnd_marker)
                        ds_bnd = ds(int(bnd_marker))
                        if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                            # first equation
                            f += -self.dt*(q_0-q_1)*dot(grad(z_01), self.normal_2d)*q_test_0*ds_bnd + \
                                 self.dt*(1./h_1*dot(grad_1_layer1, self.normal_2d)*(0.5*h_1)*(q_0+q_1)*q_test_0 +
                                          1./h_2*dot(grad_2_layer1, self.normal_2d)*(0.5*h_2)*(q_1+q_2)*q_test_0)*ds_bnd
                            # second equation
                            f += -2*self.dt*(q_0-q_1)*dot(grad(z_01), self.normal_2d)*q_test_1*ds_bnd - \
                                 self.dt*(q_1-q_2)*dot(grad(z_12), self.normal_2d)*q_test_1*ds_bnd + \
                                 self.dt*(1./h_1*dot(grad_1_layer2, self.normal_2d)*(0.5*h_1)*(q_0+q_1)*q_test_1 +
                                          1./h_2*dot(grad_2_layer2, self.normal_2d)*(0.5*h_2)*(q_1+q_2)*q_test_1 +
                                          1./h_3*dot(grad_3_layer2, self.normal_2d)*(0.5*h_3)*(q_2)*q_test_1)*ds_bnd
                            # third equation
                            f += -2*self.dt*(q_0-q_1)*dot(grad(z_01), self.normal_2d)*q_test_2*ds_bnd - \
                                 2*self.dt*(q_1-q_2)*dot(grad(z_12), self.normal_2d)*q_test_2*ds_bnd - \
                                 self.dt*(q_2)*dot(grad(z_23), self.normal_2d)*q_test_2*ds_bnd + \
                                 self.dt*(1./h_2*dot(grad_2_layer3, self.normal_2d)*(0.5*h_2)*(q_1+q_2)*q_test_2 +
                                          1./h_3*dot(grad_3_layer3, self.normal_2d)*(0.5*h_3)*(q_2)*q_test_2)*ds_bnd

                    prob = NonlinearVariationalProblem(f, self.q_mixed_three_layers)
                    solver = NonlinearVariationalSolver(prob,
                                                        solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu'})
                solver.solve()
                self.q_0, self.q_1, self.q_2 = self.q_mixed_three_layers.split()
                self.fields.q_2d.assign(self.q_0)

                # update uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = inner(uv_2d - self.dt/h_mid*(grad((self.q_0 + self.q_1)/2.*h_1) +
                                                 grad((self.q_1 + self.q_2)/2.*h_2) +
                                                 grad((self.q_2)/2.*h_3) + self.q_0*grad(z_0)), uv_test)*dx
                solve(a == l, uv_2d)
                self.uv_2d_mid.assign(uv_2d)
                # update uv_delta
                a = inner(uv_tri, uv_test)*dx
                l = inner(du_1 - self.dt/h_1*(grad((self.q_0 + self.q_1)/2.*h_1) + self.q_0*grad(z_0) - self.q_1*grad(z_1)), uv_test)*dx
                solve(a == l, self.fields.uv_delta)
                # update uv_delta_2
                a = inner(uv_tri, uv_test)*dx
                l = inner(du_2 - self.dt/h_2*(grad((self.q_1 + self.q_2)/2.*h_2) + self.q_1*grad(z_1) - self.q_2*grad(z_2)), uv_test)*dx
                solve(a == l, self.fields.uv_delta_2)
                # update w_23
                a = w_tri*w_test*dx
                l = (self.w_23 + 2.*self.dt*self.q_2/h_3)*w_test*dx
                solve(a == l, self.w_23)
                # update w_12
                a = w_tri*w_test*dx
                l = (self.w_12 + 2.*self.dt*(self.q_1 - self.q_2)/h_2)*w_test*dx
                solve(a == l, self.w_12)
                # update w_01
                a = w_tri*w_test*dx
                l = (self.w_01 + 2.*self.dt*(self.q_0 - self.q_1)/h_1)*w_test*dx
                solve(a == l, self.w_01)

                if self.simulation_time <= t_epsilon:
                    timestepper_depth_integrated.F += (self.dt*1./h_mid*inner(grad((self.q_0 + self.q_1)/2.*h_1) +
                                                                                    grad((self.q_1 + self.q_2)/2.*h_2) +
                                                                                    grad((self.q_2)/2.*h_3) + self.q_0*grad(z_0), uta_test)*dx
                                                            )
                    prob_three_layer_int = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                    solver_three_layer_int = NonlinearVariationalSolver(prob_three_layer_int,
                                                                        solver_parameters=solver_parameters)

                    timestepper_three_layer_difference_1.F += (self.dt*1./h_1*inner(grad((self.q_0 + self.q_1)/2.*h_1) +
                                                                                      self.q_0*grad(z_0) - self.q_1*grad(z_1), uv_test)*dx
                                                              )
                    prob_three_layer_dif_1 = NonlinearVariationalProblem(timestepper_three_layer_difference_1.F, self.fields.uv_delta)
                    solver_three_layer_dif_1 = NonlinearVariationalSolver(prob_three_layer_dif_1,
                                                                          solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)
                    timestepper_three_layer_difference_2.F += (self.dt*1./h_2*inner(grad((self.q_1 + self.q_2)/2.*h_2) +
                                                                                      self.q_1*grad(z_1) - self.q_2*grad(z_2), uv_test)*dx
                                                              )
                    prob_three_layer_dif_2 = NonlinearVariationalProblem(timestepper_three_layer_difference_2.F, self.fields.uv_delta_2)
                    solver_three_layer_dif_2 = NonlinearVariationalSolver(prob_three_layer_dif_2,
                                                                          solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)

                solver_three_layer_int.solve()
                #solver_three_layer_dif_1.solve()
                #solver_three_layer_dif_2.solve()
                uv_2d.assign(self.uv_2d_mid)

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()
                    self.uv_2d_dg.project(du_1)
                    self.uv_2d_dg.sub(1).assign(0.)
                    du_1.project(self.uv_2d_dg)
                    self.uv_2d_dg.project(du_2)
                    self.uv_2d_dg.sub(1).assign(0.)
                    du_2.project(self.uv_2d_dg)

            elif arbitrary_multi_layer_NH_solver: # i.e. multi-layer NH model
                ### layer thickness accounting for total depth
                alpha = self.options.alpha_nh
                if len(self.options.alpha_nh) < self.n_layers:
                    n = self.n_layers - len(self.options.alpha_nh)
                    sum = 0.
                    if len(self.options.alpha_nh) >= 1:
                        for k in range(len(self.options.alpha_nh)):
                            sum = sum + self.options.alpha_nh[k]
                    for k in range(n):
                        alpha.append((1. - sum)/n)
                if self.n_layers == 1:
                    alpha[0] = 1.
                ###
                h_old = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg_old, self.options).get_total_depth(self.elev_2d_old)
                h_layer_old = [h_old*alpha[0]]
                for k in range(self.n_layers):
                    h_layer_old.append(h_old*alpha[k])
                    # add ghost layer
                    if k == self.n_layers - 1:
                        h_layer_old.append(h_old*alpha[k])
                z_old_dic = {'z_0': -self.bathymetry_dg_old}
                for k in range(self.n_layers):
                    z_old_dic['z_'+str(k+1)] = z_old_dic['z_'+str(k)] + h_layer_old[k+1]
                    z_old_dic['z_'+str(k)+str(k+1)] = 0.5*(z_old_dic['z_'+str(k)] + z_old_dic['z_'+str(k+1)])

                # solve 2D depth-integrated equations initially
                timestepper_depth_integrated.advance(self.simulation_time, update_forcings)
                self.elev_2d_mid.assign(elev_2d)

                # update layer thickness and z-coordinate
                h_mid = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(self.elev_2d_mid)
                h_layer = [h_mid*alpha[0]]
                for k in range(self.n_layers):
                    h_layer.append(h_mid*alpha[k])
                    # add ghost layer
                    if k == self.n_layers - 1:
                        h_layer.append(h_mid*alpha[k])
                z_dic = {'z_0': -self.bathymetry_dg}
                for k in range(self.n_layers):
                    z_dic['z_'+str(k+1)] = z_dic['z_'+str(k)] + h_layer[k+1]
                    z_dic['z_'+str(k)+str(k+1)] = 0.5*(z_dic['z_'+str(k)] + z_dic['z_'+str(k+1)])

                # velocities at the interface
                u_dic = {}
                w_dic = {}
                omega_dic = {}
                for k in range(self.n_layers + 1):
                    # old uv and w velocities at the interface
                    if k == 0:
                        u_dic['z_'+str(k)] = 2.*getattr(self, 'uv_av_'+str(k+1)) - (h_layer[k+2]/(h_layer[k+1] + h_layer[k+2])*getattr(self, 'uv_av_'+str(k+1)) + 
                                                                                    h_layer[k+1]/(h_layer[k+1] + h_layer[k+2])*getattr(self, 'uv_av_'+str(k+2)))
                        w_dic['z_'+str(k)] = -self.fields.slide_source_2d + inner(u_dic['z_'+str(k)], grad(z_dic['z_'+str(k)]))
                    elif k > 0 and k < self.n_layers:
                        u_dic['z_'+str(k)] = h_layer[k+1]/(h_layer[k] + h_layer[k+1])*getattr(self, 'uv_av_'+str(k)) + \
                                             h_layer[k]/(h_layer[k] + h_layer[k+1])*getattr(self, 'uv_av_'+str(k+1))
                        w_dic['z_'+str(k)] = 2.*getattr(self, 'w_'+str(k-1)+str(k)) - w_dic['z_'+str(k-1)]
                    else: # i.e. k == self.n_layers
                        u_dic['z_'+str(k)] = 2.*getattr(self, 'uv_av_'+str(k)) - u_dic['z_'+str(k-1)]
                        w_dic['z_'+str(k)] = 2.*getattr(self, 'w_'+str(k-1)+str(k)) - w_dic['z_'+str(k-1)]
                    # relative vertical velocity due to mesh movement
                    omega_dic['z_'+str(k)] = w_dic['z_'+str(k)] - (z_dic['z_'+str(k)] - z_old_dic['z_'+str(k)])/self.dt - inner(u_dic['z_'+str(k)], grad(z_dic['z_'+str(k)]))

                if self.simulation_time <= t_epsilon:
                    if self.n_layers >= 2:
                        timestepper_dic = {}
                        for k in range(self.n_layers - 1):
                            timestepper_dic['layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_sw_mom, getattr(self, 'uv_av_'+str(k+1)),
                                                              fields, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit,
                                                              semi_implicit=False,
                                                              theta=0.5)
                            consider_mesh_relative_velocity = False
                            if consider_mesh_relative_velocity:
                                timestepper_dic['layer_'+str(k+1)].F += self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(u_dic['z_'+str(k+1)] - getattr(self, 'uv_av_'+str(k+1))) -
                                                                                          omega_dic['z_'+str(k)]*(u_dic['z_'+str(k)] - getattr(self, 'uv_av_'+str(k+1))), uv_test)*dx
                                timestepper_dic['layer_'+str(k+1)].update_solver()

                if self.n_layers >= 2:
                    sum_uv_av = 0. 
                    # except the layer adjacent to the free surface
                    for k in range(self.n_layers - 1):
                        timestepper_dic['layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                        #sum_uv_av += getattr(self, 'uv_av_'+str(k+1)) # cannot sum by this way
                        sum_uv_av = sum_uv_av + alpha[k]*getattr(self, 'uv_av_'+str(k+1))
                    getattr(self, 'uv_av_'+str(self.n_layers)).project((uv_2d - sum_uv_av)/alpha[self.n_layers-1])

                # build the solver for the mixed Poisson equations
                if self.simulation_time <= t_epsilon:
                    q_test = TestFunctions(self.function_spaces.q_mixed_n_layers)
                    q_tuple = split(self.q_mixed_n_layers)
                    if self.n_layers == 1:
                        q_test = [TestFunction(self.q_0.function_space())]
                        q_tuple = [self.q_0]
                    # re-arrange the list of q
                    q = []
                    for k in range(self.n_layers):
                        q.append(q_tuple[k])
                        if k == self.n_layers - 1:
                            # free-surface NH pressure
                            q.append(0.)
                    f = 0.
                    for k in range(self.n_layers):
                        # weak form of div(h_{k+1}*uv_av_{k+1})
                        div_hu_term = div(h_layer[k+1]*getattr(self, 'uv_av_'+str(k+1)))*q_test[k]*dx + \
                                      0.5*self.dt*h_layer[k+1]*dot(grad(q[k]+q[k+1]), grad(q_test[k]))*dx + \
                                      self.dt*(q[k]-q[k+1])*dot(grad(z_dic['z_'+str(k)+str(k+1)]), grad(q_test[k]))*dx
                        if k >= 1:
                            for i in range(k):
                                div_hu_term += 2.*(div(h_layer[i+1]*getattr(self, 'uv_av_'+str(i+1)))*q_test[k]*dx + \
                                               0.5*self.dt*h_layer[i+1]*dot(grad(q[i]+q[i+1]), grad(q_test[k]))*dx + \
                                               self.dt*(q[i]-q[i+1])*dot(grad(z_dic['z_'+str(i)+str(i+1)]), grad(q_test[k]))*dx)
                        # weak form of w_{k}{k+1}
                        vert_vel_term = 2.*(getattr(self, 'w_'+str(k)+str(k+1)) + self.dt*(q[k] - q[k+1])/h_layer[k+1])*q_test[k]*dx
                        consider_vert_adv = False#True
                        if consider_vert_adv: # TODO if make sure that considering vertical advection is benefitial, delete this logical variable
                            #vert_vel_term += -2.*self.dt*dot(getattr(self, 'uv_av_'+str(k+1)), grad(getattr(self, 'w_'+str(k)+str(k+1))))*q_test[k]*dx
                            vert_vel_term += 2.*self.dt*(div(getattr(self, 'uv_av_'+str(k+1))*q_test[k])*getattr(self, 'w_'+str(k)+str(k+1))*dx -
                                                         avg(getattr(self, 'w_'+str(k)+str(k+1)))*jump(q_test[k], inner(getattr(self, 'uv_av_'+str(k+1)), self.normal_2d))*dS)
                            if consider_mesh_relative_velocity:
                                vert_vel_term += -2.*self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(w_dic['z_'+str(k+1)] - getattr(self, 'w_'+str(k)+str(k+1))) -
                                                                                omega_dic['z_'+str(k)]*(w_dic['z_'+str(k)] - getattr(self, 'w_'+str(k)+str(k+1))), q_test[k])*dx
                        # weak form of RHS terms
                        if k == 0: # i.e. the layer adjacent to the bottom
                            if self.n_layers == 1:
                                grad_1_layer1 = grad(z_dic['z_'+str(k)] + z_dic['z_'+str(k+1)])
                                interface_term = dot(grad_1_layer1, getattr(self, 'uv_av_'+str(k+1)))*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]))*dx - \
                                                 self.dt*(1./h_layer[k+1]*dot(grad_1_layer1, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                            else:
                                grad_1_layer1 = grad(2.*z_dic['z_'+str(k)] + h_layer[k+1]*h_layer[k+2]/(h_layer[k+1] + h_layer[k+2]))
                                grad_2_layer1 = grad(h_layer[k+1]*h_layer[k+1]/(h_layer[k+1] + h_layer[k+2]))
                                interface_term = (dot(grad_1_layer1, getattr(self, 'uv_av_'+str(k+1))) + dot(grad_2_layer1, getattr(self, 'uv_av_'+str(k+2))))*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]) - div(grad_2_layer1*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                                 self.dt*(1./h_layer[k+1]*dot(grad_1_layer1, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                          1./h_layer[k+2]*dot(grad_2_layer1, grad(z_dic['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        elif k == self.n_layers - 1: # i.e. the layer adjacent to the free surface
                            grad_1_layern = grad(-h_layer[k+1]*h_layer[k+1]/(h_layer[k] + h_layer[k+1]))
                            grad_2_layern = grad(2.*z_dic['z_'+str(k+1)] - h_layer[k]*h_layer[k+1]/(h_layer[k] + h_layer[k+1]))
                            interface_term = (dot(grad_1_layern, getattr(self, 'uv_av_'+str(k))) + dot(grad_2_layern, getattr(self, 'uv_av_'+str(k+1))))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layern*q_test[k])*(q[k-1]+q[k]) - div(grad_2_layern*q_test[k])*(q[k]+q[k+1]))*dx - \
                                             self.dt*(1./h_layer[k]*dot(grad_1_layern, grad(z_dic['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                      1./h_layer[k+1]*dot(grad_2_layern, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                        else:
                            grad_1_layerk = h_layer[k+1]/(h_layer[k] + h_layer[k+1])*grad(z_dic['z_'+str(k)])
                            grad_2_layerk = h_layer[k]/(h_layer[k] + h_layer[k+1])*grad(z_dic['z_'+str(k)]) + \
                                            h_layer[k+2]/(h_layer[k+1] + h_layer[k+2])*grad(z_dic['z_'+str(k+1)])
                            grad_3_layerk = h_layer[k+1]/(h_layer[k+1] + h_layer[k+2])*grad(z_dic['z_'+str(k+1)])
                            interface_term = (dot(grad_1_layerk, getattr(self, 'uv_av_'+str(k))) + 
                                              dot(grad_2_layerk, getattr(self, 'uv_av_'+str(k+1))) + 
                                              dot(grad_3_layerk, getattr(self, 'uv_av_'+str(k+2))))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layerk*q_test[k])*(q[k-1]+q[k]) - 
                                                          div(grad_2_layerk*q_test[k])*(q[k]+q[k+1]) - 
                                                          div(grad_3_layerk*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                             self.dt*(1./h_layer[k]*dot(grad_1_layerk, grad(z_dic['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                      1./h_layer[k+1]*dot(grad_2_layerk, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                      1./h_layer[k+2]*dot(grad_3_layerk, grad(z_dic['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        # weak form of slide source term
                        if self.landslide:
                            slide_source_term = -2.*self.fields.slide_source_2d*q_test[k]*dx
                            f += slide_source_term
                        f += div_hu_term + vert_vel_term - interface_term

                        for bnd_marker in self.boundary_markers:
                            func = self.bnd_functions['shallow_water'].get(bnd_marker)
                            ds_bnd = ds(int(bnd_marker))
                            if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                                # bnd terms of div(h_{k+1}*uv_av_{k+1})
                                f += -self.dt*(q[k]-q[k+1])*dot(grad(z_dic['z_'+str(k)+str(k+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                if k >= 1:
                                    for i in range(k):
                                        f += -2*self.dt*(q[i]-q[i+1])*dot(grad(z_dic['z_'+str(i)+str(i+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                # bnd terms of RHS terms about interface connection
                                if k == 0:
                                    if self.n_layers == 1:
                                        f += 0.5*self.dt*dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1])*q_test[k]*ds_bnd
                                    else:
                                        f += 0.5*self.dt*(dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1]) + 
                                                          dot(grad_2_layer1, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd
                                elif k == self.n_layers - 1:
                                    f += 0.5*self.dt*(dot(grad_1_layern, self.normal_2d)*(q[k-1]+q[k]) + 
                                                      dot(grad_2_layern, self.normal_2d)*(q[k]+q[k+1]))*q_test[k]*ds_bnd
                                else:
                                    f += 0.5*self.dt*(dot(grad_1_layerk, self.normal_2d)*(q[k-1]+q[k]) +
                                                      dot(grad_2_layerk, self.normal_2d)*(q[k]+q[k+1]) +
                                                      dot(grad_3_layerk, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd

                    prob = NonlinearVariationalProblem(f, self.q_mixed_n_layers)
                    if self.n_layers == 1:
                        prob = NonlinearVariationalProblem(f, self.q_0)
                    solver = NonlinearVariationalSolver(prob,
                                                        solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu'})
                if self.n_layers == 1:
                    self.uv_av_1.assign(uv_2d)
                solver.solve()
                for k in range(self.n_layers):
                    if self.n_layers == 1:
                        getattr(self, 'q_'+str(k)).assign(self.q_0)
                    else:
                        getattr(self, 'q_'+str(k)).assign(self.q_mixed_n_layers.split()[k])
                    if k == self.n_layers - 1:
                        getattr(self, 'q_'+str(k+1)).assign(0.)
                self.fields.q_2d.assign(self.q_0)

                # update depth-averaged uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = 0.
                for k in range(self.n_layers):
                    l += inner(-self.dt/h_mid*grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]), uv_test)*dx
                    if k == self.n_layers - 1:
                        l += inner(uv_2d - self.dt/h_mid*(getattr(self, 'q_'+str(0))*grad(z_dic['z_'+str(0)]) - getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)])), uv_test)*dx
                solve(a == l, uv_2d)
                self.uv_2d_mid.assign(uv_2d)
                # update layer-averaged self.uv_av_{k+1}
                if self.n_layers >= 2:
                    sum_uv_av = 0.
                    for k in range(self.n_layers - 1):
                        a = inner(uv_tri, uv_test)*dx
                        l = inner(getattr(self, 'uv_av_'+str(k+1)) - self.dt/h_layer[k+1]*(grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]) + 
                                                          getattr(self, 'q_'+str(k))*grad(z_dic['z_'+str(k)]) - getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)])), uv_test)*dx
                        solve(a == l, getattr(self, 'uv_av_'+str(k+1)))
                        sum_uv_av = sum_uv_av + alpha[k]*getattr(self, 'uv_av_'+str(k+1))
                    # update layer-averaged velocity of the free-surface layer
                    getattr(self, 'uv_av_'+str(self.n_layers)).project((uv_2d - sum_uv_av)/alpha[self.n_layers-1])
                # update layer-integrated vertical velocity w_{k}{k+1}
                a = w_tri*w_test*dx
                for k in range(self.n_layers):
                    l = (getattr(self, 'w_'+str(k)+str(k+1)) + self.dt*(getattr(self, 'q_'+str(k)) - getattr(self, 'q_'+str(k+1)))/h_layer[k+1])*w_test*dx
                    if consider_vert_adv:
                        #l += -self.dt*dot(getattr(self, 'uv_av_'+str(k+1)), grad(getattr(self, 'w_'+str(k)+str(k+1))))*w_test*dx
                        l += self.dt*(div(getattr(self, 'uv_av_'+str(k+1))*w_test)*getattr(self, 'w_'+str(k)+str(k+1))*dx -
                             avg(getattr(self, 'w_'+str(k)+str(k+1)))*jump(w_test, inner(getattr(self, 'uv_av_'+str(k+1)), self.normal_2d))*dS)
                        if consider_mesh_relative_velocity:
                             l += -self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(w_dic['z_'+str(k+1)] - getattr(self, 'w_'+str(k)+str(k+1))) -
                                                              omega_dic['z_'+str(k)]*(w_dic['z_'+str(k)] - getattr(self, 'w_'+str(k)+str(k+1))), w_test)*dx
                    solve(a == l, getattr(self, 'w_'+str(k)+str(k+1)))

                if self.simulation_time <= t_epsilon:
                    for k in range(self.n_layers):
                        timestepper_depth_integrated.F += self.dt/h_mid*inner(grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]), uta_test)*dx
                        if k == self.n_layers - 1:
                            timestepper_depth_integrated.F += self.dt/h_mid*inner(getattr(self, 'q_'+str(0))*grad(z_dic['z_'+str(0)]) - 
                                                                                  getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)]), uta_test)*dx
                    prob_n_layers_int = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                    solver_n_layers_int = NonlinearVariationalSolver(prob_n_layers_int,
                                                                     solver_parameters=solver_parameters)
                    if self.n_layers >= 2:
                        solver_dic = {}
                        for k in range(self.n_layers - 1):
                            timestepper_dic['layer_'+str(k+1)].F += self.dt/h_layer[k+1]*inner(grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]) +
                                                         getattr(self, 'q_'+str(k))*grad(z_dic['z_'+str(k)]) - getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)]), uv_test)*dx
                            prob_n_layer_dif_k = NonlinearVariationalProblem(timestepper_dic['layer_'+str(k+1)].F, getattr(self, 'uv_av_'+str(k+1)))
                            solver_dic['layer_'+str(k+1)] = NonlinearVariationalSolver(prob_n_layer_dif_k,
                                                                          solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)

                # update water level elev_2d
                update_water_level = True
                solving_free_surface_eq = True
                if update_water_level:
                    if not solving_free_surface_eq:
                        solver_n_layers_int.solve()
                        #for k in range(self.n_layers - 1):
                        #    solver_dic['layer_'+str(k+1)].solve()
                        uv_2d.assign(self.uv_2d_mid)
                    else:
                        timestepper_free_surface.advance(self.simulation_time, update_forcings)
                        self.fields.elev_2d.assign(self.elev_2d_mid)

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()
                    for k in range(self.n_layers - 1):
                        self.uv_2d_dg.project(getattr(self, 'uv_av_'+str(k+1)))
                        self.uv_2d_dg.sub(1).assign(0.)
                        getattr(self, 'uv_av_'+str(k+1)).project(self.uv_2d_dg)

############
#####################
##############################

            elif arbitrary_multi_layer_NH_solver_variant_form:
                ### layer thickness accounting for total depth
                alpha = self.options.alpha_nh
                if len(self.options.alpha_nh) < self.n_layers:
                    n = self.n_layers - len(self.options.alpha_nh)
                    sum = 0.
                    if len(self.options.alpha_nh) >= 1:
                        for k in range(len(self.options.alpha_nh)):
                            sum = sum + self.options.alpha_nh[0]
                    for k in range(n):
                        alpha.append((1. - sum)/n)
                if self.n_layers == 1:
                    alpha[0] = 1.
                ###
                h_old = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg_old, self.options).get_total_depth(self.elev_2d_old)
                h_layer_old = [h_old*alpha[0]]
                for k in range(self.n_layers):
                    h_layer_old.append(h_old*alpha[k])
                    # add ghost layer
                    if k == self.n_layers - 1:
                        h_layer_old.append(h_old*alpha[k])
                z_old_dic = {'z_0': -self.bathymetry_dg_old}
                for k in range(self.n_layers):
                    z_old_dic['z_'+str(k+1)] = z_old_dic['z_'+str(k)] + h_layer_old[k+1]
                    z_old_dic['z_'+str(k)+str(k+1)] = 0.5*(z_old_dic['z_'+str(k)] + z_old_dic['z_'+str(k+1)])

                # solve 2D depth-integrated equations initially
                timestepper_depth_integrated.advance(self.simulation_time, update_forcings)
                self.elev_2d_mid.assign(elev_2d)

                # update layer thickness and z-coordinate
                h_mid = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(self.elev_2d_mid)
                h_layer = [h_mid*alpha[0]]
                for k in range(self.n_layers):
                    h_layer.append(h_mid*alpha[k])
                    # add ghost layer
                    if k == self.n_layers - 1:
                        h_layer.append(h_mid*alpha[k])
                z_dic = {'z_0': -self.bathymetry_dg}
                for k in range(self.n_layers):
                    z_dic['z_'+str(k+1)] = z_dic['z_'+str(k)] + h_layer[k+1]
                    z_dic['z_'+str(k)+str(k+1)] = 0.5*(z_dic['z_'+str(k)] + z_dic['z_'+str(k+1)])

                # velocities at the interface
                u_dic = {}
                w_dic = {}
                omega_dic = {}
                for k in range(self.n_layers + 1):
                    # uv velocities at the interface
                    if k == 0:
                        u_dic['z_'+str(k)] = 2.*getattr(self, 'uv_av_'+str(k+1)) - (h_layer[k+2]/(h_layer[k+1] + h_layer[k+2])*getattr(self, 'uv_av_'+str(k+1)) + 
                                                                                    h_layer[k+1]/(h_layer[k+1] + h_layer[k+2])*getattr(self, 'uv_av_'+str(k+2)))
                    elif k > 0 and k < self.n_layers:
                        u_dic['z_'+str(k)] = h_layer[k+1]/(h_layer[k] + h_layer[k+1])*getattr(self, 'uv_av_'+str(k)) + \
                                             h_layer[k]/(h_layer[k] + h_layer[k+1])*getattr(self, 'uv_av_'+str(k+1))
                    else: # i.e. k == self.n_layers
                        u_dic['z_'+str(k)] = 2.*getattr(self, 'uv_av_'+str(k)) - u_dic['z_'+str(k-1)]
                    # w velocities at the interface
                    if k == 0:
                        w_dic['z_'+str(k)] = -self.fields.slide_source_2d + inner(u_dic['z_'+str(k)], grad(z_dic['z_'+str(k)]))
                    else:
                        w_dic['z_'+str(k)] = getattr(self, 'w_'+str(k))
                        w_dic['z_'+str(k-1)+str(k)] = 0.5*(w_dic['z_'+str(k-1)] + w_dic['z_'+str(k)])
                    # relative vertical velocity due to mesh movement
                    omega_dic['z_'+str(k)] = w_dic['z_'+str(k)] - (z_dic['z_'+str(k)] - z_old_dic['z_'+str(k)])/self.dt - inner(u_dic['z_'+str(k)], grad(z_dic['z_'+str(k)]))

                if self.simulation_time <= t_epsilon:
                    if self.n_layers >= 2:
                        timestepper_dic = {}
                        for k in range(self.n_layers - 1):
                            timestepper_dic['layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_sw_mom, getattr(self, 'uv_av_'+str(k+1)),
                                                              fields, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit,
                                                              semi_implicit=False,
                                                              theta=0.5)
                            timestepper_dic['layer_'+str(k+1)].F += w_dic['z_'+str(k)+str(k+1)]*inner((u_dic['z_'+str(k+1)] - u_dic['z_'+str(k)])/h_layer[k+1], uv_test)*dx
                            consider_mesh_relative_velocity = False
                            if consider_mesh_relative_velocity:
                                timestepper_dic['layer_'+str(k+1)].F += self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(u_dic['z_'+str(k+1)] - getattr(self, 'uv_av_'+str(k+1))) -
                                                                                          omega_dic['z_'+str(k)]*(u_dic['z_'+str(k)] - getattr(self, 'uv_av_'+str(k+1))), uv_test)*dx
                            timestepper_dic['layer_'+str(k+1)].update_solver()

                if self.n_layers >= 2:
                    sum_uv_av = 0. 
                    # except the layer adjacent to the free surface
                    for k in range(self.n_layers - 1):
                        timestepper_dic['layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                        #sum_uv_av += getattr(self, 'uv_av_'+str(k+1)) # cannot sum by this way
                        sum_uv_av = sum_uv_av + alpha[k]*getattr(self, 'uv_av_'+str(k+1))
                    getattr(self, 'uv_av_'+str(self.n_layers)).project((uv_2d - sum_uv_av)/alpha[self.n_layers-1])

                # build the solver for the mixed Poisson equations
                if self.simulation_time <= t_epsilon:
                    q_test = TestFunctions(self.function_spaces.q_mixed_n_layers)
                    q_tuple = split(self.q_mixed_n_layers)
                    if self.n_layers == 1:
                        q_test = [TestFunction(self.q_0.function_space())]
                        q_tuple = [self.q_0]
                    # re-arrange the list of q
                    q = []
                    for k in range(self.n_layers):
                        q.append(q_tuple[k])
                        if k == self.n_layers - 1:
                            # free-surface NH pressure
                            q.append(0.)
                    f = 0.
                    for k in range(self.n_layers):
                        # weak form of `div(uv_av_{k+1})`
                        div_hu_term = div(getattr(self, 'uv_av_'+str(k+1)))*q_test[k]*dx + self.dt*dot(grad(q[k]), grad(q_test[k]))*dx
                        # weak form of `(w_{k+1}-w_{k})/h_{k+1}`
                        if k == 0: # i.e. the layer adjacent to the bottom
                            w_upper_term = 1./h_layer[k+1]*(w_dic['z_'+str(k+1)] + self.dt*(q[k] - q[k+1])/(0.5*(h_layer[k+1] + h_layer[k+2])))*q_test[k]*dx
                            w_lower_term = 0. # for flat bottom
                        elif k == self.n_layers - 1: # i.e. the layer adjacent to the free surface
                            w_upper_term = 1./h_layer[k+1]*(w_dic['z_'+str(k+1)] + self.dt*(q[k])/(0.5*h_layer[k+1]))*q_test[k]*dx
                            w_lower_term = 1./h_layer[k+1]*(w_dic['z_'+str(k)] + self.dt*(q[k-1] - q[k])/(0.5*(h_layer[k] + h_layer[k+1])))*q_test[k]*dx
                        else:
                            w_upper_term = 1./h_layer[k+1]*(w_dic['z_'+str(k+1)] + self.dt*(q[k] - q[k+1])/(0.5*(h_layer[k+1] + h_layer[k+2])))*q_test[k]*dx
                            w_lower_term = 1./h_layer[k+1]*(w_dic['z_'+str(k)] + self.dt*(q[k-1] - q[k])/(0.5*(h_layer[k] + h_layer[k+1])))*q_test[k]*dx
                        vert_vel_term = w_upper_term - w_lower_term

                        f += div_hu_term + vert_vel_term

                    prob = NonlinearVariationalProblem(f, self.q_mixed_n_layers)
                    if self.n_layers == 1:
                        prob = NonlinearVariationalProblem(f, self.q_0)
                    solver = NonlinearVariationalSolver(prob,
                                                        solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu', #'bjacobi', 'lu'
                                                               'pc_factor_mat_solver_package': "mumps",},)
                if self.n_layers == 1:
                    self.uv_av_1.assign(uv_2d)
                solver.solve()
                for k in range(self.n_layers):
                    if self.n_layers == 1:
                        getattr(self, 'q_'+str(k)).assign(self.q_0)
                    else:
                        getattr(self, 'q_'+str(k)).assign(self.q_mixed_n_layers.split()[k])
                    if k == self.n_layers - 1:
                        getattr(self, 'q_'+str(k+1)).assign(0.)
                self.fields.q_2d.assign(self.q_0)

                # update depth-averaged velocity uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = 0.
                for k in range(self.n_layers):
                    l += inner(-self.dt/h_mid*grad(getattr(self, 'q_'+str(k))*h_layer[k+1]), uv_test)*dx
                    if k == self.n_layers - 1:
                        l += inner(uv_2d, uv_test)*dx
                solve(a == l, uv_2d)
                self.uv_2d_mid.assign(uv_2d)
                # update layer-averaged self.uv_av_{k+1}
                if self.n_layers >= 2:
                    sum_uv_av = 0.
                    for k in range(self.n_layers - 1):
                        a = inner(uv_tri, uv_test)*dx
                        l = inner(getattr(self, 'uv_av_'+str(k+1)) - self.dt*grad(getattr(self, 'q_'+str(k))), uv_test)*dx
                        solve(a == l, getattr(self, 'uv_av_'+str(k+1)))
                        sum_uv_av = sum_uv_av + alpha[k]*getattr(self, 'uv_av_'+str(k+1))
                    # update layer-averaged velocity of the free-surface layer
                    getattr(self, 'uv_av_'+str(self.n_layers)).project((uv_2d - sum_uv_av)/alpha[self.n_layers-1])
                # update layer-integrated vertical velocity w_{k}{k+1}
                a = w_tri*w_test*dx
                for k in range(self.n_layers):
                    if k == self.n_layers - 1:
                        l = (w_dic['z_'+str(k+1)] + self.dt*getattr(self, 'q_'+str(k))/(0.5*h_layer[k+1]))*w_test*dx
                    else:
                        l = (w_dic['z_'+str(k+1)] + self.dt*(getattr(self, 'q_'+str(k)) - getattr(self, 'q_'+str(k+1)))/(0.5*(h_layer[k+1] + h_layer[k+2])))*w_test*dx
                    solve(a == l, w_dic['z_'+str(k+1)])

                if self.simulation_time <= t_epsilon:
                    for k in range(self.n_layers):
                        timestepper_depth_integrated.F += self.dt/h_mid*inner(grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]), uta_test)*dx
                        if k == self.n_layers - 1:
                            timestepper_depth_integrated.F += self.dt/h_mid*inner(getattr(self, 'q_'+str(0))*grad(z_dic['z_'+str(0)]) - 
                                                                                  getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)]), uta_test)*dx
                    prob_n_layers_int = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                    solver_n_layers_int = NonlinearVariationalSolver(prob_n_layers_int,
                                                                     solver_parameters=solver_parameters)
                    if self.n_layers >= 2:
                        solver_dic = {}
                        for k in range(self.n_layers - 1):
                            timestepper_dic['layer_'+str(k+1)].F += self.dt/h_layer[k+1]*inner(grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]) +
                                                         getattr(self, 'q_'+str(k))*grad(z_dic['z_'+str(k)]) - getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)]), uv_test)*dx
                            prob_n_layer_dif_k = NonlinearVariationalProblem(timestepper_dic['layer_'+str(k+1)].F, getattr(self, 'uv_av_'+str(k+1)))
                            solver_dic['layer_'+str(k+1)] = NonlinearVariationalSolver(prob_n_layer_dif_k,
                                                                          solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)

                # update water level elev_2d
                update_water_level = not True
                solving_free_surface_eq = True
                if update_water_level:
                    if not solving_free_surface_eq:
                        solver_n_layers_int.solve()
                        #for k in range(self.n_layers - 1):
                        #    solver_dic['layer_'+str(k+1)].solve()
                        uv_2d.assign(self.uv_2d_mid)
                    else:
                        timestepper_free_surface.advance(self.simulation_time, update_forcings)
                        self.fields.elev_2d.assign(self.elev_2d_mid)

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()
                    for k in range(self.n_layers - 1):
                        self.uv_2d_dg.project(getattr(self, 'uv_av_'+str(k+1)))
                        self.uv_2d_dg.sub(1).assign(0.)
                        getattr(self, 'uv_av_'+str(k+1)).project(self.uv_2d_dg)









            # Move to next time step
            self.simulation_time += self.dt
            self.iteration += 1

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= self.next_export_t - t_epsilon:
                self.i_export += 1
                self.next_export_t += self.options.simulation_export_time

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                # exporter with wetting-drying handle
                if self.options.use_wetting_and_drying:
                    self.solution_2d_tmp.assign(self.fields.solution_2d)
                    H = self.bathymetry_dg.dat.data + elev_2d.dat.data
                    ind = np.where(H[:] <= 0.)[0]
                    elev_2d.dat.data[ind] = 1E-6 - self.bathymetry_dg.dat.data[ind]
                self.export()
                if self.options.use_wetting_and_drying:
                    self.fields.solution_2d.assign(self.solution_2d_tmp)

                if export_func is not None:
                    export_func()

                if hydrostatic_solver_2d:
                    print_output('Adopting 2d hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif hydrostatic_solver_3d:
                    print_output('Adopting 3d hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif conventional_3d_NH_solver:
                    print_output('Adopting 3d non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif one_layer_NH_solver:
                    print_output('Adopting one-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif reduced_two_layer_NH_solver:
                    print_output('Adopting reduced two-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif coupled_two_layer_NH_solver:
                    print_output('Adopting coupled two-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif coupled_three_layer_NH_solver:
                    print_output('Adopting coupled three-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                else:
                    print_output('Adopting {nlayer:}-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(nlayer=self.n_layers, degree=self.options.polynomial_degree, element=self.options.element_family))
