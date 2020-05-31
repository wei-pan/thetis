"""
Module for 2D depth averaged solver in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_nh
from . import granular_cf
from .. import timeintegrator
from .. import rungekutta
from .. import implicitexplicit
from .. import coupled_timeintegrator_2d
from .. import tracer_eq_2d
import weakref
import time as time_mod
from mpi4py import MPI
from .. import exporter
from ..field_defs import field_metadata
from ..options import ModelOptions2d
from .. import callback
from ..log import *
from collections import OrderedDict
from . import limiter_nh as limiter


class FlowSolver(FrozenClass):
    """
    Main object for 2D depth averaged solver in conservative form

    **Example**

    Create mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create solver object and set some options

    .. code-block:: python

        solver_obj = solver2d_cf.FlowSolver(mesh2d, bathymetry_2d)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'CrankNicolson'
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, options=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: :class:`Function`
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions2d` instance
        """
        self._initialized = False
        self.mesh2d = mesh2d
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.normal_2d = FacetNormal(self.mesh2d)
        self.boundary_markers = self.mesh2d.exterior_facets.unique_markers

        self.dt = None
        """Time step"""

        self.options = ModelOptions2d()
        """
        Dictionary of all options. A :class:`.ModelOptions2d` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

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

        self.fields.bathymetry_2d = bathymetry_2d

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self.bnd_functions = {'shallow_water': {}, 'momentum': {}, 'tracer': {}, 'landslide_motion': {}}

        self._isfrozen = True

    def compute_time_step(self, u_scale=Constant(0.0)):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :kwarg u_scale: User provided maximum advective velocity scale
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
        return solution

    def set_time_step(self, alpha=0.05):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions2d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions2d.timestep`.

        :kwarg float alpha: CFL number scaling factor
        """
        automatic_timestep = (hasattr(self.options.timestepper_options, 'use_automatic_timestep') and
                              self.options.timestepper_options.use_automatic_timestep)
        # TODO revisit math alpha is OBSOLETE
        if automatic_timestep:
            mesh2d_dt = self.compute_time_step(u_scale=self.options.horizontal_velocity_scale)
            dt = self.options.cfl_2d*alpha*float(mesh2d_dt.dat.data.min())
            dt = self.comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        else:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            self.dt = self.options.timestep
        if self.comm.rank == 0:
            print_output('dt = {:}'.format(self.dt))
            sys.stdout.flush()

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # 2D function spaces
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, 'DG', 0, name='P0_2d')
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 2, name='P2_2d')
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')

        # function space w.r.t element family
        if self.options.element_family == 'dg-dg':
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d', vector=True)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])
        self.function_spaces.V_ls = MixedFunctionSpace([self.function_spaces.H_2d, self.function_spaces.H_2d, self.function_spaces.H_2d])

        self._isfrozen = True

    def create_functions(self):
        """
        Creates extra functions
        """
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.fields.uv_2d, self.fields.elev_2d = self.fields.solution_2d.split()
        self.solution_old = Function(self.function_spaces.V_2d)
        self.solution_mid = Function(self.function_spaces.V_2d)

        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)
        self.bathymetry_old = Function(self.function_spaces.H_2d).assign(self.bathymetry_dg)
        self.bathymetry_init = Function(self.function_spaces.H_2d).assign(self.bathymetry_dg)
        self.elev_init = Function(self.function_spaces.H_2d)
        self.fields.q_2d = Function(self.function_spaces.P2_2d)
        self.uv_2d_old = Function(self.function_spaces.U_2d)
        self.w_surface = Function(self.function_spaces.H_2d)

        # granular flow
        if self.options.flow_is_granular:
            self.fields.solution_ls = Function(self.function_spaces.V_ls, name='solution_ls')
            self.fields.h_ls = self.fields.solution_ls.split()[0]
            self.solution_ls_old = Function(self.function_spaces.V_ls)
            self.solution_ls_mid = Function(self.function_spaces.V_ls)
            self.bathymetry_ls = Function(self.function_spaces.P1_2d)
            self.phi_i = Function(self.function_spaces.P1_2d).assign(self.options.phi_i)
            self.phi_b = Function(self.function_spaces.P1_2d).assign(self.options.phi_b)
            self.kap = Function(self.function_spaces.P1_2d)
            self.uv_div_ls = Function(self.function_spaces.P1_2d)
            self.strain_rate_ls = Function(self.function_spaces.P1_2d)
            self.source_ls = Function(self.function_spaces.V_ls)
            self.fields.slide_source = Function(self.function_spaces.H_2d)
            self.p_f = Function(self.function_spaces.H_2d)

    def create_equations(self):
        """
        Creates shallow water equations
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.create_functions()
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)

        # ----- Equations
        self.eq_sw = shallowwater_nh.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options
        )
        if self.options.flow_is_granular:
            self.eq_ls = granular_cf.GranularEquations(
                self.fields.solution_ls.function_space(),
                self.bathymetry_ls,
                self.options
            )
      #  self.eq_free_surface = shallowwater_nh.FreeSurfaceEquation(
       #     TestFunction(self.function_spaces.H_2d),
        #    self.function_spaces.H_2d,
         #   self.function_spaces.U_2d,
          #  self.bathymetry_dg,
           # self.options)

        if self.options.use_wetting_and_drying:
            self.wd_modification = wetting_and_drying_modification(self.function_spaces.H_2d)

        self._isfrozen = True  # disallow creating new attributes

    def create_timestepper(self):
        """
        Creates time stepper instance
        """
        if not hasattr(self, 'eq_sw'):
            self.create_equations()

        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # ----- Time integrators
        self.fields_sw = {
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
            'uv': self.fields.solution_2d.sub(0),
            'eta': self.fields.solution_2d.sub(1),
            #'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_xstart, alpha = 10.),
            }
        if self.options.flow_is_granular:
            self.fields_ls = {
                'phi_i': self.phi_i,
                'phi_b': self.phi_b,
                #'kap': self.kap,
                'uv_div': self.uv_div_ls,
                'strain_rate': self.strain_rate_ls,
                'fluid_pressure': self.p_f,
                }
            self.fields_sw.update({'slide_source': self.fields.slide_source,})
        self.set_time_step()
        if self.options.timestepper_type == 'SSPRK33':
            self.timestepper = rungekutta.SSPRK33(self.eq_sw, self.fields.solution_2d,
                                                  self.fields_sw, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'CrankNicolson':
            self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                            self.fields_sw, self.dt,
                                                            bnd_conditions=self.bnd_functions['shallow_water'],
                                                            solver_parameters=self.options.timestepper_options.solver_parameters,
                                                            semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                            theta=self.options.timestepper_options.implicitness_theta)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))
        print_output('Using time integrator: {:}'.format(self.timestepper.__class__.__name__))
        self._isfrozen = True  # disallow creating new attributes

    def create_exporters(self):
        """
        Creates file exporters
        """
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        self._isfrozen = False
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

        self._isfrozen = True  # disallow creating new attributes

    def initialize(self):
        """
        Creates function spaces, equations, time stepper and exporters
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        if not hasattr(self, 'eq_sw'):
            self.create_equations()
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        if not hasattr(self, 'exporters'):
            self.create_exporters()
        self._initialized = True

    def assign_initial_conditions(self, elev_2d=None, uv_2d=None, h_ls=None, uv_ls=None):
        """
        Assigns initial conditions

        :kwarg elev_2d: Initial condition for water elevation
        :type elev_2d: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_2d: Initial condition for depth averaged velocity
        :type uv_2d: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()

        if elev_2d is not None:
            self.fields.elev_2d.project(elev_2d)
        # prevent non-negative initial water depth
        h_2d = self.fields.elev_2d.dat.data + self.bathymetry_dg.dat.data
        ind = np.where(h_2d[:] <= self.options.wetting_and_drying_threshold)[0]
        self.fields.elev_2d.dat.data[ind] = -self.bathymetry_dg.dat.data[ind]
        self.elev_init.assign(self.fields.elev_2d)
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)

        if self.options.flow_is_granular:
            if h_ls is not None:
                self.fields.solution_ls.sub(0).project(h_ls)
            h_ls = self.fields.solution_ls.sub(0).dat.data[:]
            ind_ls = np.where(h_ls[:] <= self.options.wetting_and_drying_threshold)[0]
            h_ls[ind_ls] = 0.
            if uv_ls is not None:
                self.fields.solution_ls.sub(1).project(self.fields.solution_ls.sub(0)*uv_ls[0])
                self.fields.solution_ls.sub(2).project(self.fields.solution_ls.sub(0)*uv_ls[1])

        self.timestepper.initialize(self.fields.solution_2d)

    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg string eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export')
        for e in self.exporters.values():
            e.export()

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces :meth:`.assign_initial_conditions` in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format. The required
        state variables are: elev_2d, uv_2d

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
            self.initialize()
        if outputdir is None:
            outputdir = self.options.output_directory
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        self.assign_initial_conditions()

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

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        if self.options.tracer_only:
            norm_q = norm(self.fields.tracer_2d)

            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'tracer norm: {q:10.4f} {cpu:5.2f}')

            print_output(line.format(iexp=self.i_export, i=self.iteration,
                                     t=self.simulation_time, q=norm_q,
                                     cpu=cputime))
        else:
            lx = self.mesh2d.coordinates.sub(0).dat.data.max() - self.mesh2d.coordinates.sub(0).dat.data.min()
            ly = self.mesh2d.coordinates.sub(1).dat.data.max() - self.mesh2d.coordinates.sub(1).dat.data.min()
            norm_eta = norm(self.fields.elev_2d) / sqrt(lx * ly)
            norm_uv = norm(self.fields.uv_2d) / sqrt(lx * ly)
            if self.options.flow_is_granular:
                norm_hs = norm(self.fields.h_ls) / sqrt(lx * ly)
            else:
                norm_hs = 0

            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} uv norm: {u:10.4f} hs norm: {h:10.4f} {cpu:5.2f}')
            print_output(line.format(iexp=self.i_export, i=self.iteration,
                                     t=self.simulation_time, e=norm_eta,
                                     u=norm_uv, h=norm_hs, cpu=cputime))
        sys.stdout.flush()

    def iterate(self, update_forcings=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.
        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        # TODO I think export function is obsolete as callbacks are in place
        if not self._initialized:
            self.initialize()

        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.simulation_export_time

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c)

        if self.options.check_tracer_conservation:
            c = callback.TracerMassConservation2DCallback('tracer_2d',
                                                          self,
                                                          export_to_hdf5=dump_hdf5,
                                                          append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self.options.check_tracer_overshoot:
            c = callback.TracerOvershootCallBack('tracer_2d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters and isinstance(self.fields.bathymetry_2d, Function):
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        if self.options.flow_is_granular:
            # solver for granular landslide motion
            a_ls = self.eq_ls.mass_term(self.eq_ls.trial)
            l_ls = (self.eq_ls.mass_term(self.fields.solution_ls) + Constant(self.dt)*
                    self.eq_ls.residual('all', self.fields.solution_ls, self.fields.solution_ls,
                                        self.fields_ls, self.fields_ls, self.bnd_functions['landslide_motion'])
                   )
            prob_ls = LinearVariationalProblem(a_ls, l_ls, self.solution_ls_mid)
            solver_ls = LinearVariationalSolver(prob_ls, solver_parameters=self.options.timestepper_options.solver_parameters)
            # solver for div(velocity)
            h_ls = self.fields.solution_ls.sub(0)
            hu_ls = self.fields.solution_ls.sub(1)
            hv_ls = self.fields.solution_ls.sub(2)
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
            tri_pf = TrialFunction(self.p_f.function_space())
            test_pf = TestFunction(self.p_f.function_space())
            a_pf = tri_pf*test_pf*dx
            l_pf = self.options.rho_fluid*physical_constants['g_grav']*(self.bathymetry_dg + self.fields.elev_2d)*test_pf*dx
            prob_pf = LinearVariationalProblem(a_pf, l_pf, self.p_f)
            solver_pf = LinearVariationalSolver(prob_pf)

        # solvers for non-hydrostatic pressure
        solve_nh_pressure = not True # TODO set in `options`
        if solve_nh_pressure:
            # Poisson solver
            theta = 1#0.5
            par = 0.5 # approximation parameter for NH terms
            d_2d = self.bathymetry_dg
            h_2d = d_2d + self.fields.elev_2d
            alpha = self.options.depth_wd_interface
            h_mid = 2 * alpha**2 / (2 * alpha + abs(h_2d)) + 0.5 * (abs(h_2d) + h_2d)
            A = theta*grad(self.fields.elev_2d - d_2d)/h_mid# + (1. - theta)*grad(self.solution_old.sub(0) - d_2d)/h_old
            B = div(A) - 2./(par*h_mid*h_mid)
            C = (div(self.fields.uv_2d) + (self.w_surface + inner(2.*self.fields.uv_2d - self.uv_2d_old, grad(d_2d)))/h_mid)/(par*self.dt)
            if self.options.flow_is_granular:
                C = (div(self.fields.uv_2d) + (self.w_surface + inner(2.*self.fields.uv_2d - self.uv_2d_old, grad(d_2d)) - self.fields.slide_source)/h_mid)/(par*self.dt)
            # weak forms
            q_2d = self.fields.q_2d
            q_test = TestFunction(self.fields.q_2d.function_space())
            f_q = (-dot(grad(q_2d), grad(q_test)) + B*q_2d*q_test)*dx - C*q_test*dx - q_2d*div(A*q_test)*dx
            # boundary conditions
            for bnd_marker in self.boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                #q_open_bc = self.q_bnd.assign(0.)
                if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                    # Neumann boundary condition => inner(grad(q), normal)=0.
                    f_q += (q_2d*inner(A, self.normal_2d))*q_test*ds_bnd
            prob_q = NonlinearVariationalProblem(f_q, self.fields.q_2d)
            solver_q = NonlinearVariationalSolver(prob_q,
                                            solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu', #'bjacobi', 'lu'
                                                               },
                                            options_prefix='poisson_solver')
            # solver to update velocities
            # update uv_2d
            uv_tri = TrialFunction(self.function_spaces.U_2d)
            uv_test = TestFunction(self.function_spaces.U_2d)
            a_u = inner(uv_tri, uv_test)*dx
            l_u = inner(self.fields.uv_2d - par*self.dt*(grad(q_2d) + A*q_2d), uv_test)*dx
            prob_u = LinearVariationalProblem(a_u, l_u, self.fields.uv_2d)
            solver_u = LinearVariationalSolver(prob_u)
            # update w_surf
            w_tri = TrialFunction(self.function_spaces.H_2d)
            w_test = TestFunction(self.function_spaces.H_2d)
            a_w = w_tri*w_test*dx
            l_w = (self.w_surface + 2.*self.dt*q_2d/h_mid + inner(self.fields.uv_2d - self.uv_2d_old, grad(d_2d)))*w_test*dx
            prob_w = LinearVariationalProblem(a_w, l_w, self.w_surface)
            solver_w = LinearVariationalSolver(prob_w)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            self.bathymetry_old.assign(self.bathymetry_dg)
            self.uv_2d_old.assign(self.fields.uv_2d)
            # original line: self.timestepper.advance(self.simulation_time, update_forcings)

            # facilitate wetting and drying treatment at each stage
            use_ssprk22 = True # i.e. compatible with nh wave model
            if self.options.timestepper_type == 'SSPRK33':
                n_stages = self.timestepper.n_stages
                coeff = [[0., 1.], [3./4., 1./4.], [1./3., 2./3.]]
            if use_ssprk22:
                n_stages = 2
                coeff = [[0., 1.], [1./2., 1./2.]]
            if self.options.flow_is_granular:
                self.solution_ls_old.assign(self.fields.solution_ls)
                for i_stage in range(n_stages):
                    #self.timestepper.solve_stage(i_stage, self.simulation_time, update_forcings)
                    solver_ls.solve()
                    self.fields.solution_ls.assign(coeff[i_stage][0]*self.solution_ls_old + coeff[i_stage][1]*self.solution_ls_mid)
                    if self.options.use_wetting_and_drying:
                        limiter_start_time = 0.
                        use_limiter = self.options.use_wd_limiter and self.simulation_time >= limiter_start_time
                        self.wd_modification.apply(self.fields.solution_ls, self.options.wetting_and_drying_threshold, use_limiter)
                    solver_div.solve()
                    solver_sr.solve()

            if not self.options.no_wave_flow:

                self.solution_old.assign(self.fields.solution_2d)

                # update landslide motion source
                if self.options.flow_is_granular:
                    # NOTE `self.bathymetry_init` initialised does not vary with time
                    self.bathymetry_dg.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]
                    # detect before hitting water
                    h_2d = self.elev_init.dat.data + self.bathymetry_dg.dat.data
                    ind = np.where(h_2d[:] <= self.options.wetting_and_drying_threshold)[0]
                    self.fields.elev_2d.dat.data[ind] = -self.bathymetry_dg.dat.data[ind]
                    if self.simulation_time >= 0.24:
                        self.fields.slide_source.dat.data[:] = (self.fields.solution_ls.sub(0).dat.data[:] - self.solution_ls_old.sub(0).dat.data[:])/self.dt
                   # if self.simulation_time >= 0.:
                    #    solver_pf.solve()

                if self.options.timestepper_type == 'CrankNicolson':
                    self.timestepper.advance(self.simulation_time, update_forcings)
                    if solve_nh_pressure:
                       # print('ssssssssssssssssssssssssssss')
                        solver_q.solve()
                       # print(self.fields.q_2d.dat.data)
                        solver_u.solve()
                        solver_w.solve()
                elif self.options.timestepper_type == 'SSPRK33':
                    for i_stage in range(n_stages):
                        #self.timestepper.solve_stage(i_stage, self.simulation_time, update_forcings)
                        solver_sw.solve()
                        self.fields.solution_2d.assign(coeff[i_stage][0]*self.solution_old + coeff[i_stage][1]*self.solution_mid)
                        if self.options.use_wetting_and_drying:
                            limiter_start_time = 0.
                            use_limiter = self.options.use_wd_limiter and self.simulation_time >= limiter_start_time
                            self.wd_modification.apply(self.fields.solution_2d, self.options.wetting_and_drying_threshold, 
                                                       use_limiter, use_eta_solution=True, bathymetry=self.bathymetry_dg)

                       # E = self.options.wetting_and_drying_threshold
                       # ind = np.where(self.fields.solution_2d.sub(0).dat.data[:] <= E)[0]
                       # self.fields.solution_2d.sub(0).dat.data[ind] = E
                       # self.fields.solution_2d.sub(1).dat.data[ind] = 0.
                       # self.fields.solution_2d.sub(2).dat.data[ind] = 0.

                    if solve_nh_pressure:
                        u_2d = conditional(h_mid <= 0, zero(self.fields.hu_2d.ufl_shape), self.fields.hu_2d/h_mid)
                        v_2d = conditional(h_mid <= 0, zero(self.fields.hv_2d.ufl_shape), self.fields.hv_2d/h_mid)
                        self.fields.uv_2d.interpolate(as_vector((u_2d, v_2d)))
                       # print('ssssssssssssssssssssssssssss')
                        solver_q.solve()
                       # print(self.fields.q_2d.dat.data)
                        solver_u.solve()
                        self.fields.solution_2d.sub(1).interpolate(h_mid*self.fields.uv_2d.sub(0))
                        self.fields.solution_2d.sub(2).interpolate(h_mid*self.fields.uv_2d.sub(1))
                        solver_w.solve()

            # Move to next time step
            self.iteration += 1
            internal_iteration += 1
            self.simulation_time = initial_simulation_time + internal_iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                self.i_export += 1
                next_export_t += self.options.simulation_export_time

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                # exporter with wetting-drying handle
                if self.options.use_wetting_and_drying:
                    self.solution_mid.assign(self.fields.solution_2d)
                    H = self.bathymetry_dg.dat.data + self.fields.elev_2d.dat.data
                    ind = np.where(H[:] <= 0.)[0]
                    self.fields.elev_2d.dat.data[ind] = 1E-6 - self.bathymetry_dg.dat.data[ind]
                self.export()
                if self.options.use_wetting_and_drying:
                    self.fields.solution_2d.assign(self.solution_mid)
                if export_func is not None:
                    export_func()
