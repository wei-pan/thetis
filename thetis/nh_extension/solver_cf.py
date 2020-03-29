"""
Module for 2D depth averaged solver in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_cf
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

        solver_obj = solver_cf.FlowSolver(mesh2d, bathymetry_2d)
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

        self.bnd_functions = {'shallow_water': {}, 'momentum': {}, 'tracer': {}}

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
        # ----- function spaces: elev in H, hu/hv in U, mixed is W
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, 'DG', 0, name='P0_2d')
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 2, name='P2_2d')
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')

        # function space w.r.t element family
        if self.options.element_family == 'dg-dg':
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.H_2d, self.function_spaces.U_2d, self.function_spaces.U_2d])

        self._isfrozen = True

    def create_functions(self):
        """
        Creates extra functions
        """
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.fields.h_2d, self.fields.hu_2d, self.fields.hv_2d = self.fields.solution_2d.split()

        self.solution_old = Function(self.function_spaces.V_2d)
        self.solution_wd = Function(self.function_spaces.V_2d)

        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        self.source = Function(self.function_spaces.V_2d, name='source')

        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)
        self.fields.elev_2d = Function(self.function_spaces.H_2d)

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
        self.eq_sw = shallowwater_cf.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.fields.bathymetry_2d,
            self.options
        )
      #  self.eq_free_surface = shallowwater_cf.FreeSurfaceEquation(
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
        self.field_dic = {
            'source': self.source,}
        if self.options.landslide is True:
            self.field_dic.update({'slide_source': self.fields.slide_source,})
        fields = self.field_dic
        self.set_time_step()
        if self.options.timestepper_type == 'SSPRK33':
            self.timestepper = rungekutta.SSPRK33(self.eq_sw, self.fields.solution_2d,
                                                  fields, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'CrankNicolson':
            self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                            fields, self.dt,
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

    def assign_initial_conditions(self, h_2d=None, hu_2d=None, hv_2d=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv: Initial condition for depth averaged velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()

        if h_2d is not None:
            self.fields.h_2d.project(h_2d)
        if hu_2d is not None:
            self.fields.hu_2d.project(hu_2d)
        if hv_2d is not None:
            self.fields.hv_2d.project(hv_2d)

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
        state variables are: h_2d, hu_2d, hv_2d

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
        state_fields = ['h_2d', 'hu_2d', 'hv_2d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['h_2d'].load(i_export, self.fields.h_2d)
        e.exporters['hu_2d'].load(i_export, self.fields.hu_2d)
        e.exporters['hv_2d'].load(i_export, self.fields.hv_2d)
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
            norm_h = norm(self.fields.solution_2d.split()[0]) / sqrt(lx * ly)
            norm_hu = norm(self.fields.solution_2d.split()[1]) / sqrt(lx * ly)
            norm_hv = norm(self.fields.solution_2d.split()[2]) / sqrt(lx * ly)

            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'h norm: {e:10.4f} hu norm: {hu:10.4f} hv norm: {hv:10.4f} {cpu:5.2f}')
            print_output(line.format(iexp=self.i_export, i=self.iteration,
                                     t=self.simulation_time, e=norm_h,
                                     hu=norm_hu, hv=norm_hv, cpu=cputime))
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
            self.fields.elev_2d.dat.data[:] = self.fields.h_2d.dat.data[:] - self.bathymetry_dg.dat.data[:]
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters and isinstance(self.fields.bathymetry_2d, Function):
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        a_rk = self.eq_sw.mass_term(self.eq_sw.trial)
        l_rk = self.eq_sw.mass_term(self.fields.solution_2d) + Constant(self.dt)*self.eq_sw.residual('all',
                                                         self.fields.solution_2d, self.fields.solution_2d,
                                                         self.field_dic, self.field_dic,
                                                         self.bnd_functions['shallow_water'])
        prob = LinearVariationalProblem(a_rk, l_rk, self.solution_wd)
        solver = LinearVariationalSolver(prob, solver_parameters=self.options.timestepper_options.solver_parameters)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            self.solution_old.assign(self.fields.solution_2d)
            # original line: self.timestepper.advance(self.simulation_time, update_forcings)
            if self.options.timestepper_type == 'CrankNicolson':
                self.timestepper.advance(self.simulation_time, update_forcings)
                assert (not self.options.use_wetting_and_drying)
            elif self.options.timestepper_type == 'SSPRK33':
                # facilitate wetting and drying treatment at each stage
                n_stages = self.timestepper.n_stages
                coeff = [[0., 1.], [3./4., 1./4.], [1./3., 2./3.]]
                for i_stage in range(n_stages):
                    #self.timestepper.solve_stage(i_stage, self.simulation_time, update_forcings)
                    solver.solve()
                    self.fields.solution_2d.assign(coeff[i_stage][0]*self.solution_old + coeff[i_stage][1]*self.solution_wd)
                    if self.options.use_wetting_and_drying:
                        limiter_start_time = 0.
                        use_limiter = self.options.use_wd_limiter and self.simulation_time >= limiter_start_time
                        self.wd_modification.apply(self.fields.solution_2d, 
                                                   self.options.wetting_and_drying_threshold, 
                                                   use_limiter)
                       # E = self.options.wetting_and_drying_threshold
                       # ind = np.where(self.fields.solution_2d.sub(0).dat.data[:] <= E)[0]
                       # self.fields.solution_2d.sub(0).dat.data[ind] = E
                       # self.fields.solution_2d.sub(1).dat.data[ind] = 0.
                       # self.fields.solution_2d.sub(2).dat.data[ind] = 0.

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

                self.fields.elev_2d.dat.data[:] = self.fields.h_2d.dat.data[:] - self.bathymetry_dg.dat.data[:]
                self.export()
                if export_func is not None:
                    export_func()
