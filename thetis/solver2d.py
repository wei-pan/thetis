"""
Module for 2D depth averaged solver
"""
from __future__ import absolute_import
from .utility import *
from . import shallowwater_eq
from . import timeintegrator
from . import rungekutta
from . import implicitexplicit
import time as time_mod
from mpi4py import MPI
from . import exporter
from .field_defs import field_metadata
from .options import ModelOptions
from . import callback
from .log import *
from firedrake.output import is_cg

class FlowSolver2d(FrozenClass):
    """
    Main object for 2D depth averaged solver

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

        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.order = 1
        options.timestepper_type = 'cranknicolson'
        options.t_export = 50.0
        options.t_end = 3600.
        options.dt = 25.0

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
        :type options: :class:`.ModelOptions` instance
        """
        self._initialized = False
        self.mesh2d = mesh2d
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len

        self.dt = None
        """Time step"""
        
        self.options = ModelOptions()
        """
        Dictionary of all options. A :class:`.ModelOptions` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.t_export

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

        self.bnd_functions = {'shallow_water': {}}

        self._isfrozen = False

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
 
    def adaptive_time_step(self, solution): # try AdaptiveTimestepping, Wei, 26/05/2017
        """
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Velocity scalar :math:`U = \sqrt{g H} + U` where
        :math:`U` is abs(velocity:sqrt(u^2 + v^2)).

        :arg solution: Computational results at each step, used to compute dt
        """
        g = physical_constants['g_grav'].dat.data[0]   
        min_cell_length = Function(FunctionSpace(self.mesh2d, 'DG', 0)).interpolate(MinCellEdgeLength(self.mesh2d))

        uv_2d, elev_2d = solution.split()
        if self.options.element_family == 'rt-dg':
            native_space = uv_2d.function_space()
            visu_space = exporter.get_visu_space(native_space)
            tmp_proj_func = visu_space.get_work_function()
            Interpolator(uv_2d, tmp_proj_func).interpolate()
            visu_space.restore_work_function(tmp_proj_func)
        else:
            tmp_proj_func = uv_2d 
        uv_mod = tmp_proj_func

        fs_tmp = FunctionSpace(self.mesh2d, "DG", self.options.order)
        
        H_mod = Function(fs_tmp).assign(self.fields.bathymetry_2d + elev_2d) 
        H_mod.dat.data[H_mod.dat.data < self.options.wd_mindep] = self.options.wd_mindep
        u_mod = Function(fs_tmp).assign(uv_mod.sub(0)) 
        v_mod = Function(fs_tmp).assign(uv_mod.sub(1)) 

        cell_time = Function(FunctionSpace(self.mesh2d, 'DG', 0))

        min_time_wave_speed_kernel_2d = """ const double g=%(gravity)s; float max=-10000000, wave_speed=0;
        for(int i=0;i<vert_u_cell.dofs;i++){
            wave_speed=sqrt(pow(vert_u_cell[i][0],2)+pow(vert_v_cell[i][0],2))+sqrt(g*vert_h_cell[i][0]);         
            max=fmax(wave_speed,max);
        }
        cell_wave_time[0][0]=cell_lengths[0][0]/max;
        """ % {"gravity": g}  

        par_loop(min_time_wave_speed_kernel_2d, dx, {"cell_wave_time": (cell_time, RW),
                                                     "vert_h_cell": (H_mod, READ),
                                                     "vert_u_cell": (u_mod, READ),
                                                     "vert_v_cell": (v_mod, READ),
                                                     "cell_lengths": (min_cell_length, READ)})  
        return cell_time

    def set_time_step(self, solution, alpha=1.0): # change default alpha to 1.0 from 0.05, Wei
        """
        Sets the model the model time step

        Uses ``options.dt`` if set, otherwise sets the maximum time step
        allowed by the CFL condition (see :meth:`.compute_time_step`).

        :kwarg float alpha: CFL number scaling factor
        """
        # TODO revisit math alpha is OBSOLETE
        #mesh2d_dt = self.compute_time_step(u_scale=self.options.u_advection)
        mesh2d_dt = self.adaptive_time_step(solution) # Wei
        dt_max = self.comm.allreduce(self.options.cfl_2d*alpha*float(mesh2d_dt.dat.data.min()), op=MPI.MIN)
        cfl_2d = self.options.cfl_2d
        if self.timestepper is not None:
            cfl_2d = self.timestepper.cfl_coeff     
        if self.options.dt is None:
            print_output('Time step is not set initially, so varing adaptively:')
            if cfl_2d == np.inf:
                print_output('CFL is unconditionally stable')
                dt_max = self.options.cfl_2d*alpha*float(self.compute_time_step(u_scale=self.options.u_advection).dat.data.min())
                dt_max = 2.0*self.comm.allreduce(dt_max, op=MPI.MIN)
            self.dt = dt_max
        else:
            if cfl_2d == np.inf:
                print_output('CFL is unconditionally stable')
                self.dt = self.options.dt
            elif dt_max <= self.options.dt:
                self.dt = dt_max 
            else:
                self.dt = self.options.dt
        if self.comm.rank == 0:
            print_output('dt = {:}'.format(self.dt))
            sys.stdout.flush()
        return self.dt

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0_2d = FunctionSpace(self.mesh2d, 'DG', 0)
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1)
        self.function_spaces.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        self.function_spaces.P1DGv_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            #self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order) # Wei, for use of slopemodification & better results
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        elif self.options.element_family == 'dg-cg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'CG', self.options.order+1)
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        self._isfrozen = True

    def create_equations(self):
        """
        Creates shallow water equations
        """
        if self.options.wd_alpha is not None:
            self.options.wd_alpha = self.get_alpha(self.fields.bathymetry_2d.dat.data.min()) # Wei, 16/05/2017

        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.solution_2d_old = Function(self.function_spaces.V_2d, name='solution_2d_old') # added for exporter of adaptive time stepping, Wei
        self.exchange_tmp = Function(self.function_spaces.V_2d, name='exchange_tmp') # Wei
        self.export_tmp = Function(self.function_spaces.V_2d, name='export_tmp') # Wei
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)

        # ----- Equations
        self.eq_sw = shallowwater_eq.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.fields.bathymetry_2d,
            nonlin=self.options.nonlin,
            wd_alpha=self.options.wd_alpha, 
            wd_mindep=self.options.wd_mindep,
            include_grad_div_viscosity_term=self.options.include_grad_div_viscosity_term,
            include_grad_depth_viscosity_term=self.options.include_grad_depth_viscosity_term
        )
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self._isfrozen = True  # disallow creating new attributes

    def create_timestepper(self):
        """
        Creates time stepper instance
        """
        if not hasattr(self, 'eq_sw'):
            self.create_equations()

        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.outputdir), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # ----- Time integrators
        fields = {
            'linear_drag': self.options.linear_drag,
            'quadratic_drag': self.options.quadratic_drag,
            'mu_manning': self.options.mu_manning,
            'viscosity_h': self.options.h_viscosity,
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': self.options.uv_source_2d,
            'elev_source': self.options.elev_source_2d, }

        self.timestepper = None # for set_time_step, Wei

        self.set_time_step(self.fields.solution_2d) # set time step initially, Wei

        if self.options.timestepper_type.lower() == 'ssprk33':
            self.timestepper = rungekutta.SSPRK33(self.eq_sw, self.fields.solution_2d,
                                                  fields, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'ssprk33semi':
            self.timestepper = rungekutta.SSPRK33SemiImplicit(self.eq_sw, self.fields.solution_2d,
                                                              fields, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=self.options.solver_parameters_sw,
                                                              semi_implicit=self.options.use_linearized_semi_implicit_2d,
                                                              theta=self.options.shallow_water_theta)

        elif self.options.timestepper_type.lower() == 'forwardeuler':
            self.timestepper = timeintegrator.ForwardEuler(self.eq_sw, self.fields.solution_2d,
                                                           fields, self.dt,
                                                           bnd_conditions=self.bnd_functions['shallow_water'],
                                                           solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'backwardeuler':
            self.timestepper = rungekutta.BackwardEuler(self.eq_sw, self.fields.solution_2d,
                                                        fields, self.dt,
                                                        bnd_conditions=self.bnd_functions['shallow_water'],
                                                        solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'cranknicolson':
            self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                            fields, self.dt,
                                                            bnd_conditions=self.bnd_functions['shallow_water'],
                                                            solver_parameters=self.options.solver_parameters_sw,
                                                            semi_implicit=self.options.use_linearized_semi_implicit_2d,
                                                            theta=self.options.shallow_water_theta)
        elif self.options.timestepper_type.lower() == 'dirk22':
            self.timestepper = rungekutta.CrankNicolsonRK(self.eq_sw, self.fields.solution_2d,
                                                          fields, self.dt,
                                                          bnd_conditions=self.bnd_functions['shallow_water'],
                                                          solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'dirk33':
            self.timestepper = rungekutta.DIRK33(self.eq_sw, self.fields.solution_2d,
                                                 fields, self.dt,
                                                 bnd_conditions=self.bnd_functions['shallow_water'],
                                                 solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'steadystate':
            self.timestepper = timeintegrator.SteadyState(self.eq_sw, self.fields.solution_2d,
                                                          fields, self.dt,
                                                          bnd_conditions=self.bnd_functions['shallow_water'],
                                                          solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'pressureprojectionpicard':

            u_test = TestFunction(self.function_spaces.U_2d)
            self.eq_mom = shallowwater_eq.ShallowWaterMomentumEquation(
                u_test, self.function_spaces.H_2d, self.function_spaces.U_2d,
                self.fields.bathymetry_2d,
                nonlin=self.options.nonlin,
                include_grad_div_viscosity_term=self.options.include_grad_div_viscosity_term,
                include_grad_depth_viscosity_term=self.options.include_grad_depth_viscosity_term
            )
            self.eq_mom.bnd_functions = self.bnd_functions['shallow_water']
            self.timestepper = timeintegrator.PressureProjectionPicard(self.eq_sw, self.eq_mom, self.fields.solution_2d,
                                                                       fields, self.dt,
                                                                       bnd_conditions=self.bnd_functions['shallow_water'],
                                                                       solver_parameters=self.options.solver_parameters_sw,
                                                                       solver_parameters_mom=self.options.solver_parameters_sw_momentum,
                                                                       semi_implicit=self.options.use_linearized_semi_implicit_2d,
                                                                       theta=self.options.shallow_water_theta)

        elif self.options.timestepper_type.lower() == 'sspimex':
            # TODO meaningful solver params
            sp_impl = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'multiplicative',
            }
            sp_expl = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'multiplicative',
            }
            self.timestepper = implicitexplicit.IMEXLPUM2(self.eq_sw, self.fields.solution_2d, fields, self.dt,
                                                          bnd_conditions=self.bnd_functions['shallow_water'],
                                                          solver_parameters=sp_expl,
                                                          solver_parameters_dirk=sp_impl)
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
        # correct treatment of the split 2d functions
        uv_2d, elev_2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = elev_2d
        self.exporters = {}
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.outputdir,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            hdf5_dir = os.path.join(self.options.outputdir, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        self._isfrozen = True  # disallow creating new attributes

    def SlopeLimiter(self, solution): # Wei, 05/05/2017
        """
        Creates slope limiter for the prevention of shocks. This is from Kuzmin (2011)
        """ 
        uv_2d, elev_2d = solution.split()
        if self.options.element_family == 'rt-dg':
            native_space = uv_2d.function_space()
            visu_space = exporter.get_visu_space(native_space)
            tmp_proj_func = visu_space.get_work_function()
            Interpolator(uv_2d, tmp_proj_func).interpolate()
            visu_space.restore_work_function(tmp_proj_func)
        else:
            tmp_proj_func = uv_2d

        SL_eta = VertexBasedLimiter(self.function_spaces.H_2d)
        SL_eta.apply(solution.sub(1))
        
        func_tmp = Function(self.function_spaces.U_2d)               
        fs_u =FunctionSpace(self.mesh2d, 'DG', self.options.order)

        func_tmp = Function(fs_u) 
        #func_tmp.assign(solution.sub(0).sub(0)*(solution.sub(1) + self.fields.bathymetry_2d))
        func_tmp.assign(tmp_proj_func.sub(0)*(solution.sub(1) + self.fields.bathymetry_2d))
        SL_uv = VertexBasedLimiter(fs_u)
        SL_uv.apply(func_tmp)
        tmp_proj_func.sub(0).assign(func_tmp/(solution.sub(1) + self.fields.bathymetry_2d))

        func_tmp.assign(tmp_proj_func.sub(1)*(solution.sub(1) + self.fields.bathymetry_2d))
        SL_uv.apply(func_tmp)
        tmp_proj_func.sub(1).assign(func_tmp/(solution.sub(1) + self.fields.bathymetry_2d))

        uv_2d.project(tmp_proj_func)
        
        return solution	

    def SlopeModification(self, solution): # Wei, 05/05/2017
        """
        Creates slope modification for the prevention of non negative flows in some verticies of cells. This is from Ern et al (2011)
        """
        uv_2d, elev_2d = solution.split()
        if self.options.element_family == 'rt-dg':
            native_space = uv_2d.function_space()
            visu_space = exporter.get_visu_space(native_space)
            tmp_proj_func = visu_space.get_work_function()
            Interpolator(uv_2d, tmp_proj_func).interpolate()
            visu_space.restore_work_function(tmp_proj_func)
        else:
            tmp_proj_func = uv_2d 

        fs_tmp = FunctionSpace(self.mesh2d, "DG", self.options.order)

        H = Function(fs_tmp).assign(elev_2d + self.fields.bathymetry_2d)
        u = Function(fs_tmp).assign(tmp_proj_func.sub(0)) 
        v = Function(fs_tmp).assign(tmp_proj_func.sub(1))    

        func_eta_tmp = Function(fs_tmp)
        func_u_tmp = Function(fs_tmp)
        func_v_tmp = Function(fs_tmp)

        # add error if p>1 - in future make an option where we can project to p=1 from p>1
        if self.function_spaces.V_2d.ufl_element().degree() != 1:
            raise ValueError('Slope Modification technique cannot be used with DGp, p > 1')

        slope_modification_2d_kernel = """ double new_cell = 0; const double E=%(epsilon)s; int a=0, n1, n2, n3, a1=0, a3=0, flag1, flag2, flag3, deltau, deltav, npos;
        #define STEP(X) (((X) <= 0) ? 0 : 1)
        for(int i=0;i<vert_h_cell.dofs;i++){
            new_cell+=vert_h_cell[i][0];
        }
        new_cell=new_cell/vert_h_cell.dofs;
        for(int i=0;i<new_vert_h_cell.dofs;i++){
            if (vert_h_cell[i][0]>E){
                a=a+1;
            }
        }
        if (a==new_vert_h_cell.dofs){
            for(int i=0;i<new_vert_h_cell.dofs;i++){
                new_vert_h_cell[i][0]=vert_h_cell[i][0];
            }
        }
        if (new_cell<=E){
            for(int i=0;i<new_vert_h_cell.dofs;i++){
                new_vert_h_cell[i][0]=new_cell;
            }
        }
        if (new_cell>E){
            if (a<new_vert_h_cell.dofs){
                for(int i=1;i<new_vert_h_cell.dofs;i++){
                    if (vert_h_cell[0][0]>=vert_h_cell[i][0]){
                        a1=a1+1;
                    }
                }
                for(int i=0;i<new_vert_h_cell.dofs-1;i++){
                    if (vert_h_cell[2][0]>=vert_h_cell[i][0]){
                        a3=a3+1;
                    }
                }
                if (a1==2){
                    n3=0;
                    if (a3==2){
                        n2=2;
                        n1=1;
                    }
                    if (a3==1){
                        n2=2;
                        n1=1;
                    }
                    if (a3==0){
                        n1=2;
                        n2=1;
                    }
                }
                if (a1==0){
                    n1=0;
                    if (a3==2){
                        n3=2;
                        n2=1;
                    }
                    if (a3==1){
                        n2=2;
                        n3=1;
                    }
                    if (a3==0){
                        n2=2;
                        n3=1;
                    }
                }
                if (a1==1){
                    n2=0;
                    if (a3==0){
                        n3=1;
                        n1=2;
                    }
                    if (a3==2){
                        n3=2;
                        n1=1;
                    }
                    if (a3==1){
                        if (vert_h_cell[0][0]>=vert_h_cell[2][0]){
                            n3=1;
                            n1=2;
                        }
                        if (vert_h_cell[0][0]<vert_h_cell[2][0]){
                            n3=2;
                            n1=1;
                        }
                    }
                }
                new_vert_h_cell[n1][0]=E;
                new_vert_h_cell[n2][0]=fmax(E,vert_h_cell[n2][0]-(new_vert_h_cell[n1][0]-vert_h_cell[n1][0])/2.0);
                new_vert_h_cell[n3][0]=vert_h_cell[n3][0]-(new_vert_h_cell[n1][0]-vert_h_cell[n1][0])-(new_vert_h_cell[n2][0]-vert_h_cell[n2][0]);
            }
        }

        for(int i=0;i<new_vert_h_cell.dofs;i++){
            if  (new_vert_h_cell[i][0]<=E){
                new_vert_h_cell[i][0]=E;
            }
        }

        flag1=STEP(new_vert_h_cell[0][0]-(E+1E-7));
        flag2=STEP(new_vert_h_cell[1][0]-(E+1E-7));
        flag3=STEP(new_vert_h_cell[2][0]-(E+1E-7));
        npos=flag1+flag2+flag3;
        deltau=(vert_u_cell[0][0]*vert_h_cell[0][0]*(1-flag1))+(vert_u_cell[1][0]*vert_h_cell[1][0]*(1-flag2))+(vert_u_cell[2][0]*vert_h_cell[2][0]*(1-flag3));
        deltav=(vert_v_cell[0][0]*vert_h_cell[0][0]*(1-flag1))+(vert_v_cell[1][0]*vert_h_cell[1][0]*(1-flag2))+(vert_v_cell[2][0]*vert_h_cell[2][0]*(1-flag3));
        if (npos>0){
            new_vert_u_cell[0][0]=flag1*(vert_u_cell[0][0]*vert_h_cell[0][0]+(deltau/npos))/new_vert_h_cell[0][0];
            new_vert_u_cell[1][0]=flag2*(vert_u_cell[1][0]*vert_h_cell[1][0]+(deltau/npos))/new_vert_h_cell[1][0];
            new_vert_u_cell[2][0]=flag3*(vert_u_cell[2][0]*vert_h_cell[2][0]+(deltau/npos))/new_vert_h_cell[2][0];
            new_vert_v_cell[0][0]=flag1*(vert_v_cell[0][0]*vert_h_cell[0][0]+(deltav/npos))/new_vert_h_cell[0][0];
            new_vert_v_cell[1][0]=flag2*(vert_v_cell[1][0]*vert_h_cell[1][0]+(deltav/npos))/new_vert_h_cell[1][0];
            new_vert_v_cell[2][0]=flag3*(vert_v_cell[2][0]*vert_h_cell[2][0]+(deltav/npos))/new_vert_h_cell[2][0];
        }
        if (npos==0){
            new_vert_u_cell[0][0]=0;
            new_vert_u_cell[1][0]=0;
            new_vert_u_cell[2][0]=0;
            new_vert_v_cell[0][0]=0;
            new_vert_v_cell[1][0]=0;
            new_vert_v_cell[2][0]=0;
        }
        """ % {"epsilon": self.options.wd_mindep} 

        args = {
                "new_vert_v_cell": (func_v_tmp, RW),
                "new_vert_u_cell": (func_u_tmp, RW),
                "new_vert_h_cell": (func_eta_tmp, RW),
                "vert_h_cell": (H, READ),
                "vert_u_cell": (u, READ),
                "vert_v_cell": (v, READ)
               }  

        par_loop(slope_modification_2d_kernel, dx, args)  

        tmp_proj_func.sub(0).assign(func_u_tmp)
        tmp_proj_func.sub(1).assign(func_v_tmp)
        if self.options.element_family == 'rt-dg':
            uv_2d.project(tmp_proj_func) 
        else:
            uv_2d.assign(tmp_proj_func) 
        elev_2d.assign(func_eta_tmp - self.fields.bathymetry_2d) 

        return solution

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

    def assign_initial_conditions(self, elev=None, uv=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv: Initial condition for depth averaged velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        if elev is not None:
            elev_2d.project(elev)
        if uv is not None:
            uv_2d.project(uv)
        self.solution_2d_old.assign(self.fields.solution_2d) # added, Wei
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
   
        for key in self.exporters:
            self.exporters[key].export()

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
            By default ``options.outputdir``.
        :kwarg float t: simulation time. Overrides the time stamp stored in the
            hdf5 files.
        :kwarg int iteration: Overrides the iteration count in the hdf5 files.
        """
        if not self._initialized:
            self.initialize()
        if outputdir is None:
            outputdir = self.options.outputdir
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
        self.next_export_t = self.i_export*self.options.t_export
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.outputdir
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.t_export
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(self.i_export + offset)

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        norm_h = norm(self.fields.solution_2d.split()[1])
        norm_u = norm(self.fields.solution_2d.split()[0])

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def get_alpha(self, H0):
        """
        Alternative to try alpha, finding minimum alpha to let all depths below the threshold wd_mindep

        :arg H0: Minimum water depth
        """     
        if H0 >= 0.:
            return Constant(0.)
        elif not self.options.constant_alpha:
            return Constant(np.sqrt(0.25*self.options.wd_mindep**2 - 0.5*self.options.wd_mindep*H0) + 0.5*self.options.wd_mindep) # new formulated function, Wei
            #return Constant(np.sqrt(self.options.wd_mindep**2 - self.options.wd_mindep*H0) + self.options.wd_mindep) # artificial porosity method
            #return Constant(np.sqrt(4*self.options.wd_mindep*(self.options.wd_mindep-H0))) # original bathymetry changed method
        else:
            return Constant(self.options.wd_alpha)

    def thacker_compare(self, t, eta):
        # Print prescribed results, for Thacker # Wei
        D0 = Constant(50.)
        L = Constant(430620.)
        eta0 = Constant(2.)
        a = ((D0+eta0)**2-D0**2)/((D0+eta0)**2+D0**2)
        A = Constant(a)
        R = 430620.
        D = 50.
        omega = (8*9.81*D/(R*R))**0.5

        x = SpatialCoordinate(self.mesh2d)
        bathy = D0*(1-(x[0]*x[0]+x[1]*x[1])/(L*L))

        TMP = Constant(np.cos(omega*t))
        elev_t=D0*(sqrt(1-A*A)/(1-A*TMP) - 1 - (x[0]*x[0]+x[1]*x[1])*((1-A*A)/((1-A*TMP)**2)-1)/(L*L))
        elev_analytic = conditional(elev_t < -bathy, -bathy+self.options.wd_mindep, elev_t)
 
        #uv, eta = solution.split()
        eta_tilde = Function(FunctionSpace(self.mesh2d, 'DG', self.options.order))#Function(eta.function_space())
        wd_bath_displacement = self.eq_sw.water_height_displacement
        eta_tilde.project(eta + wd_bath_displacement(eta))
        #eta_tilde.project(eta)

        area = pi*475823.23**2
        e = errornorm(elev_analytic, eta_tilde)/np.sqrt(area)
        print_output(e)

    def iterate(self, update_forcings=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.t_end`` is reached.
        Exports fields to disk on ``options.t_export`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        # TODO I think export function is obsolete as callbacks are in place
        if not self._initialized:
            self.initialize()
        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.t_export

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_vol_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c)

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.t_end + t_epsilon:
            # uv & eta modified based on the wetting-drying method: a thin film # Wei
            self.solution_2d_old.assign(self.fields.solution_2d) # for exporting with adaptive time stepping, Wei
            self.timestepper.dt_const.assign(self.set_time_step(self.fields.solution_2d)) # Wei
            if self.options.wd_alpha is None:
                if self.options.wd_mindep is not None: 
                    # this thin-film wetting-drying scheme extremely relies on the high resolution at wetting-drying front
                    # has been benchmarked by Thacker test, but needs further benchmark by orther tests, Wei
                    uv_2d, elev_2d = self.fields.solution_2d.split()
                    native_space = uv_2d.function_space()
                    visu_space = exporter.get_visu_space(native_space)
                    tmp_proj_func = visu_space.get_work_function()
                    Interpolator(uv_2d, tmp_proj_func).interpolate()
                    visu_space.restore_work_function(tmp_proj_func)
                    UV = tmp_proj_func.dat.data
                    H = self.fields.bathymetry_2d.dat.data + elev_2d.dat.data 

                    ind = np.where(H[:] <= self.options.wd_mindep)[0]
                    H[ind] = self.options.wd_mindep
                    elev_2d.dat.data[ind] = (H - self.fields.bathymetry_2d.dat.data)[ind]
                    #uv_2d.dat.data[ind] = [0., 0.]
                    #UV[ind] = [0., 0.]
                    #uv = conditional((elev_2d + self.fields.bathymetry_2d) <= self.options.wd_mindep, zero(tmp_proj_func.ufl_shape), tmp_proj_func)
                    #uv_2d.project(tmp_proj_func)

                    #UV[H <= self.options.wd_mindep] = [0., 0.]
                    #H[H < self.options.wd_mindep] = self.options.wd_mindep

                    #eta_vector = self.fields.solution_2d.dat.data[1]
                    #tmp_vector = H - self.fields.bathymetry_2d.dat.data
                
                    #assert tmp_vector.shape[0] == eta_vector.shape[0]
                    #for i, ele in enumerate(tmp_vector):
                    #    eta_vector[i] = ele

                    #for j, uv in enumerate(UV):
                    #    self.fields.solution_2d.dat.data[0][j] = uv

                    #uv_2d.assign(tmp_proj_func)
                    #uv_2d.project(tmp_proj_func)

                    #self.fields.solution_2d = self.SlopeLimiter(self.fields.solution_2d)
                    self.fields.solution_2d = self.SlopeModification(self.fields.solution_2d)
            else:
                uv_2d, elev_2d = self.fields.solution_2d.split()
                H = self.fields.bathymetry_2d.dat.data + elev_2d.dat.data
                H_min = H.min()
                self.options.wd_alpha.assign(self.get_alpha(H_min)) 
                print_output('alpha = {:}'.format(self.options.wd_alpha.dat.data))

            self.timestepper.advance(self.simulation_time, update_forcings)

            #if self.options.wd_alpha is not None: # only for thacker test, Wei
                #t = self.simulation_time + self.dt
                #self.thacker_compare(t, elev_2d)
                #uv11, ele11 = split(self.fields.solution_2d)
                #self.thacker_compare(t, ele11) # to see what the difference between split(splution) and solution.split() is

            # Move to next time step
            self.iteration += 1
            #self.simulation_time = self.iteration*self.dt # it seems be not appropriate for adaptive time step? Wei; because no adaptive temporarily, could be added latter
            self.simulation_time = self.simulation_time + self.dt
            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon: # has been modified to export properly when adaptive timestepping; seems higher time cost, Wei, 58/05/2017
                if self.options.dt is not None and self.options.t_export % self.options.dt <= 1E-10:
                    self.export()
                else:
                    fac = (next_export_t - (self.simulation_time - self.dt)) / self.dt
                    tmp = fac * (self.fields.solution_2d - self.solution_2d_old) + self.solution_2d_old
                    self.export_tmp.assign(tmp)
                    self.exchange_tmp.assign(self.fields.solution_2d)
                    self.fields.solution_2d.assign(self.export_tmp)
                    self.export()
                    self.fields.solution_2d.assign(self.exchange_tmp)

                self.i_export += 1
                next_export_t += self.options.t_export

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                if next_export_t <= self.simulation_time:
                    raise ValueError('Export time step is too small that needs to be re-prescribed!')

                if export_func is not None:
                    export_func()


