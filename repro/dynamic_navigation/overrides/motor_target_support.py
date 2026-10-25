"""Jointly choose support force and a rate-bounded leg position request."""
import math

import numpy as np


def _interior_point_qp(hessian, linear, rows, lo, hi):
    """Solve the same scaled inequalities with a strict conic optimum.

    This is a numerical fallback, never a feasibility-only motor command.
    AlmostSolved is rejected and the caller still checks original units.
    """
    from types import SimpleNamespace
    from scipy import sparse
    import clarabel

    # Eliminating an airborne foot's fixed variables leaves exact 0<=b
    # inequalities. Row scaling can turn their harmless b into 1e13 and
    # stall a conic solver. Drop only mathematically tautological zero rows;
    # retain every nonzero row, and any infeasible zero row (0>b or 0<a).
    zero_rows=np.asarray(abs(rows).sum(axis=1)).ravel()==0.
    redundant=zero_rows & (lo<=0.) & (hi>=0.)
    removed_zero_rows=int(redundant.sum())
    rows,lo,hi=rows[~redundant],lo[~redundant],hi[~redundant]
    upper=np.isfinite(hi);lower=np.isfinite(lo)
    matrix=sparse.vstack([rows[upper],-rows[lower]],format='csc')
    right=np.r_[hi[upper],-lo[lower]]
    settings=clarabel.DefaultSettings()
    settings.verbose=False;settings.max_iter=200
    settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9;settings.tol_feas=1e-9
    optimizer=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(hessian)),linear,
        matrix,right,[clarabel.NonnegativeConeT(len(right))],settings)
    result=optimizer.solve()
    success=result.status==clarabel.SolverStatus.Solved
    return SimpleNamespace(x=np.asarray(result.x),success=success,
        status=0 if success else int(result.status),message=str(result.status),
        solver='Clarabel',solver_status=int(result.status),iterations=int(result.iterations),
        primal_residual=float(result.r_prim),dual_residual=float(result.r_dual),
        objective_gap=float(abs(result.obj_val-result.obj_val_dual)),
        solver_version=clarabel.__version__,removed_zero_rows=removed_zero_rows)


def _solve_convex_qp(hessian, linear, matrix, right, lower, upper, initial):
    """Use a convex QP solver, eliminating fixed (airborne) force variables.

    All original rows and bounds are retained. A feasibility-only LP is never
    an output: only a converged optimum proceeds to the independent checks.
    """
    from types import SimpleNamespace
    import osqp
    from scipy import sparse

    fixed=lower==upper;free=~fixed;offset=np.zeros(len(lower));offset[fixed]=lower[fixed]
    # Force intervals are hundreds of N; servo intervals can be .15 Nm.
    # Express both in their own interval units, then scale each row. This is
    # an invertible coordinate change, not a relaxation of physical limits.
    scale=np.maximum(1e-4,(upper[free]-lower[free])/2)
    reduced_hessian=hessian[np.ix_(free,free)]*scale[:,None]*scale[None,:]
    reduced_linear=(hessian@offset-linear)[free]*scale
    objective_scale=max(1.,float(abs(reduced_hessian).max()))
    reduced_matrix=matrix[:,free]*scale
    row_scale=np.maximum(1e-12,abs(reduced_matrix).max(axis=1))
    reduced_matrix/=row_scale[:,None]
    rows=sparse.vstack([sparse.csc_matrix(reduced_matrix),sparse.eye(int(free.sum()),format='csc')],format='csc')
    lo=np.r_[np.full(len(right),-np.inf),lower[free]/scale]
    hi=np.r_[(right-matrix@offset)/row_scale,upper[free]/scale]
    optimizer=osqp.OSQP()
    optimizer.setup(P=sparse.csc_matrix(np.triu(reduced_hessian/objective_scale)),
        q=reduced_linear/objective_scale,A=rows,l=lo,u=hi,
        verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=10000,
        adaptive_rho_interval=25,polishing=True)
    optimizer.warm_start(x=initial[free]/scale)
    result=optimizer.solve(raise_error=False)
    attempts=[dict(solver='OSQP',native_status=int(result.info.status_val),
        message=str(result.info.status),iterations=int(result.info.iter),
        primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res))]
    # Saved Y3/I warehouse counterexamples reach the iteration cap with either
    # "maximum iterations" or "solved inaccurate". Neither is a command.
    # Change numerical method, retaining the exact objective and constraints;
    # infeasibility and non-convex statuses remain hard failures.
    if result.info.status_val in (2,7):
        try:
            alternative=_interior_point_qp(reduced_hessian/objective_scale,
                reduced_linear/objective_scale,rows,lo,hi)
            attempts.append(dict(solver=alternative.solver,native_status=alternative.solver_status,
                message=alternative.message,iterations=alternative.iterations,
                primal_residual=alternative.primal_residual,dual_residual=alternative.dual_residual,
                objective_gap=alternative.objective_gap,version=alternative.solver_version,
                removed_zero_rows=alternative.removed_zero_rows))
            if alternative.success:
                value=offset.copy();value[free]=alternative.x*scale
                alternative.x=value;alternative.fixed_variables=int(fixed.sum())
                alternative.attempts=attempts
                return alternative
        except ImportError:
            attempts.append(dict(solver='Clarabel',message='optional dependency unavailable; no fallback command'))
    value=offset.copy()
    if result.x is not None:value[free]=result.x*scale
    else:value[:]=np.nan
    # OSQP's "solved inaccurate" is deliberately not accepted.
    success=result.info.status_val==1
    return SimpleNamespace(x=value,success=success,status=0 if success else int(result.info.status_val),
        message=str(result.info.status),solver='OSQP',solver_status=int(result.info.status_val),
        iterations=int(result.info.iter),primal_residual=float(result.info.prim_res),
        dual_residual=float(result.info.dual_res),fixed_variables=int(fixed.sum()),attempts=attempts)


def solve(feet,contacts,vertical,moment,horizontal,*,jacobians,friction,per_foot_limit,
          feedforward_limit,effort_scale,motor,previous_target,position_lower,position_upper,max_step):
    """Decide actuator targets and FF, retaining original slew and DC limits."""
    from scipy.optimize import linprog
    from .balance_recovery import SupportAllocationError
    feet=np.asarray(feet,dtype=float);mask=np.asarray(contacts,dtype=bool);jac=np.asarray(jacobians,dtype=float)
    arrays=[np.asarray(motor[key],dtype=float) for key in
        ('position_target_rad','pd_bias_nm','stiffness','lower_nm','upper_nm')]
    previous,lower,upper=[np.asarray(value,dtype=float) for value in (previous_target,position_lower,position_upper)]
    scalars=(vertical,*moment,*horizontal,friction,per_foot_limit,feedforward_limit,effort_scale,max_step)
    if (feet.shape!=(4,3) or mask.shape!=(4,) or jac.shape!=(4,3,3)
            or any(a.shape!=(12,) for a in (*arrays,previous,lower,upper))
            or not all(np.isfinite(a).all() for a in (feet,jac,*arrays,previous,lower,upper))
            or not all(math.isfinite(v) for v in scalars) or not 0<friction<=1
            or not 0<effort_scale<=1 or min(per_foot_limit,feedforward_limit,max_step)<=0 or np.any(lower>=upper)):
        raise ValueError('Joint support requires finite dynamics and original position/motor bounds')
    nominal,bias,kp,motor_lo,motor_hi=arrays
    if np.any(kp<=0) or np.any(motor_lo+.1>=motor_hi-.1):
        raise ValueError('Joint support requires positive stiffness and nonempty motor margin')
    # Inherited violations may decrease at the original rate, never worsen.
    target_lo=np.maximum(previous-max_step,np.minimum(previous,lower))
    target_hi=np.minimum(previous+max_step,np.maximum(previous,upper))
    bias_lo=kp*(target_lo-nominal);bias_hi=kp*(target_hi-nominal)
    rows=[];rhs=[];wrench=np.zeros((5,24));bounds=[]
    for leg in range(4):
        sl=slice(3*leg,3*leg+3);x,y,z=feet[leg]
        wrench[:3,sl]=np.eye(3);wrench[3:,sl]=[[0.,-z,y],[z,0.,-x]]
        cap=per_foot_limit if mask[leg] else 0.
        bounds.extend([(-cap,cap),(-cap,cap),(0.,cap)])
        for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            row=np.zeros(24);row[sl]=[sx,sy,-friction];rows.append(row);rhs.append(0.)
        for joint,torque in enumerate(-jac[leg].T):
            row=np.zeros(24);row[sl]=torque
            rows.extend([row,-row]);rhs.extend([feedforward_limit,feedforward_limit])
            row=row*effort_scale;index=leg+4*joint;row[12+index]=1.
            rows.extend([row,-row]);rhs.extend([motor_hi[index]-.1-bias[index],bias[index]-motor_lo[index]-.1])
    bounds.extend(zip(bias_lo,bias_hi))
    matrix=np.asarray(rows);right=np.asarray(rhs)
    bound_lo=np.array([a for a,_ in bounds]);bound_hi=np.array([b for _,b in bounds])
    feasibility=linprog(np.zeros(24),A_ub=matrix,b_ub=right,bounds=bounds,method='highs')
    result=None;feasible=False
    if feasibility.status==0:
        desired=np.array([*horizontal,vertical,*moment]);weights=np.array([.2,.2,1.,8.,8.])
        objective=weights[:,None]*wrench;goal=weights*desired
        hessian=objective.T@objective+np.diag([1e-4]*12+[25.]*12)
        linear=objective.T@goal
        result=_solve_convex_qp(hessian,linear,matrix,right,bound_lo,bound_hi,feasibility.x)
        value=result.x
        feasible=bool(np.isfinite(value).all() and np.all(matrix@value<=right+1e-5)
            and np.all(value>=bound_lo-1e-6) and np.all(value<=bound_hi+1e-6))
    report=dict(enabled=True,previous_target_rad=previous.tolist(),nominal_target_rad=nominal.tolist(),
        position_lower_rad=lower.tolist(),position_upper_rad=upper.tolist(),max_step_rad=float(max_step),
        target_lower_rad=target_lo.tolist(),target_upper_rad=target_hi.tolist(),
        feasibility_lp_status=int(feasibility.status),constraint_set_feasible=feasibility.status==0,
        qp_status=None if result is None else int(result.status),
        qp_message=str(feasibility.message if result is None else result.message),qp_solution_feasible=feasible,
        scope='Joint actuator position and contact-force request inside original rate, inherited position box and PD/DC motor limits')
    if result is not None:
        report.update(qp_solver=result.solver,qp_native_status=result.solver_status,
            qp_iterations=result.iterations,qp_primal_residual=result.primal_residual,
            qp_dual_residual=result.dual_residual,qp_fixed_variables=result.fixed_variables,
            qp_attempts=result.attempts)
    if result is None or not result.success or not feasible:
        report.update(command_sent=False,feet_relative=feet.tolist(),contacts=mask.tolist(),jacobians=jac.tolist(),
            motor=motor,effort_scale=float(effort_scale),vertical_force_n=float(vertical),
            moment_xy_nm=list(map(float,moment)),horizontal_force_n=list(map(float,horizontal)),
            friction=float(friction),per_foot_limit_n=float(per_foot_limit),feedforward_limit_nm=float(feedforward_limit))
        raise SupportAllocationError('Joint motor support allocation failed: '+report['qp_message'],report)
    forces=value[:12].reshape(4,3);target=nominal+value[12:]/kp
    report.update(selected_target_rad=target.tolist(),maximum_correction_rad=float(abs(target-nominal).max()))
    residual=np.array([vertical,*moment])-(wrench@value)[2:]
    return forces,residual,target,report
