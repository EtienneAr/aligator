
import numpy as np
import aligator
g= 9.8
l = 0.6

class CustomFunction(aligator.StageFunction):
    def __init__(self, space: aligator.manifolds.ManifoldAbstract, nu):
        self.space = space
        ndx = space.ndx
        super().__init__(ndx, nu, ndx)

    def __getinitargs__(self):
        return (self.space, self.nu)

    def evaluate(self, x, u, data: aligator.StageFunctionData):
        data.value[:] = self.space.difference(x, self.space.neutral())

    def computeJacobians(self, x, u, data: aligator.StageFunctionData):
        data.Jx[:] = self.space.Jdifference(x, self.space.neutral(), 0)

class UpkieDynamics(aligator.dynamics.ExplicitDynamicsModel):
    #state = (rdot, phidot, theta, thetadot)

    def __init__(self,space, dt: float):
        self.dt = dt
        super().__init__(space, 2)

    def __getinitargs__(self):
        return (self.space, self.nu, self.dt)

    def forward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        rdot , phidot, theta, thetadot = x
        rdotdot, phidotdot = u
        return (rdot + rdotdot*self.dt,
                phidot + phidotdot*self.dt,
                theta + thetadot*self.dt,
                thetadot + self.dt * (np.sin(theta)*g/l - np.cos(theta)*rdotdot/l))

    def dForward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        rdot , phidot, theta, thetadot = x
        rdotdot, phidotdot = u
        J_x = np.eye(4)
        J_x[3,2] = self.dt
        dthetadot_dthetadotdot = self.dt  * (np.cos(theta) * g/l + np.sin(theta) * rdotdot/l)
        J_x[3,3] = dthetadot_dthetadotdot
        J_u = np.zeros((2,4))
        J_u[0,0] = self.dt
        J_u[1,1] = self.dt
        J_u[0,3] = -self.dt * np.cos(theta)/l
        data.J_x = J_x
        data.J_u = J_u

def test_controller():
    space = aligator.manifolds.VectorSpace(4)

    x0 =np.array([0,0,np.pi/6,0])

    cost = aligator.QuadraticStateCost(space, 2, np.zeros(4), np.eye(4))
    dynamics = UpkieDynamics(space,0.01)
    stage = aligator.StageModel(cost, dynamics)

    stages = [stage, ]*100
    prob = aligator.TrajOptProblem(x0, stages, cost)
    pd = aligator.TrajOptData(prob)
    print(pd)
    mu_init = 1e-8
    verbose = aligator.VerboseLevel.VERBOSE
    TOL = 1e-6
    MAX_ITER = 300
    solver = aligator.SolverProxDDP(TOL, mu_init, max_iters=MAX_ITER, verbose=verbose)
    solver.bcl_params.mu_lower_bound = 1e-11
    callback = aligator.HistoryCallback(solver)
    solver.registerCallback("his", callback)

    u0 = np.zeros(2)
    us_i = [u0] * 100
    xs_i = aligator.rollout(dynamics, x0, us_i)

    solver.setup(prob)
    solver.run(prob, xs_i, us_i)
    res = solver.results
    print(res)



if __name__ == "__main__":
    test_controller()
