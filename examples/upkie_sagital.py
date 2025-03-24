
#state = (rdot, phidot, theta, thetadot)
import numpy as np
import aligator
g= 9.8
l = 0.6




import aligator
import numpy as np
import pytest


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


class TwistModelExplicit(aligator.dynamics.ExplicitDynamicsModel):
    def __init__(self, space, nu, dt: float):
        self.dt = dt
        super().__init__(space, nu)

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


def test_abstract():
    space = aligator.manifolds.SE2()
    ndx = space.ndx
    nu = 3
    nr = 1
    fun = aligator.StageFunction(ndx, nu, nr)
    data = fun.createData()
    print(data)


def test_custom_controlbox():
    space = aligator.manifolds.SE2()
    ndx = space.ndx
    nu = 3

    fun = CustomFunction(space, nu)
    data1: aligator.StageFunctionData = fun.createData()

    lbd0 = np.zeros(fun.nr)
    x0 = space.rand()
    u0 = np.random.randn(nu)

    fun.evaluate(x0, u0, data1)
    fun.computeJacobians(x0, u0, data1)
    print(data1.value)
    print(data1.Ju)

    # expected behavior: initial value of vhp_buffer is 0
    assert np.allclose(data1.vhp_buffer, 0.0)

    rdm = np.random.randn(*data1.vhp_buffer.shape)
    data1.vhp_buffer[:, :] = rdm
    fun.computeVectorHessianProducts(x0, u0, lbd0, data1)
    # expected behavior: unimplemented computeVectorHessianProducts does nothing.
    assert np.allclose(data1.vhp_buffer, rdm)

    cost = aligator.QuadraticStateCost(space, nu, space.neutral(), np.eye(ndx))
    dynamics = TwistModelExplicit(space, nu, 0.1)
    stage = aligator.StageModel(cost, dynamics)
    stage.addConstraint(fun, aligator.constraints.EqualityConstraintSet())
    data = stage.createData()
    stage.evaluate(x0, u0, x0, data)

    stages = [stage, stage, stage]
    prob = aligator.TrajOptProblem(x0, stages, cost)
    pd = aligator.TrajOptData(prob)
    print(pd)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
