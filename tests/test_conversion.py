import unittest
from importlib.util import find_spec

import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize._numdiff import approx_derivative

if find_spec("pandas"):
    from ads.conversion import HORIZON, delay_cdf, delayed_objective, observed_labels


@unittest.skipUnless(find_spec("pandas"), "requires requirements-conversion.txt")
class ConversionTests(unittest.TestCase):
    def test_training_does_not_see_future_conversion(self):
        timestamp = np.array([10,10,10,10])
        conversion = np.array([20,40,-1,10+HORIZON+1])
        np.testing.assert_array_equal(observed_labels(timestamp,conversion,30),[1,0,0,0])
        np.testing.assert_array_equal(observed_labels(timestamp,conversion,10+HORIZON+10),[1,1,0,0])
        with self.assertRaises(ValueError):
            observed_labels(timestamp,conversion,9)

    def test_cdf_is_horizon_conditional(self):
        actual=delay_cdf(np.array([0,1,5,HORIZON]), np.array([1,2,4,6]))
        np.testing.assert_allclose(actual,[.02,.25,.75,1])

    def test_delayed_likelihood_gradient(self):
        x=csr_matrix([[1.,0],[0,1],[1,1],[0,0]])
        y=np.array([1,0,1,0])
        cdf=np.array([.2,.4,.8,1.])
        theta=np.array([.2,-.3,-1.])
        numerical=approx_derivative(lambda p: delayed_objective(p,x,y,cdf)[0],theta).ravel()
        np.testing.assert_allclose(delayed_objective(theta,x,y,cdf)[1],numerical,rtol=1e-5)
        # A completely mature cohort reduces to ordinary binary cross entropy.
        p=1/(1+np.exp(-(x@theta[:-1]+theta[-1])))
        loss=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
        self.assertAlmostEqual(delayed_objective(theta,x,y,np.ones(4),0)[0],loss)
