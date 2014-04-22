'''
Copyright CERN 2014
Author: Adrian Oeftiger, oeftiger@cern.ch

This module provides various numerical integration methods
for Hamiltonian vector fields on (currently two-dimensional) phase space.
The integrators are separated according to symplecticity.
The method is_symple() is provided to check for symplecticity 
of a given integration method -- it may be used generically
for any integration method with the described signature.
'''
from __future__ import division
import numpy as np
import libTPSA

def is_symple(integrator):
	'''returns whether the given integrator is symplectic w.r.t. to a certain
	numerical tolerance (fixed by numpy.allclose).
	The decision is taken on whether the Jacobian determinant remains 1
	(after a time step of 1 while modelling a harmonic oscillator).
	The integrator input should be a function taking the argument 
	signature (x_initial, p_initial, timestep, H_p, H_x), where
	the first three arguments are numbers and H_p(p) and H_x(x) are
	functions taking one argument.'''
	x_initial = libTPSA.TPS([2, 1, 0])
	p_initial = libTPSA.TPS([0, 0, 1])
	timestep = 1
	x_final, p_final = integrator(x_initial, p_initial, 
								timestep, lambda pp:pp, lambda xx:xx)
	jacobian = np.linalg.det( [x_final.diff(), p_final.diff()] )
	return np.allclose(jacobian, 1.0)

class symple(object):
	'''Contains *symplectic* integrator algorithms. 
	The integrator input should be a function taking the argument 
	signature (x_initial, p_initial, timestep, H_p, H_x).
	It is assumed that the Hamiltonian is separable into a kinetic
	part T(p) (giving rise to H_p(p) = dH/dp which only depends on the 
	conjugate momentum p) and into a potential part V(x) (giving rise 
	to H_x(x) = dH/dx which only depends on the spatial coordinate x).'''

	@staticmethod
	def Euler_Cromer(x_initial, p_initial, timestep, H_p, H_x):
		'''Symplectic one-dimensional Euler Cromer O(T^2) Algorithm.
		This Euler_Cromer is explicite! keyword: drift-kick mechanism'''
		x_final = x_initial + timestep * H_p(p_initial)
		p_final = p_initial - timestep * H_x(x_final)
		return x_final, p_final

	@staticmethod
	def Verlet(x_initial, p_initial, timestep, H_p, H_x):
		'''Symplectic one-dimensional (Velocity) Verlet O(T^3) Algorithm.
		keyword: leapfrog mechanism'''
		p_intermediate = p_initial - 0.5 * timestep * H_x(x_initial)
		x_final = x_initial + timestep * H_p(p_intermediate)
		p_final = p_intermediate - 0.5 * timestep * H_x(x_final)
		return x_final, p_final

	@staticmethod
	def Ruth(x_initial, p_initial, timestep, H_p, H_x):
		'''Symplectic one-dimensional Ruth and Forest O(T^5) Algorithm.
		Harvard: 1992IAUS..152..407Y'''
		twoot = np.power(2, 1. / 3)
		fc = 1. / (2 - twoot)
		# ci: drift, di: kick
		c1 = fc / 2.;				c4 = c1
		c2 = (1 - twoot) * fc / 2.;	c3 = c2
		d1 = fc;					d3 = d1
		d2 = -twoot * fc
		# d4 = 0
		x_intermediate = x_initial + timestep * c4 * H_p(p_initial)
		p_intermediate = p_initial - timestep * d3 * H_x(x_intermediate)
		x_intermediate += timestep * c3 * H_p(p_intermediate)
		p_intermediate -= timestep * d2 * H_x(x_intermediate)
		x_intermediate += timestep * c2 * H_p(p_intermediate)
		p_final = p_intermediate - timestep * d1 * H_x(x_intermediate)
		x_final = x_intermediate + timestep * c1 * H_p(p_final)
		return x_final, p_final

class non_symple(object):
	'''Contains *non-symplectic* integrator algorithms.
	The integrator input should be a function taking the argument 
	signature (x_initial, p_initial, timestep, H_p, H_x).
	H_x(x) = dH/dx is a function of x only while
	H_p(p) = dH/dp is a function of p only.'''

	@staticmethod
	def Euler(x_initial, p_initial, timestep, H_p, H_x):
		'''Non-symplectic one-dimensional Euler O(T^2) Algorithm.'''
		x_final = x_initial + timestep * H_p(p_initial)
		p_final = p_initial - timestep * H_x(x_initial)
		return x_final, p_final

	@staticmethod
	def RK2(x_initial, p_initial, timestep, H_p, H_x):
		'''Non-symplectic one-dimensional Runge Kutta 2 O(T^3) Algorithm.'''
		x_a =   timestep * H_p(p_initial)
		p_a = - timestep * H_x(x_initial)
		x_b =   timestep * H_p(p_initial + 0.5 * p_a)
		p_b = - timestep * H_x(x_initial + 0.5 * x_a)
		x_final = x_initial + x_b
		p_final = p_initial + p_b
		return x_final, p_final

	@staticmethod
	def RK4(x_initial, p_initial, timestep, H_p, H_x):
		'''Non-symplectic one-dimensional Runge Kutta 4 O(T^5) Algorithm.'''
		x_a =   timestep * H_p(p_initial)
		p_a = - timestep * H_x(x_initial)
		x_b =   timestep * H_p(p_initial + 0.5 * p_a)
		p_b = - timestep * H_x(x_initial + 0.5 * x_a)
		x_c =   timestep * H_p(p_initial + 0.5 * p_b)
		p_c = - timestep * H_x(x_initial + 0.5 * x_b)
		x_d =   timestep * H_p(p_initial + p_c)
		p_d = - timestep * H_x(x_initial + x_c)
		x_final = x_initial + x_a / 6. + x_b / 3. + x_c / 3. + x_d / 6.
		p_final = p_initial + p_a / 6. + p_b / 3. + p_c / 3. + p_d / 6.
		return x_final, p_final
