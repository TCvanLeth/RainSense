#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:39:12 2018

@author: tcvanleth

NB: The Code in this module was ported from the Minpack Fortran library, with
minor alterations to allow breaking off the solver early if certain conditions
are met.

Minpack Copyright Notice (1999) University of Chicago.  All rights reserved

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above
copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials
provided with the distribution.

3. The end-user documentation included with the
redistribution, if any, must include the following
acknowledgment:

   "This product includes software developed by the
   University of Chicago, as Operator of Argonne National
   Laboratory.

Alternately, this acknowledgment may appear in the software
itself, if and wherever such third-party acknowledgments
normally appear.

4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS"
WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDER, THE
UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND
THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE
OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY
OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF
THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4)
DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION
UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL
BE CORRECTED.

5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT
HOLDER, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF
ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF
ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER
SUCH LIABILITY IS ASSERTED ON THE BASIS OF CONTRACT, TORT
(INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE,
EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
POSSIBILITY OF SUCH LOSS OR DAMAGES.

"""

import numpy as np
from numba import jit


@jit(nopython=True)
def hybrj(func, Dfun, args, x, xmin, xmax, maxfev=0, xtol=1.49012e-08, factor=100):
    """
     subroutine hybrj

     the purpose of hybrj is to find a zero of a system of
     n nonlinear functions in n variables by a modification
     of the powell hybrid method. the user must provide a
     subroutine which calculates the functions and the jacobian.

     the subroutine statement is

       subroutine hybrj(fcn,n,x,fvec,fjac,ldfjac,xtol,maxfev,diag,
                        mode,factor,nprint,info,nfev,njev,r,lr,qtf,
                        wa1,wa2,wa3,wa4)

     where

       fcn is the name of the user-supplied subroutine which
         calculates the functions and the jacobian. fcn must
         be declared in an external statement in the user
         calling program, and should be written as follows.

         subroutine fcn(n,x,fvec,fjac,ldfjac,iflag)
         integer n,ldfjac,iflag
         double precision x(n),fvec(n),fjac(ldfjac,n)
         ----------
         if iflag = 1 calculate the functions at x and
         return this vector in fvec. do not alter fjac.
         if iflag = 2 calculate the jacobian at x and
         return this matrix in fjac. do not alter fvec.
         ---------
         return
         end

         the value of iflag should not be changed by fcn unless
         the user wants to terminate execution of hybrj.
         in this case set iflag to a negative integer.

       x is an array of length n. on input x must contain
         an initial estimate of the solution vector. on output x
         contains the final estimate of the solution vector.

       fvec is an output array of length n which contains
         the functions evaluated at the output x.

       fjac is an output n by n array which contains the
         orthogonal matrix q produced by the qr factorization
         of the final approximate jacobian.

       xtol is a nonnegative input variable. termination
         occurs when the relative error between two consecutive
        iterates is at most xtol.

       maxfev is a positive integer input variable. termination
         occurs when the number of calls to fcn with iflag = 1
         has reached maxfev.

       diag is an array of length n. if mode = 1 (see
         below), diag is internally set. if mode = 2, diag
         must contain positive entries that serve as
         multiplicative scale factors for the variables.

       mode is an integer input variable. if mode = 1, the
         variables will be scaled internally. if mode = 2,
         the scaling is specified by the input diag. other
         values of mode are equivalent to mode = 1.

       factor is a positive input variable used in determining the
         initial step bound. this bound is set to the product of
         factor and the euclidean norm of diag*x if nonzero, or else
         to factor itself. in most cases factor should lie in the
         interval (.1,100.). 100. is a generally recommended value.

       nprint is an integer input variable that enables controlled
         printing of iterates if it is positive. in this case,
         fcn is called with iflag = 0 at the beginning of the first
         iteration and every nprint iterations thereafter and
         immediately prior to return, with x and fvec available
         for printing. fvec and fjac should not be altered.
         if nprint is not positive, no special calls of fcn
         with iflag = 0 are made.

       info is an integer output variable. if the user has
         terminated execution, info is set to the (negative)
         value of iflag. see description of fcn. otherwise,
         info is set as follows.

         info = 0   improper input parameters.

         info = 1   relative error between two consecutive iterates
                    is at most xtol.

         info = 2   number of calls to fcn with iflag = 1 has
                    reached maxfev.

         info = 3   xtol is too small. no further improvement in
                    the approximate solution x is possible.

         info = 4   iteration is not making good progress, as
                    measured by the improvement from the last
                    five jacobian evaluations.

         info = 5   iteration is not making good progress, as
                    measured by the improvement from the last
                    ten iterations.

       nfev is an integer output variable set to the number of
         calls to fcn with iflag = 1.

       njev is an integer output variable set to the number of
         calls to fcn with iflag = 2.

       r is an output array of length lr which contains the
         upper triangular matrix produced by the qr factorization
         of the final approximate jacobian, stored rowwise.

       qtf is an output array of length n which contains
         the vector (q transpose)*fvec.

     subprograms called

       user-supplied ...... fcn

     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more
    """
    # epsmch is the machine precision.
    epsmch = np.finfo(np.float64).eps
    nfev = 0
    njev = 0
    fnorm = 0

    # check the input parameters for errors.
    if xtol < 0. or factor <= 0.:
        return x, 0
    n = len(x)
    if maxfev <= 0:
        maxfev = 200 * (n + 1)

    # evaluate the function at the starting point
    # and calculate its norm.
    fvec = func(x, args)
    if np.any(np.isinf(fvec) | np.isnan(fvec)):
        return x, 5
    nfev = 1
    fnorm = enorm(n, fvec)

    # initialize iteration counter and monitors.
    niter = 0
    ncsuc = 0
    ncfail = 0
    nslow1 = 0
    nslow2 = 0

    # beginning of the outer loop.
    while True:
        jeval = True

        # calculate the jacobian matrix.
        fjac = Dfun(x, args)
        if np.any(np.isinf(fjac) | np.isnan(fjac)):
            return x, 4
        njev += 1

        # compute the qr factorization of the jacobian.
        fjac, wa1, wa2 = qrfac(n, n, fjac)

        # on the first iteration and if mode is 1, scale according
        # to the norms of the columns of the initial jacobian.
        if niter == 0:
            wa2 = np.empty(n)
            diag = np.empty(n)
            for j in range(n):
                if wa2[j] == 0:
                    diag[j] = 1.
                else:
                    diag[j] = wa2[j]

            # on the first iteration, calculate the norm of the scaled x
            # and initialize the step bound delta.
            wa3 = diag * x
            xnorm = enorm(n, wa3)
            delta = factor * xnorm
            if delta == 0.:
                delta = factor

        # form (q transpose)*fvec and store in qtf.
        qtf = fvec.copy()
        for j in range(n):
            if fjac[j, j] != 0.:
                ssum = 0.
                for i in range(j, n):
                    ssum += fjac[j, i] * qtf[i]
                temp = -ssum / fjac[j, j]
                for i in range(j, n):
                    qtf[i] = qtf[i] + fjac[j, i] * temp

        # copy the triangular factor of the qr factorization into r.
        r = np.empty(len(triu_indices(n)))
        for j in range(n):
            l = j
            if (j >= 1):
                for i in range(j):
                    r[l] = fjac[j, i]
                    l += n - i - 1
            r[l] = wa1[j]

        # accumulate the orthogonal factor in fjac.
        fjac = qform(n, n, fjac)

        # rescale if necessary.
        diag = np.maximum(diag, wa2)

        # beginning of the inner loop.
        while True:
            # determine the direction p.
            wa1 = -dogleg(n, r, diag, qtf, delta)

            # store the direction p and x + p. calculate the norm of p.
            wa2 = x + wa1
            wa3 = diag * wa1
            pnorm = enorm(n, wa3)

            # on the first iteration, adjust the initial step bound.
            if niter == 0:
                delta = min(delta, pnorm)

            # evaluate the function at x + p and calculate its norm.
            wa4 = func(wa2, args)
            if np.any(np.isinf(wa4) | np.isnan(wa4)):
                return x, 5
            nfev += 1
            fnorm1 = enorm(n, wa4)

            # compute the scaled actual reduction.
            if fnorm1 < fnorm:
                actred = 1. - (fnorm1 / fnorm)**2
            else:
                actred = -1.

            # compute the scaled predicted reduction.
            l = 0
            for i in range(n):
                ssum = 0.
                for j in range(i, n):
                    ssum += r[l] * wa1[j]
                    l += 1
                wa3[i] = qtf[i] + ssum
            temp = enorm(n, wa3)
            if temp < fnorm:
                prered = 1. - (temp / fnorm)**2
            else:
                prered = 0.

            # compute the ratio of the actual to the predicted reduction.
            if prered > 0.:
                ratio = actred / prered
            else:
                ratio = 0.

            # update the step bound.
            if ratio >= 0.1:
                ncfail = 0
                ncsuc += 1
                if ratio >= 0.5 or ncsuc > 1:
                    delta = max(delta, pnorm / 0.5)
                if abs(ratio - 1.) <= 0.1:
                    delta = pnorm / 0.5
            else:
                ncsuc = 0
                ncfail += 1
                delta *= 0.5

            # test for successful iteration.
            if ratio >= 1e-4:
                # successful iteration. update x, fvec, and their norms.
                x = wa2.copy()
                wa2 = diag * x
                fvec = wa4.copy()
                xnorm = enorm(n, wa2)
                fnorm = fnorm1
                niter += 1

            # determine the progress of the iteration.
            nslow1 += 1
            if actred >= 1e-3:
                nslow1 = 0
            if jeval:
                nslow2 += 1
            if actred >= 0.1:
                nslow2 = 0

            # test for convergence.
            if delta <= xtol * xnorm or fnorm == 0.:
                return x, 1

            # tests for termination and stringent tolerances.
            if nslow1 == 10:
                return x, 5
            elif nslow2 == 5:
                return x, 4
            elif 0.1 * max(0.1 * delta, pnorm) <= epsmch * xnorm:
                return x, 3
            elif nfev >= maxfev:
                return x, 2
            elif np.any(x > xmax) or np.any(x < xmin):
                return x, 6

            # criterion for recalculating jacobian.
            if ncfail == 2:
                break

            # calculate the rank one modification to the jacobian
            # and update qtf if necessary.
            wa1 = diag * ((diag * wa1) / pnorm)

            for j in range(n):
                ssum = 0.
                for i in range(n):
                    ssum += fjac[j, i] * wa4[i]
                wa2[j] = (ssum - wa3[j]) / pnorm
                if ratio >= 1e-4:
                    qtf[j] = ssum

            # compute the qr factorization of the updated jacobian.
            r, wa2, wa3 = r1updt(n, n, r, wa1, wa2)
            fjac = r1mpyq(n, fjac, wa2, wa3)
            qtf = r1mpyq(n, qtf, wa2, wa3)

            # end of the inner loop.
            jeval = False


@jit(nopython=True, cache=True)
def dogleg(n, r, diag, qtb, delta):
    """
     subroutine dogleg

     given an m by n matrix a, an n by n nonsingular diagonal
     matrix d, an m-vector b, and a positive number delta, the
     problem is to determine the convex combination x of the
     gauss-newton and scaled gradient directions that minimizes
     (a*x - b) in the least squares sense, subject to the
     restriction that the euclidean norm of d*x be at most delta.

     this subroutine completes the solution of the problem
     if it is provided with the necessary information from the
     qr factorization of a. that is, if a = q*r, where q has
     orthogonal columns and r is an upper triangular matrix,
     then dogleg expects the full upper triangle of r and
     the first n components of (q transpose)*b.

     the subroutine statement is

       subroutine dogleg(n,r,lr,diag,qtb,delta,x,wa1,wa2)

     where

       n is a positive integer input variable set to the order of r.

       r is an input array of length lr which must contain the upper
         triangular matrix r stored by rows.

       lr is a positive integer input variable not less than
         (n*(n+1))/2.

       diag is an input array of length n which must contain the
         diagonal elements of the matrix d.

       qtb is an input array of length n which must contain the first
         n elements of the vector (q transpose)*b.

       delta is a positive input variable which specifies an upper
         bound on the euclidean norm of d*x.

       x is an output array of length n which contains the desired
         convex combination of the gauss-newton direction and the
         scaled gradient direction.

       wa1 and wa2 are work arrays of length n.

     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more
    """

    # epsmch is the machine precision.
    epsmch = np.finfo(np.float64).eps

    # first, calculate the gauss-newton direction.
    x = np.empty(n)
    jj = (n * (n + 1)) // 2
    for k in range(1, n+1):
        j = n - k
        jj -= k
        l = jj + 1
        ssum = 0.
        if (n >= j+2):
            for i in range(j+1, n):
                ssum += r[l] * x[i]
                l += 1
        temp = r[jj]
        if (temp == 0.):
            l = j
            for i in range(j+1):
                temp = max(temp, abs(r[l]))
                l += n - i - 1
            temp *= epsmch
            if (temp == 0.):
                temp = epsmch
        x[j] = (qtb[j] - ssum) / temp

    # test whether the gauss-newton direction is acceptable.
    wa1 = np.zeros(n)
    wa2 = diag * x
    qnorm = enorm(n, wa2)
    if (qnorm <= delta):
        return x

    # the gauss-newton direction is not acceptable.
    # next, calculate the scaled gradient direction.
    l = 0
    for j in range(n):
        temp = qtb[j]
        for i in range(j, n):
            wa1[i] += r[l] * temp
            l += 1
        wa1[j] /= diag[j]

    # calculate the norm of the scaled gradient and test for
    # the special case in which the scaled gradient is zero.
    gnorm = enorm(n, wa1)
    sgnorm = 0.
    alpha = delta / qnorm
    if (gnorm != 0.):
        # calculate the point along the scaled gradient
        # at which the quadratic is minimized.
        wa1 = (wa1 / gnorm) / diag
        l = 0
        for j in range(n):
            ssum = 0.
            for i in range(j, n):
                ssum += r[l] * wa1[i]
                l += 1
            wa2[j] = ssum
        temp = enorm(n, wa2)
        sgnorm = (gnorm / temp) / temp

        # test whether the scaled gradient direction is acceptable.
        alpha = 0.
        if (sgnorm < delta):
            # the scaled gradient direction is not acceptable.
            # finally, calculate the point along the dogleg
            # at which the quadratic is minimized.
            bnorm = enorm(n, qtb)
            temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta)
            temp = (temp - (delta / qnorm) * (sgnorm / delta)**2
                    + np.sqrt((temp - (delta / qnorm))**2
                    + (1. - (delta / qnorm)**2) * (1. - (sgnorm / delta)**2)))
            alpha = ((delta / qnorm) * (1. - (sgnorm / delta)**2)) / temp

    # form appropriate convex combination of the gauss-newton
    # direction and the scaled gradient direction.
    temp = (1. - alpha) * min(sgnorm, delta)
    x = temp * wa1 + alpha * x
    return x


@jit(nopython=True, cache=True)
def r1updt(m, n, s, u, v):
    """
     subroutine r1updt

     given an m by n lower trapezoidal matrix s, an m-vector u,
     and an n-vector v, the problem is to determine an
     orthogonal matrix q such that

                   t
           (s + u*v )*q

     is again lower trapezoidal.

     this subroutine determines q as the product of 2*(n - 1)
     transformations

           gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)

     where gv(i), gw(i) are givens rotations in the (i,n) plane
     which eliminate elements in the i-th and n-th planes,
     respectively. q itself is not accumulated, rather the
     information to recover the gv, gw rotations is returned.

     the subroutine statement is

       subroutine r1updt(m,n,s,ls,u,v,w,sing)

     where

       m is a positive integer input variable set to the number
        of rows of s.

       n is a positive integer input variable set to the number
         of columns of s. n must not exceed m.

       s is an array of length ls. on input s must contain the lower
         trapezoidal matrix s stored by columns. on output s contains
         the lower trapezoidal matrix produced as described above.

       ls is a positive integer input variable not less than
         (n*(2*m-n+1))/2.

       u is an input array of length m which must contain the
         vector u.

       v is an array of length n. on input v must contain the vector
         v. on output v(i) contains the information necessary to
         recover the givens rotation gv(i) described above.

       w is an output array of length m. w(i) contains information
         necessary to recover the givens rotation gw(i) described
         above.

     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more,
     john l. nazareth
    """
    # giant is the largest magnitude
    giant = np.finfo(np.float64).max

    # initialize the diagonal element pointer.
    jj = (n * (2 * m - n + 1)) // 2 - (m - n) - 1
    nm1 = n - 1

    # move the nontrival part of the last column of s into w.
    w = np.empty(m)
    l = jj
    for i in range(nm1, m):
        w[i] = s[l]
        l += 1

    # rotate the vector v into a multiple of the n-th unit vector
    # in such a way that a spike is introduced into w.
    if nm1 >= 1:
        for nmj in range(nm1):
            j = nm1 - nmj - 1
            jj -= (m - j)
            w[j] = 0.
            if v[j] == 0.:
                continue

            # determine a givens rotation which eliminates the
            # j-th element of v.
            if abs(v[nm1]) >= abs(v[j]):
                tan = v[j] / v[nm1]
                cos = 0.5 / np.sqrt(0.25 + 0.25 * tan**2)
                sin = cos * tan
                tau = sin
            else:
                cotan = v[nm1] / v[j]
                sin = 0.5 / np.sqrt(0.25 + 0.25 * cotan**2)
                cos = sin * cotan
                if abs(cos) * giant > 1.:
                    tau = 1. / cos
                else:
                    tau = 1.


            # apply the transformation to v and store the information
            # necessary to recover the givens rotation.
            v[nm1] = sin * v[j] + cos * v[nm1]
            v[j] = tau

            # apply the transformation to s and extend the spike in w.
            l = jj
            for i in range(j, m):
                temp = cos * s[l] - sin * w[i]
                w[i] = sin * s[l] + cos * w[i]
                s[l] = temp
                l += 1

    # add the spike from the rank 1 update to w.
    w += v[nm1] * u

    # eliminate the spike.
    if nm1 >= 1:
        for j in range(nm1):
            if w[j] != 0.:
                # determine a givens rotation which eliminates the
                # j-th element of the spike.
                if abs(s[jj]) >= abs(w[j]):
                    tan = w[j] / s[jj]
                    cos = 0.5 / np.sqrt(0.25 + 0.25 * tan**2)
                    sin = cos * tan
                    tau = sin
                else:
                    cotan = s[jj] / w[j]
                    sin = 0.5 / np.sqrt(0.25 + 0.25 * cotan**2)
                    cos = sin * cotan
                    if abs(cos) * giant > 1.:
                        tau = 1. / cos
                    else:
                        tau = 1.

                # apply the transformation to s and reduce the spike in w.
                l = jj
                for i in range(j, m):
                    temp = cos * s[l] + sin * w[i]
                    w[i] = -sin * s[l] + cos * w[i]
                    s[l] = temp
                    l += 1

                # store the information necessary to recover the
                # givens rotation.
                w[j] = tau

            # test for zero diagonal elements in the output s.
            jj += (m - j)

    # move w back into the last column of the output s.
    l = jj
    for i in range(nm1, m):
        s[l] = w[i]
        l += 1
    return s, v, w


@jit(nopython=True, cache=True)
def r1mpyq(n, a, v, w):
    """
     subroutine r1mpyq

     given an m by n matrix a, this subroutine computes a*q where
     q is the product of 2*(n - 1) transformations

           gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)

     and gv(i), gw(i) are givens rotations in the (i,n) plane which
     eliminate elements in the i-th and n-th planes, respectively.
     q itself is not given, rather the information to recover the
     gv, gw rotations is supplied.

     the subroutine statement is

       subroutine r1mpyq(m,n,a,lda,v,w)

     where

       m is a positive integer input variable set to the number
         of rows of a.

       n is a positive integer input variable set to the number
         of columns of a.

       a is an m by n array. on input a must contain the matrix
         to be postmultiplied by the orthogonal matrix q
         described above. on output a*q has replaced a.

       v is an input array of length n. v(i) must contain the
         information necessary to recover the givens rotation gv(i)
         described above.

       w is an input array of length n. w(i) must contain the
         information necessary to recover the givens rotation gw(i)
         described above.

     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more
    """
    # apply the first set of givens rotations to a.
    nm1 = n - 1
    if (nm1 < 1):
        return a
    for nmj in range(nm1):
        j = n - nmj - 2
        if abs(v[j]) > 1.:
            cos = 1. / v[j]
        if abs(v[j]) > 1.:
            sin = np.sqrt(1. - cos**2)
        if abs(v[j]) <= 1.:
            sin = v[j]
        if abs(v[j]) <= 1.:
            cos = np.sqrt(1. - sin**2)
        temp = cos * a[j] - sin * a[nm1]
        a[nm1] = sin * a[j] + cos * a[nm1]
        a[j] = temp

    # apply the second set of givens rotations to a.
    for j in range(nm1):
        if abs(w[j]) > 1.:
            cos = 1. / w[j]
        if abs(w[j]) > 1.:
            sin = np.sqrt(1. - cos**2)
        if abs(w[j]) <= 1.:
            sin = w[j]
        if abs(w[j]) <= 1.:
            cos = np.sqrt(1. - sin**2)
        temp = cos * a[j] + sin * a[nm1]
        a[nm1] = -sin * a[j] + cos * a[nm1]
        a[j] = temp
    return a


@jit(nopython=True, cache=True)
def qrfac(m, n, a):
    """
     subroutine qrfac

     this subroutine uses householder transformations with column
     pivoting (optional) to compute a qr factorization of the
     m by n matrix a. that is, qrfac determines an orthogonal
     matrix q, a permutation matrix p, and an upper trapezoidal
     matrix r with diagonal elements of nonincreasing magnitude,
     such that a*p = q*r. the householder transformation for
     column k, k = 1,2,...,min(m,n), is of the form

                           t
           i - (1/u(k))*u*u

     where u has zeros in the first k-1 positions. the form of
     this transformation and the method of pivoting first
     appeared in the corresponding linpack subroutine.

     the subroutine statement is

       subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)

     where

       m is a positive integer input variable set to the number
         of rows of a.

       n is a positive integer input variable set to the number
         of columns of a.

       a is an m by n array. on input a contains the matrix for
         which the qr factorization is to be computed. on output
         the strict upper trapezoidal part of a contains the strict
         upper trapezoidal part of r, and the lower trapezoidal
         part of a contains a factored form of q (the non-trivial
         elements of the u vectors described above).

       lda is a positive integer input variable not less than m
         which specifies the leading dimension of the array a.

       pivot is a logical input variable. if pivot is set true,
         then column pivoting is enforced. if pivot is set false,
         then no column pivoting is done.

       ipvt is an integer output array of length lipvt. ipvt
         defines the permutation matrix p such that a*p = q*r.
         column j of p is column ipvt(j) of the identity matrix.
         if pivot is false, ipvt is not referenced.

       lipvt is a positive integer input variable. if pivot is false,
         then lipvt may be as small as 1. if pivot is true, then
         lipvt must be at least n.

       rdiag is an output array of length n which contains the
         diagonal elements of r.

       acnorm is an output array of length n which contains the
         norms of the corresponding columns of the input matrix a.
         if this information is not needed, then acnorm can coincide
         with rdiag.

       wa is a work array of length n. if pivot is false, then wa
         can coincide with rdiag.

     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more
    """
    # compute the initial column norms and initialize several arrays.
    acnorm = np.empty(n)
    for j in range(n):
        acnorm[j] = enorm(m, np.array([a[j, 1]]))
    rdiag = acnorm.copy()

    # reduce a to r with householder transformations.
    minmn = min(m, n)
    for j in range(minmn):
        # compute the householder transformation to reduce the
        # j-th column of a to a multiple of the j-th unit vector.
        ajnorm = enorm(m-j, np.array([a[j, j]]))
        if (ajnorm != 0.):
            if (a[j, j] < 0.):
                ajnorm = -ajnorm
            for i in range(j, m):
                a[j, i] /= ajnorm
            a[j, j] += 1.

            # apply the transformation to the remaining columns
            # and update the norms.
            if (n >= j+2):
                jp1 = j + 1
                for k in range(jp1, n):
                    ssum = 0.
                    for i in range(j, m):
                        ssum += a[j, i] * a[k, i]
                    temp = ssum / a[j, j]
                    for i in range(j, m):
                        a[k, i] -= temp * a[j, i]
        rdiag[j] = -ajnorm
    return a, rdiag, acnorm


@jit(nopython=True, cache=True)
def qform(m, n, q):
    """
     subroutine qform

     this subroutine proceeds from the computed qr factorization of
     an m by n matrix a to accumulate the m by m orthogonal matrix
     q from its factored form.

     the subroutine statement is

       subroutine qform(m,n,q,ldq,wa)

     where

       m is a positive integer input variable set to the number
         of rows of a and the order of q.

       n is a positive integer input variable set to the number
         of columns of a.

       q is an m by m array. on input the full lower trapezoid in
         the first min(m,n) columns of q contains the factored form.
         on output q has been accumulated into a square matrix.

       ldq is a positive integer input variable not less than m
         which specifies the leading dimension of the array q.

       wa is a work array of length m.


     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more
    """
    # zero out upper triangle of q in the first min(m,n) columns.
    minmn = min(m, n)
    if (minmn >= 2):
        for j in range(1, minmn):
            for i in range(j):
                q[j, i] = 0.

    # initialize remaining columns to those of the identity matrix.
    if (m >= n+1):
        for j in range(n, m):
            for i in range(m):
                q[j, i] = 0.
            q[j, j] = 1.

    # accumulate q from its factored form.
    wa = np.empty(m)
    for l in range(minmn):
        k = minmn - l - 1
        for i in range(k, m):
            wa[i] = q[k, i]
            q[k, i] = 0.
        q[k, k] = 1.
        if (wa[k] != 0.):
            for j in range(k, m):
                ssum = 0.
                for i in range(k, m):
                    ssum += q[j, i] * wa[i]
                temp = ssum / wa[k]
                for i in range(k, m):
                    q[j, i] -= temp * wa[i]
    return q


@jit(nopython=True, cache=True)
def enorm(n, x):
    """
     function enorm

     given an n-vector x, this function calculates the
     euclidean norm of x.

     the euclidean norm is computed by accumulating the sum of
     squares in three different sums. the sums of squares for the
     small and large components are scaled so that no overflows
     occur. non-destructive underflows are permitted. underflows
     and overflows do not occur in the computation of the unscaled
     sum of squares for the intermediate components.
     the definitions of small, intermediate and large components
     depend on two constants, rdwarf and rgiant. the main
     restrictions on these constants are that rdwarf**2 not
     underflow and rgiant**2 not overflow. the constants
     given here are suitable for every known computer.

     the function statement is

       double precision function enorm(n,x)

     where

       n is a positive integer input variable.

       x is an input array of length n.

     argonne national laboratory. minpack project. march 1980.
     burton s. garbow, kenneth e. hillstrom, jorge j. more
    """
    if np.any(np.isnan(x)):
        return np.nan
    rdwarf = 3.834e-20
    rgiant = 1.304e19
    s1 = 0.
    s2 = 0.
    s3 = 0.
    x1max = 0.
    x3max = 0.
    agiant = rgiant / n
    n = len(x)
    for i in range(n):
        xabs = abs(x[i])
        if (xabs <= rdwarf or xabs >= agiant):
            if (xabs > rdwarf):
                # sum for large components.
                if (xabs > x1max):
                    s1 = 1. + s1 * (x1max / xabs)**2
                    x1max = xabs
                else:
                    s1 += (xabs / x1max)**2

            # sum for small components.
            elif (xabs > x3max):
                s3 = 1. + s3 * (x3max / xabs)**2
                x3max = xabs
            elif (xabs != 0.):
                s3 += (xabs / x3max)**2
        else:
            # sum for intermediate components.
            s2 += xabs**2

    # calculation of norm.
    if s1 != 0.:
        enorm = x1max * np.sqrt(s1 + (s2 / x1max) / x1max)
    elif s2 != 0.:
        if s2 >= x3max:
            enorm = np.sqrt(s2 * (1. + (x3max / s2) * (x3max * s3)))
        else:
            enorm = np.sqrt(x3max * ((s2 / x3max) + (x3max * s3)))
    else:
        enorm = x3max * np.sqrt(s3)
    return enorm


@jit(nopython=True, cache=True)
def triu_indices(n):
    start = range(n)
    length = np.arange(1, n+1).sum()
    ys = np.empty(length, dtype=np.int64)
    for i in start:
        stop = i * n + (n - i)
        y = np.arange(i, n)
        y += i * n
        ys[i*n:stop] = y
    return ys
