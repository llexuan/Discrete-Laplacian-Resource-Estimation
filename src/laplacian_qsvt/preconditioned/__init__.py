"""Nested block encoding of the 1D preconditioned periodic Laplacian.

The conditioned operator L_cond is diagonal in the Fourier basis with
eigenvalues sinc^2(pi m_k / N). It is block encoded by an inner QSVT that
applies the sinc^2 polynomial P_2d to U_X, an exact LCU block encoding of
X = diag(2 m_k / N). The outer matrix-inversion QSVT is unchanged.
"""
