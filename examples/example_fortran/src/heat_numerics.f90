! heat_numerics.f90
! Simple 1-D heat-equation numerics used to demonstrate TransJAX.

module heat_numerics
  implicit none

  ! Physical / numerical parameters
  real, parameter :: DEFAULT_ALPHA = 0.01   ! thermal diffusivity
  real, parameter :: DEFAULT_DX    = 0.1    ! grid spacing
  real, parameter :: DEFAULT_DT    = 0.001  ! time step

  contains

  ! ------------------------------------------------------------------
  ! dot_product_1d  — dot product of two real vectors
  ! ------------------------------------------------------------------
  function dot_product_1d(a, b, n) result(s)
    integer, intent(in) :: n
    real,    intent(in) :: a(n), b(n)
    real :: s
    integer :: i
    s = 0.0
    do i = 1, n
      s = s + a(i) * b(i)
    end do
  end function dot_product_1d

  ! ------------------------------------------------------------------
  ! tridiag_matvec  — multiply a tridiagonal matrix by a vector
  !   lower(:)  sub-diagonal  (indices 2..n)
  !   diag(:)   main diagonal (indices 1..n)
  !   upper(:)  super-diagonal(indices 1..n-1)
  ! ------------------------------------------------------------------
  subroutine tridiag_matvec(lower, diag, upper, x, y, n)
    integer, intent(in)  :: n
    real,    intent(in)  :: lower(n), diag(n), upper(n), x(n)
    real,    intent(out) :: y(n)
    integer :: i

    y(1) = diag(1)*x(1) + upper(1)*x(2)
    do i = 2, n-1
      y(i) = lower(i)*x(i-1) + diag(i)*x(i) + upper(i)*x(i+1)
    end do
    y(n) = lower(n)*x(n-1) + diag(n)*x(n)
  end subroutine tridiag_matvec

  ! ------------------------------------------------------------------
  ! euler_step  — one explicit-Euler step of the 1-D heat equation
  !   u_new(i) = u(i) + alpha * dt/dx^2 * (u(i+1) - 2*u(i) + u(i-1))
  ! ------------------------------------------------------------------
  subroutine euler_step(u, u_new, n, alpha, dx, dt)
    integer, intent(in)  :: n
    real,    intent(in)  :: u(n), alpha, dx, dt
    real,    intent(out) :: u_new(n)
    real    :: r
    integer :: i

    r = alpha * dt / (dx * dx)
    u_new(1) = u(1)  ! fixed left boundary
    do i = 2, n-1
      u_new(i) = u(i) + r * (u(i+1) - 2.0*u(i) + u(i-1))
    end do
    u_new(n) = u(n)  ! fixed right boundary
  end subroutine euler_step

end module heat_numerics
