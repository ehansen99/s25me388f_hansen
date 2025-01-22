program matrixsolve1

  ! Link to LAPACK
  ! On my machine I had to link to the Apple installed framework
  ! To link I referenced Stack Overflow
  ! mpif90 -framework Accelerate -o ms1.o matrixsolve1.f90

  use mpi

  ! Lapack read from Netlib

  implicit none

  real(kind=8), allocatable :: sup(:),sub(:),diag(:)
  integer(kind=4) :: k,n
  real(kind=8), allocatable :: b(:,:),x(:,:)
  integer :: i,j
  integer :: info,ierr
  character(1) :: arg
  integer(kind=4), allocatable :: ipiv(:)
  real(kind=8) :: t1,t2,t3,t4

  ! From Michael Thorne's Homepage at University of Utah
  CALL GETARG(1,arg)
  READ(arg,*) k
  
  CALL MPI_INIT(ierr)
  
  n = 10**k
  ALLOCATE(diag(n))
  ALLOCATE(sub(n-1))
  ALLOCATE(sup(n-1))
  ALLOCATE(b(n,1))
  ALLOCATE(x(n,1))
  ALLOCATE(ipiv(n))

  b = 1.0

  sub = 0.0
  diag = 1.0
  sup = 0.0

  print *, "a and b assigned"
  t1 = MPI_WTIME()

  ! This matrix is tridiagonal; just the superdiagonal is zero
  
  CALL dgtsv(n,1,sub,diag,sup,b,n,info)

  t2 = MPI_WTIME()

  print *, "Time",t2-t1

  t1 = MPI_WTIME()

  ! The matrix A is just the identity so we know the solution will be the RHS

  b = 1.0
  
  x = b
  t2 = MPI_WTIME()
  
  print *, "Time",t2-t1
  if (n.eq.10) print *, b
  
  print *, "LAPACK info",info

  DEALLOCATE(diag)
  DEALLOCATE(sub)
  DEALLOCATE(sup)
  DEALLOCATE(b)

  CALL MPI_FINALIZE(ierr)

end program matrixsolve1
