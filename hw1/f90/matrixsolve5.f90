program matrixsolve5

  ! mpif90 -framework Accelerate -o p5.o matrixsolve5.f90
  ! to run, mpirun -n 1 ./p5.o k
  
  use mpi

  ! Lapack read from Netlib

  implicit none

  real(kind=8), allocatable :: a(:,:)
  integer(kind=4) :: k,n
  real(kind=8), allocatable :: b(:,:)
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
  ALLOCATE(a(16,n))
  ALLOCATE(b(n,1))
  ALLOCATE(ipiv(n))

  b = 1.0

  ! From dgbsv storage convention place number of subdiagonals above main array
  
  a(6,:) = -1.0
  a(7,:) = 0.0
  a(8,:) = 0.0
  a(9,:) = 0.0
  a(10,:) = -1.0
  a(11,:) = 5.0
  a(12,:) = -1.0
  a(13,:) = 0.0
  a(14,:) = 0.0
  a(15,:) = 0.0
  a(16,:) = -1.0

  print *, "a and b assigned"
  t1 = MPI_WTIME()

  CALL dgbsv(n,5,5,1,a,16,ipiv,b,n,info)
  
  t2 = MPI_WTIME()

  ! This works for all but k = 9 where memory seems to run out
  ! It's possible that this matrix is positive definite and the Cholesky solver
  ! would save enough memory for even k = 9 to work but I haven't been able to implement it yet
  ! Perhaps at a future date I could also apply a superfast O(N log(N)^2)
  ! Toeplitz solver e.g. https://doi.org/10.1016/0196-6774(80)90013-9 
  
  print *, "Time",t2-t1
  if (n.eq.10) print *, b
  
  print *, "LAPACK info",info

  a = 0.0
  b = 1.0

  DEALLOCATE(a)
  DEALLOCATE(b)
  DEALLOCATE(ipiv)

  CALL MPI_FINALIZE(ierr)

end program matrixsolve5
