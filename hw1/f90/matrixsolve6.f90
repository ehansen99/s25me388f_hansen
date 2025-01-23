program matrixsolve6

  ! mpif90 -framework Accelerate -o p6.o matrixsolve6.f90
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
  ALLOCATE(a(n,n))
  ALLOCATE(b(n,1))
  ALLOCATE(ipiv(n))

  a = 0.0
  b = 1.0

  do i = 1,n
     ! Upper triangular elements
     if (i.ne.n) then
        do j = i+1,n
           a(i,j) = -1.0/(j-i)
        enddo
     endif

     ! Lower triangular elements
     if (i.ne.1) then
        do j = 1,i-1
           a(i,j) = -1.0/(i-j)
        enddo
     endif
     
     a(i,i) = 1.0 - sum(a(i,:))
     
  enddo

  ! Full matrix routine

  print *, "a and b assigned"
  t1 = MPI_WTIME()

  CALL dgesv(n,1,a,n,ipiv,b,n,info)
  t2 = MPI_WTIME()
  
  print *, "Time",t2-t1
  if (n.eq.10) print *, b
  
  print *, "LAPACK info",info

  ! This matrix is also symmetric, and based on Mathematica could be positive definite, so we attempt those routines
  
  a = 0.0
  b = 1.0
  
  do i = 1,n
     if (i.ne.n) then
        do j = i+1,n
           a(i,j) = -1.0/(j-i)
        enddo
     endif
     if (i.ne.1) then
        do j = 1,i-1
           a(i,j) = -1.0/(i-j)
        enddo
     endif

     a(i,i) = 1.0 - sum(a(i,:))

  enddo

  print *, "a and b assigned"
  t1 = MPI_WTIME()

  CALL dposv("U",n,1,a,n,b,n,info)
  t2 = MPI_WTIME()

  print *, "Time",t2-t1
  if (n.eq.10) print *, b
  print *, "LAPACK Info",info

  DEALLOCATE(a)
  DEALLOCATE(b)

  CALL MPI_FINALIZE(ierr)

end program matrixsolve6
