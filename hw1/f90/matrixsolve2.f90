program matrixsolve2

  ! mpif90 -framework Accelerate -o ms2.o matrixsolve2.f90
  use mpi

  ! Lapack read from Netlib

  implicit none

  real(kind=8), allocatable :: offdiag(:),diag(:)
  integer(kind=4) :: k,n
  real(kind=8), allocatable :: b(:)
  integer :: i,j
  integer :: info,ierr
  character(1) :: arg
  real(kind=8) :: t1,t2,t3,t4

  ! From Michael Thorne's Homepage at University of Utah
  CALL GETARG(1,arg)
  READ(arg,*) k
  
  CALL MPI_INIT(ierr)
  
  n = 10**k

  
  ! We're not doing to try using the full matrix due to memory constraints
  ! All diagonals except the main diagonal are constant, so 
  ! we only store the diagonal elements and subdiagonal values
  ! and will solve recursively

  
  ALLOCATE(offdiag(n))
  ALLOCATE(diag(n))
  ALLOCATE(b(n))

  offdiag(n) = 0.0
  do i = 1,n-1
     offdiag(i) = -1.0/(n-i)
  enddo

  print *, "Off Diagonal Assigned"

  do i = 0,n-1
     diag(i+1) = 1- sum(offdiag(n-i:n))
  enddo

  print *, diag

  print *, "Diagonal Elements Assigned"

  b = 1.0
  
  t1 = MPI_WTIME()
  b(1) = b(1)/diag(1)

  do i = 2,n
     b(i) = (b(i) - sum(offdiag(n-i+1:n-1)*b(1:i-1)))/diag(i)
  enddo

  t2 = MPI_WTIME()
  print *, "Time",t2-t1
  
  if (n.eq.10) print *, b

  DEALLOCATE(offdiag)
  DEALLOCATE(diag)
  DEALLOCATE(b)

  CALL MPI_FINALIZE(ierr)

end program matrixsolve2
