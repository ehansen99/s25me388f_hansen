program matrixsolve7

  ! Link to LAPACK
  ! On my machine I had to link to the Apple installed framework
  ! To link I referenced Stack Overflow
  ! mpif90 -framework Accelerate -o p7.o matrixsolve7.f90

  use mpi

  ! Lapack read from Netlib

  implicit none

  real(kind=8), allocatable :: sup(:),sub(:),diag(:)
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
  ALLOCATE(diag(n))
  ALLOCATE(sub(n-1))
  ALLOCATE(sup(n-1))
  ALLOCATE(b(n,1))
  ALLOCATE(ipiv(n))

  b = 0.0
  b(1,1) = 1.0

  sub = -0.9
  diag = 1.0
  sup = 0.0

  print *, "a and b assigned"
  t1 = MPI_WTIME()

  ! This matrix is tridiagonal; just the superdiagonal is zero
  
  CALL dgtsv(n,1,sub,diag,sup,b,n,info)
  t2 = MPI_WTIME()

  ! For n = 100 output for end of assignment
  print *, "Time",t2-t1
  if (n.eq.100) then
     print *, "Writing"
     OPEN(unit=100,access="sequential",file="xout.csv",action="write",form="formatted",status="replace")
     do i = 1,n
        write(100,*) b(i,1),","
     enddo
     CLOSE(100)
  endif
  
     
  
  print *, "LAPACK info",info

  DEALLOCATE(diag)
  DEALLOCATE(sub)
  DEALLOCATE(sup)
  DEALLOCATE(b)

  CALL MPI_FINALIZE(ierr)

end program matrixsolve7
