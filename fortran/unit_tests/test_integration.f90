program test_integration
    use, intrinsic :: iso_fortran_env
    use integration_m
    implicit none

    real(real64), parameter :: pi = 3.14159265359_real64
    real(real64), allocatable :: fx(:), x(:)
    real(real64) :: dx, integral

    dx = 0.000001_real64

    ! Integrate [0, 2pi]
    call grid_real64_1d(0.0_real64, 2._real64 * pi, dx, x)
    allocate(fx, source=sin(x))
    write(*, *) 'Limits: ', x(1), x(size(x))
    integral = trapezium_integration(fx, dx)
    write(*, *) 'Expect integral_0^2pi sin(x) = 0: ', integral
    deallocate(fx)
    deallocate(x)


    ! Integrate [0, pi]
    call grid_real64_1d(0.0_real64, pi, dx, x)
    write(*, *) 'Limits: ', x(1), x(size(x))
    allocate(fx, source=sin(x))
    integral = trapezium_integration(fx, dx)
    write(*, *) 'Expect integral_0^pi sin(x) = 2: ', integral
    deallocate(fx)
    deallocate(x)

contains

    subroutine grid_real64_1d(a, b, dx, x)
        real(real64), intent(in) :: a, b !< limits
        real(real64), intent(in) :: dx   !< Line element

        real(real64), allocatable, intent(out) :: x(:)

        integer :: n_points, i

        n_points = int((b - a) / dx) + 1
        allocate(x(n_points))

        do i = 1, n_points
            x(i) = a + (i-1) * dx
        end do

    end subroutine grid_real64_1d

end program test_integration
