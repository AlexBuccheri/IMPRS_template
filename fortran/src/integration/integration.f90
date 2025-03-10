module integration_m
    use, intrinsic :: iso_fortran_env
    implicit none
    private
    public :: trapezium_integration

contains

    function trapezium_integration(fx, dx) result(integral)
        real(real64), intent(in) :: fx(:) !< Function evaluated at x
        real(real64), intent(in) :: dx
        real(real64) :: integral

        integer :: i, n_points

        n_points = size(fx)
        integral = 0.0_real64

        do i = 2, n_points - 1
            integral = integral + fx(i)
        enddo

        integral = integral + 0.5 * (fx(1) + fx(n_points))
        integral = integral * dx

    end function trapezium_integration

end module integration_m
