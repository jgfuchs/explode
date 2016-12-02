
void __kernel init_grid(
    __write_only image3d_t U,
    __write_only image3d_t P)
{

}

void __kernel advect_field(
    const float dt,
    __read_only image3d_t F,
    __read_only image3d_t Q,
    __write_only image3d_t Q_out)
{

}

void __kernel advect_scalar()
{

}

void __kernel project()
{

}
