############################################################

@views function VxPlusMinus(Vxm::DatArray_k, Vxp::DatArray_k, Vx::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        ixi = ix + 1
        if ( @participate_a(Vxm) )
            if ( @West(Vx) < 0.00 ) @all(Vxm) = @West(Vx); end
            if ( @West(Vx) > 0.00 ) @all(Vxm) = 0.00; end
        end

        if ( @participate_a(Vxp) )
            if ( @East(Vx) > 0.00 ) @all(Vxp) = @East(Vx); end
            if ( @East(Vx) < 0.00 ) @all(Vxp) = 0.00;      end
        end
    end
    return nothing
end

@views function VyPlusMinus(Vym::DatArray_k, Vyp::DatArray_k, Vy::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        iyi = iy + 1
        if ( @participate_a(Vym) )
            if ( @South(Vy) < 0.00 ) @all(Vym) = @South(Vy); end
            if ( @South(Vy) > 0.00 ) @all(Vym) = 0.00;       end
        end
        if ( @participate_a(Vyp) )
            if ( @North(Vy) > 0.00  ) @all(Vyp) = @North(Vy); end
            if ( @North(Vy) < 0.00  ) @all(Vyp) = 0.00;       end
        end
    end
    return nothing
 end

@views function VzPlusMinus(Vzm::DatArray_k, Vzp::DatArray_k, Vz::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        izi = iz + 1
        if ( @participate_a(Vzm) )
            if ( @Back(Vz) < 0.00 ) @all(Vzm) = @Back(Vz); end
            if ( @Back(Vz) > 0.00 ) @all(Vzm) = 0.00;      end
        end
        if ( @participate_a(Vzp) )
            if ( @Front(Vz) > 0.00 ) @all(Vzp) = @Front(Vz); end
            if ( @Front(Vz) < 0.00 ) @all(Vzp) = 0.00;      end
        end
    end
    return nothing
end

 @views function dFdx_Weno5(dFdxi::DatArray_k, V1::DatArray_k, V2::DatArray_k, V3::DatArray_k, V4::DatArray_k, V5::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if (@participate_a(V1))  v1   = @all(V1); end
        if (@participate_a(V2))  v2   = @all(V2); end
        if (@participate_a(V3))  v3   = @all(V3); end
        if (@participate_a(V4))  v4   = @all(V4); end
        if (@participate_a(V5))  v5   = @all(V5); end
        if (@participate_a(V1))  p1   = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3; end
        if (@participate_a(V1))  p2   =-v2/6.0 + 5.0/6.0*v3 + v4/3.0;      end
        if (@participate_a(V1))  p3   = v3/3.0 + 5.0/6.0*v4 - v5/6.0;      end
        if (@participate_a(V1))  maxV = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2); end
        if (@participate_a(V1))  e    = 10^(-99) + 1e-6*maxV; end
        if (@participate_a(V1))  w1   = 13.0/12.0*(v1-2.0*v2+v3)^2 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2; end
        if (@participate_a(V1))  w2   = 13.0/12.0*(v2-2.0*v3+v4)^2 + 1.0/4.0*(v2-v4)^2; end
        if (@participate_a(V1))  w3   = 13.0/12.0*(v3-2.0*v4+v5)^2 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2; end
        if (@participate_a(V1))  w1   = 0.1/(w1+e)^2; end
        if (@participate_a(V1))  w2   = 0.6/(w2+e)^2; end
        if (@participate_a(V1))  w3   = 0.3/(w3+e)^2; end
        if (@participate_a(V1))  w    = (w1+w2+w3);   end
        if (@participate_a(V1))  w1   = w1/w; end
        if (@participate_a(V1))  w2   = w2/w; end
        if (@participate_a(V1))  w3   = w3/w; end
        # dFdxi[ix,iy,iz] = w1*p1 + w2*p2 + w3*p3
        if @participate_a(dFdxi) @all(dFdxi) = w1*p1 + w2*p2 + w3*p3; end
    end
return nothing
end

@views function Gradients_minus_x_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_a(v1) @all(v1) = 1.0/dx*( @in_xxx_xm2(Fc_exxx) -  @in_xxx_xm3(Fc_exxx) ); end
        if @participate_a(v2) @all(v2) = 1.0/dx*( @in_xxx_xm1(Fc_exxx) -  @in_xxx_xm2(Fc_exxx) ); end
        if @participate_a(v3) @all(v3) = 1.0/dx*(     @in_xxx(Fc_exxx) -  @in_xxx_xm1(Fc_exxx) ); end
        if @participate_a(v4) @all(v4) = 1.0/dx*( @in_xxx_xp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
        if @participate_a(v5) @all(v5) = 1.0/dx*( @in_xxx_xp2(Fc_exxx) -  @in_xxx_xp1(Fc_exxx) ); end
end
return nothing
end

@views function Gradients_plus_x_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_a(v1) @all(v1) = 1.0/dx*( @in_xxx_xp3(Fc_exxx) -  @in_xxx_xp2(Fc_exxx) ); end
        if @participate_a(v2) @all(v2) = 1.0/dx*( @in_xxx_xp2(Fc_exxx) -  @in_xxx_xp1(Fc_exxx) ); end
        if @participate_a(v3) @all(v3) = 1.0/dx*( @in_xxx_xp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
        if @participate_a(v4) @all(v4) = 1.0/dx*(     @in_xxx(Fc_exxx) -  @in_xxx_xm1(Fc_exxx) ); end
        if @participate_a(v5) @all(v5) = 1.0/dx*( @in_xxx_xm1(Fc_exxx) -  @in_xxx_xm2(Fc_exxx) ); end
end
return nothing
end

@views function Gradients_minus_y_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_a(v1) @all(v1) = 1.0/dy*( @in_xxx_ym2(Fc_exxx) -  @in_xxx_ym3(Fc_exxx) ); end
        if @participate_a(v2) @all(v2) = 1.0/dy*( @in_xxx_ym1(Fc_exxx) -  @in_xxx_ym2(Fc_exxx) ); end
        if @participate_a(v3) @all(v3) = 1.0/dy*(     @in_xxx(Fc_exxx) -  @in_xxx_ym1(Fc_exxx) ); end
        if @participate_a(v4) @all(v4) = 1.0/dy*( @in_xxx_yp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
        if @participate_a(v5) @all(v5) = 1.0/dy*( @in_xxx_yp2(Fc_exxx) -  @in_xxx_yp1(Fc_exxx) ); end
end
return nothing
end

@views function Gradients_plus_y_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_a(v1) @all(v1) = 1.0/dy*( @in_xxx_yp3(Fc_exxx) -  @in_xxx_yp2(Fc_exxx) ); end
        if @participate_a(v2) @all(v2) = 1.0/dy*( @in_xxx_yp2(Fc_exxx) -  @in_xxx_yp1(Fc_exxx) ); end
        if @participate_a(v3) @all(v3) = 1.0/dy*( @in_xxx_yp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
        if @participate_a(v4) @all(v4) = 1.0/dy*(     @in_xxx(Fc_exxx) -  @in_xxx_ym1(Fc_exxx) ); end
        if @participate_a(v5) @all(v5) = 1.0/dy*( @in_xxx_ym1(Fc_exxx) -  @in_xxx_ym2(Fc_exxx) ); end
end
return nothing
end

@views function Gradients_minus_z_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_a(v1) @all(v1) = 1.0/dz*( @in_xxx_zm2(Fc_exxx) -  @in_xxx_zm3(Fc_exxx) ); end
        if @participate_a(v2) @all(v2) = 1.0/dz*( @in_xxx_zm1(Fc_exxx) -  @in_xxx_zm2(Fc_exxx) ); end
        if @participate_a(v3) @all(v3) = 1.0/dz*(     @in_xxx(Fc_exxx) -  @in_xxx_zm1(Fc_exxx) ); end
        if @participate_a(v4) @all(v4) = 1.0/dz*( @in_xxx_zp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
        if @participate_a(v5) @all(v5) = 1.0/dz*( @in_xxx_zp2(Fc_exxx) -  @in_xxx_zp1(Fc_exxx) ); end
    end
return nothing
end

@views function Gradients_plus_z_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_a(v1) @all(v1) = 1.0/dz*( @in_xxx_zp3(Fc_exxx) -  @in_xxx_zp2(Fc_exxx) ); end
        if @participate_a(v2) @all(v2) = 1.0/dz*( @in_xxx_zp2(Fc_exxx) -  @in_xxx_zp1(Fc_exxx) ); end
        if @participate_a(v3) @all(v3) = 1.0/dz*( @in_xxx_zp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
        if @participate_a(v4) @all(v4) = 1.0/dz*(     @in_xxx(Fc_exxx) -  @in_xxx_zm1(Fc_exxx) ); end
        if @participate_a(v5) @all(v5) = 1.0/dz*( @in_xxx_zm1(Fc_exxx) -  @in_xxx_zm2(Fc_exxx) ); end
    end
return nothing
end

@views function  Boundaries_x_Weno5( Fc_exxx::DatArray_k, Fc::DatArray_k, type_W::Integer, val_W::DAT, type_E::Integer, val_E::DAT, nx::Integer, ny::Integer, nz::Integer )

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_ixxx(Fc_exxx) @in_xxx(Fc_exxx) = @all(Fc); end
        if (type_W ==0 ) # Neumann
            if (ix==1 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[3,iy-3,iz-3]; end
            if (ix==2 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[2,iy-3,iz-3]; end
            if (ix==3 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[1,iy-3,iz-3]; end
        end

        if (type_W ==1 ) # Dirichlet
            if (ix==1 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_W - Fc[3,iy-3,iz-3]; end
            if (ix==2 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_W - Fc[2,iy-3,iz-3]; end
            if (ix==3 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_W - Fc[1,iy-3,iz-3]; end
        end

        if (type_W ==2 ) # Periodic
            if (ix==1 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[nx-2,iy-3,iz-3]; end
            if (ix==2 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[nx-1,iy-3,iz-3]; end
            if (ix==3 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[nx-0,iy-3,iz-3]; end
        end

        if (type_E ==0 ) # Neumann
            if (ix==nx+6-0 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[nx-2,iy-3,iz-3]; end
            if (ix==nx+6-1 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[nx-1,iy-3,iz-3]; end
            if (ix==nx+6-2 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[nx-0,iy-3,iz-3]; end
        end

        if (type_E ==1 ) # Dirichlet
            if (ix==nx+6-0 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_E - Fc[nx-2,iy-3,iz-3]; end
            if (ix==nx+6-1 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_E - Fc[nx-1,iy-3,iz-3]; end
            if (ix==nx+6-2 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_E - Fc[nx-0,iy-3,iz-3]; end
        end

        if (type_E ==2 ) # Periodic
            if (ix==nx+6-0 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[3,iy-3,iz-3]; end
            if (ix==nx+6-1 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[2,iy-3,iz-3]; end
            if (ix==nx+6-2 && iy>3 && iy<ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[1,iy-3,iz-3]; end
        end

    end
return nothing
end

@views function  Boundaries_y_Weno5( Fc_exxx::DatArray_k, Fc::DatArray_k, type_S::Integer, val_S::DAT, type_N::Integer, val_N::DAT, nx::Integer, ny::Integer, nz::Integer )

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_ixxx(Fc_exxx) @in_xxx(Fc_exxx) = @all(Fc); end
        if (type_S ==0 ) # Neumann
            if (ix>3 && ix<nx+6-2 && iy==1 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,3,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,2,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==3 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,1,iz-3]; end
        end

        if (type_S ==1 ) # Dirichlet
            if (ix>3 && ix<nx+6-2 && iy==1 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_S - Fc[ix-3,3,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_S - Fc[ix-3,2,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==3 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_S - Fc[ix-3,1,iz-3]; end
        end

        if (type_S ==2 ) # Periodic
            if (ix>3 && ix<nx+6-2 && iy==1 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,ny-2,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,ny-1,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==3 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,ny-0,iz-3]; end
        end

        if (type_N ==0 ) # Neumann
            if (ix>3 && ix<nx+6-2 && iy==ny+6-0 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,ny-2,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==ny+6-1 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,ny-1,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,ny-0,iz-3]; end
        end

        if (type_N ==1 ) # Dirichlet
            if (ix>3 && ix<nx+6-2 && iy==ny+6-0 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_N - Fc[ix-3,ny-2,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==ny+6-1 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_N - Fc[ix-3,ny-1,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_N - Fc[ix-3,ny-0,iz-3]; end
        end

        if (type_N ==2 ) # Periodic
            if (ix>3 && ix<nx+6-2 && iy==ny+6-0 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,3,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==ny+6-1 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,2,iz-3]; end
            if (ix>3 && ix<nx+6-2 && iy==ny+6-2 && iz>3 && iz<nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,1,iz-3]; end
        end

    end
return nothing
end

@views function  Boundaries_z_Weno5( Fc_exxx::DatArray_k, Fc::DatArray_k, type_B::Integer, val_B::DAT, type_F::Integer, val_F::DAT, nx::Integer, ny::Integer, nz::Integer )

    @threadids_or_loop (nx+6,ny+6,nz+6) begin
        ixiii = ix + 3
        iyiii = iy + 3
        iziii = iz + 3
        if @participate_ixxx(Fc_exxx) @in_xxx(Fc_exxx) = @all(Fc); end
        if (type_B ==0 ) # Neumann
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,3]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,2]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==3) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,1]; end
        end

        if (type_B ==1 ) # Dirichlet
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==1) Fc_exxx[ix,iy,iz] = 2*val_B - Fc[ix-3,iy-3,3]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==2) Fc_exxx[ix,iy,iz] = 2*val_B - Fc[ix-3,iy-3,2]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==3) Fc_exxx[ix,iy,iz] = 2*val_B - Fc[ix-3,iy-3,1]; end
        end

        if (type_B ==2 ) # Periodic
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,nz-2]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,nz-1]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==3) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,nz-0]; end
        end

        if (type_F ==0 ) # Neumann
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-0) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,nz-2]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,nz-1]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,nz-0]; end
        end

        if (type_F ==1 ) # Dirichlet
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-0) Fc_exxx[ix,iy,iz] = 2*val_F - Fc[ix-3,iy-3,nz-2]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-1) Fc_exxx[ix,iy,iz] = 2*val_F - Fc[ix-3,iy-3,nz-1]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-2) Fc_exxx[ix,iy,iz] = 2*val_F - Fc[ix-3,iy-3,nz-0]; end
        end

        if (type_F ==2 ) # Periodic
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-0) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,3]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-1) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,2]; end
            if (ix>3 && ix<nx+6-2 && iy>3 && iy<ny+6-2 && iz==nz+6-2) Fc_exxx[ix,iy,iz] = Fc[ix-3,iy-3,1]; end
        end

    end
return nothing
end

@views function  Advect(Fc::DatArray_k, Vxp::DatArray_k, dTdxm::DatArray_k, Vxm::DatArray_k, dTdxp::DatArray_k, dt::DAT, nx::Integer, ny::Integer, nz::Integer )

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if @participate_a(Fc) @all(Fc) = @all(Fc) - dt*(@all(Vxp)*@all(dTdxm) + @all(Vxm)*@all(dTdxp)); end
    end
return nothing
end

@views function  TimeAveraging(Fc::DatArray_k, Fold::DatArray_k, order::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if @participate_a(Fc) @all(Fc) = (1.0/order)*@all(Fc) + (1.0-1.0/order)*@all(Fold); end
    end
return nothing
end

@views function  ArrayEqualArray(F1::DatArray_k, F2::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if @participate_a(F1) @all(F1) = @all(F2); end
    end
return nothing
end

@views function  InitialCondition(Tc::DatArray_k, xc2::DatArray_k, yc2::DatArray_k, zc2::DatArray_k, x0::DAT, y0::DAT, z0::DAT, sig2::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if (USE_GPU==1)
            if @participate_a(Tc) @all(Tc) = CUDAnative.exp( -(@all(xc2)-x0)^2/sig2 - (@all(yc2)-y0)^2/sig2 - (@all(zc2)-z0)^2/sig2); end
        end
        if (USE_GPU==0)
            if @participate_a(Tc) @all(Tc) = exp( -(@all(xc2)-x0)^2/sig2 - (@all(yc2)-y0)^2/sig2 - (@all(zc2)-z0)^2/sig2); end
        end
    end
return nothing
end


############################################################

# @views function MainWeno()
# # General
# order  = 2.0;
# # # Boundaries
# # BC_type_W = 0
# # BC_val_W  = 0.0
# # BC_type_E = 0
# # BC_val_E  = 0.0
#
# # Boundaries
# BC_type_W = 1
# BC_val_W  = 1.0
# BC_type_E = 1
# BC_val_E  = 1.0
#
# BC_type_S = 1
# BC_val_S  = 1.0
# BC_type_N = 1
# BC_val_N  = 1.0
#
# BC_type_B = 1
# BC_val_B  = 1.0
# BC_type_F = 1
# BC_val_F  = 1.0
#
# # Domain
# xmin     = -0.5;  xmax = 0.5;  Lx = xmax - xmin;
# ymin     = -0.5;  ymax = 0.5;  Ly = ymax - ymin;
# zmin     = -0.5;  zmax = 0.5;  Lz = zmax - zmin;
# # Numerics
# nout     = 10;
# nx       = fact*32-6;
# ny       = fact*32-6;
# nz       = fact*32-6;
# Nix      = USE_MPI ? nx_g() : nx;                                               #SO: TODO: this is obtained from the global_grid for MPI.
# Niy      = USE_MPI ? ny_g() : ny;                                               #SO: this is obtained from the global_grid.
# Niz      = USE_MPI ? nz_g() : nz;
# dx       = Lx/Nix;                                                              #SO: why not dx = Lx/(nx-1) or dx = Lx/(Nix-1) respectively
# dy       = Ly/Niy;
# dz       = Lz/Niz;
# # Grid
# xc  = LinRange(xmin+dx/2, xmax-dx/2, nx)
# yc  = LinRange(ymin+dy/2, ymax-dy/2, ny)
# zc  = LinRange(xmin+dz/2, zmax-dz/2, nz)
# xce = LinRange(xmin-dx/2, xmax+dx/2, nx+2)
# yce = LinRange(ymin-dy/2, ymax+dy/2, ny+2)
# zce = LinRange(xmin-dz/2, zmax+dz/2, nz+2)
# xv  = LinRange(xmin, xmax, nx+1)
# yv  = LinRange(ymin, ymax, ny+1)
# zv  = LinRange(xmin, zmax, nz+1)
# (xce2,yce2,zce2) = ([x for x=xce,y=yce,z=zce], [y for x=xce,y=yce,z=zce], [z for x=xce,y=yce,z=zce]); #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
# (xc2,yc2,zc2)    = ([x for x=xc,y=yc,z=zc], [y for x=xc,y=yc,z=zc], [z for x=xc,y=yc,z=zc]); #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
# (xv2,yv2,zv2)    = ([x for x=xv,y=yv,z=zv], [y for x=xv,y=yv,z=zv], [z for x=xv,y=yv,z=zv]);
#
# (xvx,yvx,zvx)    = ([x for x=xv,y=yc,z=zc], [y for x=xv,y=yc,z=zc], [z for x=xv,y=yc,z=zc]);
# (xvy,yvy,zvy)    = ([x for x=xc,y=yv,z=zc], [y for x=xc,y=yv,z=zc], [z for x=xc,y=yv,z=zc]);
#
# @printf("Grid was set up!\n")
# # Initial conditions
# Vx       =  myones(nx+1,ny+0,nz+0);
# Vy       =  myones(nx+0,ny+1,nz+0);
# #Vy       =  myzeros(nx+0,ny+1,nz+0);
# # Vx       = Array(Vx);
# # Vy       = Array(Vy);
# # Vx      .=  yvx;#myones(nx+1,ny+0,nz+0);
# # Vy      .= -xvy;#myzeros(nx+0,ny+1,nz+0);
# # Vx       = DatArray(Vx);
# # Vy       = DatArray(Vy);
# Vz       =  myones(nx+0,ny+0,nz+1);
# Tc       =  myzeros(nx+0,ny+0,nz+0);
# # x0       = 0.9
# # y0       = 0.5*(ymin+ymax)
# # z0       = 0.5*(zmin+zmax)
# x0       = -0.25
# y0       = -0.25
# z0       = -0.25
# sig2     = 0.005
# Tbg      = 1;
# #Define kernel launch params (used only if USE_GPU set true).
# cuthreads = (32, 8, 1 )
# cublocks  = ( 1, 4, 32).*fact
# # cublocks = ceil.(Int, (nx+2, ny+2, nz+2)./cuthreads)
# # here need to make a kernel for InitialCondition as exp does not work outside of kernels...
# xc2_d     =  myzeros(nx+0,ny+0,nz+0);
# yc2_d     =  myzeros(nx+0,ny+0,nz+0);
# zc2_d     =  myzeros(nx+0,ny+0,nz+0);
# xc2_d     = Array(xc2_d)
# yc2_d     = Array(yc2_d)
# zc2_d     = Array(zc2_d)
# xc2_d    .= xc2 # DEBUG: @. not working - needs .=
# yc2_d    .= yc2
# zc2_d    .= zc2
# xc2_d     = DatArray(xc2_d)
# yc2_d     = DatArray(yc2_d)
# zc2_d     = DatArray(zc2_d)
# @kernel cublocks cuthreads InitialCondition(Tc, xc2_d, yc2_d, zc2_d, x0, y0, z0, sig2, nx, ny, nz);  @devicesync();
# # Compute Courant criteria
# dt = 0.25*min(dx,dy,dz) / max( maximum_g(abs.(Vx)), maximum_g(abs.(Vy)), maximum_g(abs.(Vz)))
# # Upwind velocities
# Vxm     =  myzeros(nx+0,ny+0,nz+0);
# Vxp     =  myzeros(nx+0,ny+0,nz+0);
# @kernel cublocks cuthreads VxPlusMinus(Vxm, Vxp, Vx, nx, ny, nz); @devicesync();
# Vym     =  myzeros(nx+0,ny+0,nz+0);
# Vyp     =  myzeros(nx+0,ny+0,nz+0);
# @kernel cublocks cuthreads VyPlusMinus(Vym, Vyp, Vy, nx, ny, nz); @devicesync();
# Vzm     =  myzeros(nx+0,ny+0,nz+0);
# Vzp     =  myzeros(nx+0,ny+0,nz+0);
# @kernel cublocks cuthreads VzPlusMinus(Vzm, Vzp, Vz, nx, ny, nz);  @devicesync();
# # Pre-processing
# Tc_exxx =  myzeros(nx+6,ny+6,nz+6);
# Tc_ex   =  myzeros(nx+2,ny+2,nz+2);
# v1      =  myzeros(nx+0,ny+0,nz+0);
# v2      =  myzeros(nx+0,ny+0,nz+0);
# v3      =  myzeros(nx+0,ny+0,nz+0);
# v4      =  myzeros(nx+0,ny+0,nz+0);
# v5      =  myzeros(nx+0,ny+0,nz+0);
# dTdxp   =  myzeros(nx+0,ny+0,nz+0);
# dTdxm   =  myzeros(nx+0,ny+0,nz+0);
# Told    =  myzeros(nx+0,ny+0,nz+0);
# mT_evo  =  zeros(DAT, Int(ceil(nt/nout)+1) ); ii=1;
# time    = 0
# # Time loop
# for it=0:nt
#
#     time += dt
#     @printf("Time step #%04d\n", it)
#
#     if Upwind == 0 # --- WENO-5
#
        # # Advect in x direction
        # @kernel cublocks cuthreads ArrayEqualArray(Told, Tc, nx, ny, nz);                                                     @devicesync();
        # for io=1:order
        #     @kernel cublocks cuthreads Boundaries_x_Weno5(Tc_exxx, Tc, BC_type_W, BC_val_W, BC_type_E, BC_val_E, nx, ny, nz); @devicesync();
        #     @kernel cublocks cuthreads Gradients_minus_x_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz);          @devicesync();
        #     @kernel cublocks cuthreads dFdx_Weno5(dTdxm, v1, v2, v3, v4, v5, nx, ny, nz);                                     @devicesync();
        #     @kernel cublocks cuthreads Gradients_plus_x_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz);           @devicesync();
        #     @kernel cublocks cuthreads dFdx_Weno5(dTdxp, v1, v2, v3, v4, v5, nx, ny, nz);                                     @devicesync();
        #     @kernel cublocks cuthreads Advect(Tc, Vxp, dTdxm, Vxm, dTdxp, dt, nx, ny, nz);                                    @devicesync();
        # end
        # @kernel cublocks cuthreads TimeAveraging(Tc, Told, order, nx, ny, nz);                                                @devicesync();

        # # Advect in y direction
        # @kernel cublocks cuthreads ArrayEqualArray(Told, Tc, nx, ny, nz);                                                     @devicesync();
        # for io=1:order
        #     @kernel cublocks cuthreads Boundaries_y_Weno5(Tc_exxx, Tc, BC_type_S, BC_val_S, BC_type_N, BC_val_N, nx, ny, nz); @devicesync();
        #     @kernel cublocks cuthreads Gradients_minus_y_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz);          @devicesync();
        #     @kernel cublocks cuthreads dFdx_Weno5(dTdxm, v1, v2, v3, v4, v5, nx, ny, nz);                                     @devicesync();
        #     @kernel cublocks cuthreads Gradients_plus_y_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz);           @devicesync();
        #     @kernel cublocks cuthreads dFdx_Weno5(dTdxp, v1, v2, v3, v4, v5, nx, ny, nz);                                     @devicesync();
        #     @kernel cublocks cuthreads Advect(Tc, Vyp, dTdxm, Vym, dTdxp, dt, nx, ny, nz);                                    @devicesync();
        # end
        # @kernel cublocks cuthreads TimeAveraging(Tc, Told, order, nx, ny, nz);                                                @devicesync();
        #
        # # Advect in z direction
        # @kernel cublocks cuthreads ArrayEqualArray(Told, Tc, nx, ny, nz);                                                     @devicesync();
        # for io=1:order
        #     @kernel cublocks cuthreads Boundaries_z_Weno5(Tc_exxx, Tc, BC_type_B, BC_val_B, BC_type_F, BC_val_F, nx, ny, nz); @devicesync();
        #     @kernel cublocks cuthreads Gradients_minus_z_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz);          @devicesync();
        #     @kernel cublocks cuthreads dFdx_Weno5(dTdxm, v1, v2, v3, v4, v5, nx, ny, nz);                                     @devicesync();
        #     @kernel cublocks cuthreads Gradients_plus_z_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz);           @devicesync();
        #     @kernel cublocks cuthreads dFdx_Weno5(dTdxp, v1, v2, v3, v4, v5, nx, ny, nz);                                     @devicesync();
        #     @kernel cublocks cuthreads Advect(Tc, Vzp, dTdxm, Vzm, dTdxp, dt, nx, ny, nz);                                    @devicesync();
        # end
        # @kernel cublocks cuthreads TimeAveraging(Tc, Told, order, nx, ny, nz);                                                @devicesync();
#     end
#
#     # Check mean field
#     if (mod(it,nout)==0)
#         Tc2 = Tc; if (USE_GPU==1) Tc2 = Array(Tc); end
#         nT = mean_g(abs.(Tc2[:]))/sqrt(nx*ny*nz);
#         mT = maximum_g(abs.(Tc2[:]));
#         @printf("Norm of Tc = %2.2e\n", nT)
#         @printf("Max  of Tc = %2.2e\n", mT)
#         mT_evo[ii] = mT;
#         ii = ii+1;
#         if (Vizu == 1)
#         	X = Tc; if (USE_GPU==1) X = Array(Tc); end
#         	p1 = heatmap(xc, zc, transpose(X[:,Int(ceil(ny/2)),: ]),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
#             p3 = heatmap(xc, zc, transpose(X[:,:,1 ]),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
#         	p2 = plot(mT_evo)
#         	display(plot( p1, p2, p3))
#         	@printf("Image sliced at z index %d over nx = %d, ny = %d, nz = %d\n", Int(ceil(nz/2)), nx, ny, nz)
#         end
#     end
#
#     if ( Save==1 && mod(it,nout)==0 )
#     	filename = @sprintf("./WenoOutput%05d.h5", it)
#     	if isfile(filename)==1
#     		rm(filename)
#     	end
#         h5write(filename, "Tc", Array(Tc))
#         h5write(filename, "xc", Array(xc))
#         h5write(filename, "yc", Array(yc))
#         h5write(filename, "zc", Array(zc))
#     end
#
# end
#
# # Expected location of Gaussian center
# # display(x0 + time*1.0)
# # display(y0 + time*1.0)
#
# # if (Vizu == 1)
# #     X = Tc; if (USE_GPU==1) X = Array(Tc); end
# #     p1 = heatmap(xc, yc, transpose(X[:,:,Int(ceil(nz/2))]),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
# #     p2 = plot(mT_evo)
# #     display(plot( p1, p2))
# #     @printf("Image sliced at z index %d over nx = %d, ny = %d, nz = %d\n", Int(ceil(nz/2)), nx, ny, nz)
# # end
#
# end
#
# @time MainWeno()
