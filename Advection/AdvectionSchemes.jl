
function Upwind(Tc,Told,Tc_ex,dTdxp,dTdxm,Vxp,Vxm,Vyp,Vym,Vzp,Vzm,dx,dy,dz,dt,order)

@. Told = Tc;
  for io=1:order
      @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
      @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
      @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
      @. dTdxp = 1.0/dx*(Tc_ex[3:end-0,2:end-1,2:end-1] - Tc_ex[2:end-1,2:end-1,2:end-1])
      @. dTdxm = 1.0/dx*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[1:end-2,2:end-1,2:end-1])
      @. Tc    = Tc - dt*(Vxp*dTdxm + Vxm*dTdxp);
  end
  @. Tc = (1.0/order)*Tc + (1.0-(1.0/order))*Told;

  @. Told = Tc
  for io=1:order
      @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
      @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
      @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
      @. dTdxp = 1.0/dy*(Tc_ex[2:end-1,3:end-0,2:end-1] - Tc_ex[2:end-1,2:end-1,2:end-1])
      @. dTdxm = 1.0/dy*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[2:end-1,1:end-2,2:end-1])
      @. Tc    = Tc - dt*(Vyp*dTdxm + Vym*dTdxp);
  end
  @. Tc = (1.0/order)*Tc + (1.0-(1.0/order))*Told;

  @. Told = Tc;
  for io=1:order
      @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
      @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
      @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
      @. dTdxp = 1.0/dz*(Tc_ex[2:end-1,2:end-1,3:end-0] - Tc_ex[2:end-1,2:end-1,2:end-1])
      @. dTdxm = 1.0/dz*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[2:end-1,2:end-1,1:end-2])
      @. Tc    = Tc - dt*(Vzp*dTdxm + Vzm*dTdxp);
  end
  @. Tc = (1.0/order)*Tc + (1.0-1.0/order)*Told;

end
