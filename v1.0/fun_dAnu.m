
function dAnu = fun_dAnu(Xleft1,hX,M,dt,Rf,scaling)

kern = (Xleft1 - hX)*dt;
dAnu = scaling*(-Rf/M*sum(sum(kern(:,1:M-1))));

end

