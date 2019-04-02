
function [Xu1,Xd1,Xd2,Xleft1,hX] = fun_getPieces(X,nu,dt)

Xu1 = circshift(X,-1,1);
Xd1 = circshift(X,1,1);
Xd2 = circshift(X,2,1);

Xleft1 = circshift(X,-1,2);

Xleft1u1 = circshift(Xleft1,-1,1);
Xleft1d1 = circshift(Xleft1,1,1);
Xleft1d2 = circshift(Xleft1,2,1);

hX = X + dt/2*((Xu1 - Xd2).*Xd1 - X + nu) ...
    + dt/2*((Xleft1u1 - Xleft1d2).*Xleft1d1 - Xleft1 + nu);

end

