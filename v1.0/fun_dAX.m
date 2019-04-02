
function dAX = fun_dAX(X,Xu1,Xd1,Xd2,Xleft1,hX,...
    eyeD,eyeDleft2,eyeDleft1,eyeDright1,D,dim_obs,M,Y,dt,Rm,Rf,scaling)

GX = eyeDleft1.*permute(Xu1 - Xd2,[1 3 2]) + ...
    (eyeDright1 - eyeDleft2).*permute(Xd1,[1 3 2]) - eyeD;

kern2 = permute(X-circshift(hX,1,2),[1 3 2]).*(eyeD - dt/2*GX);
kern2 = Rf/M*permute(sum(kern2,1),[2 3 1]);
kern2(:,1) = 0;

kern3 = permute(Xleft1-hX,[1 3 2]).*(eyeD + dt/2*GX);
kern3 = -Rf/M*permute(sum(kern3,1),[2 3 1]);
kern3(:,M) = 0;

kern1 = zeros(D,M);
kern1(dim_obs,:) = Rm/M*(X(dim_obs,:) - Y);

dAX = scaling*(kern1 + kern2 + kern3);

end

