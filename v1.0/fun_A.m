
function action = fun_A(X,Xleft1,hX,dim_obs,M,Y,Rm,Rf)

kern2 = Xleft1 - hX;
kern2 = Rf/(2*M)*sumsqr(kern2(:,1:M-1));

kern1 = Rm/(2*M)*sumsqr(X(dim_obs,:) - Y);

action = kern1 + kern2;

end

