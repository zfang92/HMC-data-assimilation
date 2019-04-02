
function L96 = F(x,nu)

% Extract dimensionality
D = length(x);
% Initialze output
L96 = zeros(D,1);

% Edge cases for i=1,2,D
L96(1) = (x(2)-x(D-1))*x(D) - x(1);
L96(2) = (x(3)-x(D))*x(1) - x(2);
L96(D) = (x(1)-x(D-2))*x(D-1) - x(D);
% All the rest dimensions
for d = 3:D-1
    L96(d) = (x(d+1)-x(d-2))*x(d-1) - x(d);
end

% Add forcing
L96 = L96 + nu;

end
