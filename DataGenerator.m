
% This code generates multidimentional Lorenz 96 time series. It performs 
% Runge-Kutta integration. After that, it down samples the generated series 
% to a user-defined interval.

clear, clc;
codename = mfilename;

%--------------------------------------------------------------------------
% Tunable global parameters and switches

% Dimension of Lorenz 96
D = 20;

% Discretized length for the downsampled time series
len = 5e3;

% Forcing term in Lorenz 96
nu = 8.17;

% Noise switch and noise level
AddNoise = 0;
sigma = 0.4;

% Integrator step size
dt = 0.02;

% Down sampled interval at the end
delta_t = 0.02;

% Runge-Kutta integrator
RK = 2;

% Switch for saving down sampled data at the end
savedata = 0;

% Plot
plotdata = 1;

%--------------------------------------------------------------------------
% Generate and visualize Lorenz96 time series

% Raw length (doubled for transient), delta_t/dt must be an integer
rawlen = 2*len*delta_t/dt;

% Initialize the trajectory
Y = ones(D,rawlen);
% Give it a deterministic perturbation
Y(1,1) = 0.01;

tic;
if RK == 2
    % Second order Runge-Kutta
    for k = 1:rawlen-1
        Y(:,k+1) = Y(:,k) + dt*F(Y(:,k)+dt/2*F(Y(:,k),nu),nu);
    end
elseif RK == 4
    % Fourth order Runge-Kutta
    for k = 1:rawlen-1
        u1 = dt*F(Y(:,k),nu);
        u2 = dt*F(Y(:,k)+u1/2,nu);
        u3 = dt*F(Y(:,k)+u2/2,nu);
        u4 = dt*F(Y(:,k)+u3,nu);
        Y(:,k+1) = Y(:,k) + u1/6 + u2/3 + u3/3 + u4/6;
    end
end
toc;

if plotdata == 1
    plot3(Y(1,:),Y(3,:),Y(5,:),'LineWidth',1); grid on;
    xlabel('$y_1$','Interpreter','latex');
    ylabel('$y_3$','Interpreter','latex');
    zlabel('$y_5$','Interpreter','latex','Rotation',0);
    title(['Lorenz 96 with $D=$ ' num2str(D)],'Interpreter','latex');
end

%--------------------------------------------------------------------------
% Post processing

% Burn transient, keep only the second half, delta_t/dt must be an integer
Y = Y(:,(len*delta_t/dt+1):end);

% Down sample, delta_t/dt must be an integer
indices = delta_t/dt*linspace(1,len,len);
Data = Y(:,indices);

% Normally, noise should only be added within the training window of the
% observed dimensions. However, for comparison sake we just add noise to
% every single data point in the matrix.
if AddNoise == 1
    Data = Data + normrnd(0,sigma,D,len);
end

% Save noisy or noiseless data
if (savedata == 1) && (AddNoise == 1)
    save(['C:\Users\zfang\Downloads\data\Lorenz96\L96_D' num2str(D) ... 
        '_nu' num2str(nu) '_sig' num2str(sigma) '_dt' num2str(dt) ... 
        '_deltat' num2str(delta_t) '_RK' num2str(RK) '.mat'],...
        'Data','delta_t');
elseif (savedata == 1) && (AddNoise ~= 1)
    save(['C:\Users\zfang\Downloads\data\Lorenz96\L96_D' num2str(D) ... 
        '_nu' num2str(nu) '_noiseless_dt' num2str(dt) ... 
        '_deltat' num2str(delta_t) '_RK' num2str(RK) '.mat'],...
        'Data','delta_t');
end

%--------------------------------------------------------------------------
% Define useful functions

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
L96 = L96+nu;

end
























