
% This code generates multidimentional Lorenz 96 time series.

clear, clc;
codename = mfilename;

%--------------------------------------------------------------------------
% Tunable global parameters and switches

% Dimension of Lorenz 96
D = 20;

% Discretized length for the downsampled time series
len = 1e3;

% Forcing term in Lorenz 96
nu = 8.17;

% Noise switch and noise level
AddNoise = 1;
sigma = 0.4;

% Integrator step size
dt = 0.025;

% Down sampled interval at the end
delta_t = 0.025;

%   2: second order Runge-Kutta
%   4: fourth order Runge-Kutta
% 999: trapezoidal rule
integrator = 999;

% Switch for saving down sampled data at the end
savedata = 0;

% Plot
plotdata = 1;

%--------------------------------------------------------------------------
% Initialize some vectorized dirac delta functions to be used repetitively

% Below are D-by-D matrices of $\delta_{i-1,j}$, $\delta_{i-2,j}$,
% $\delta_{i+1,j}$, and $\delta_{ij}$, respectively
eyeDleft1 = circshift(eye(D),-1,2);
eyeDleft2 = circshift(eye(D),-2,2);
eyeDright1 = circshift(eye(D),1,2);
eyeD = eye(D);

%--------------------------------------------------------------------------
% Generate and visualize Lorenz96 time series

% Raw length (doubled for transient), delta_t/dt must be an integer
rawlen = 2*len*delta_t/dt;

% Initialize the trajectory
Y = ones(D,rawlen);
% Give it a deterministic perturbation
Y(1,1) = 0.01;

tic;
if integrator == 2
    
    % Second order Runge-Kutta
    for k = 1:rawlen-1
        Y(:,k+1) = Y(:,k) + dt*F(Y(:,k)+dt/2*F(Y(:,k),nu),nu);
    end
    
elseif integrator == 4
    
    % Fourth order Runge-Kutta
    for k = 1:rawlen-1
        u1 = dt*F(Y(:,k),nu);
        u2 = dt*F(Y(:,k)+u1/2,nu);
        u3 = dt*F(Y(:,k)+u2/2,nu);
        u4 = dt*F(Y(:,k)+u3,nu);
        Y(:,k+1) = Y(:,k) + u1/6 + u2/3 + u3/3 + u4/6;
    end
    
elseif integrator == 999
    
    % Trapezoidal rule
    for k = 1:rawlen-1
        % Initial guess with the Euler method
        x_old = Y(:,k) + dt*F(Y(:,k),nu);
        
        % First iteration for Newton-Raphson
        g_x = dt/2*F(x_old,nu) - x_old + dt/2*F(Y(:,k),nu) + Y(:,k);
        delta_x = Jacobian(x_old,...
            eyeD,eyeDleft2,eyeDleft1,eyeDright1,dt)\g_x;
        x_new = x_old - delta_x;
        x_old = x_new;
        
        % Iterate until the corrections reach the lower bound
        while sum(abs(delta_x)) > 1e-14
            g_x = dt/2*F(x_old,nu) - x_old + dt/2*F(Y(:,k),nu) + Y(:,k);
            delta_x = Jacobian(x_old,...
                eyeD,eyeDleft2,eyeDleft1,eyeDright1,dt)\g_x;
            x_new = x_old - delta_x;
            x_old = x_new;
        end
        Y(:,k+1) = x_new;
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
if (savedata == 1) && (AddNoise == 1) && (integrator ~= 999)
    save(['C:\Users\zfang\Downloads\data\Lorenz96\L96_D' num2str(D) ... 
        '_nu' num2str(nu) '_sig' num2str(sigma) '_dt' num2str(dt) ... 
        '_deltat' num2str(delta_t) '_RK' num2str(integrator) '.mat'],...
        'Data','delta_t');
elseif (savedata == 1) && (AddNoise ~= 1) && (integrator ~= 999)
    save(['C:\Users\zfang\Downloads\data\Lorenz96\L96_D' num2str(D) ... 
        '_nu' num2str(nu) '_noiseless_dt' num2str(dt) ... 
        '_deltat' num2str(delta_t) '_RK' num2str(integrator) '.mat'],...
        'Data','delta_t');
end

if (savedata == 1) && (AddNoise == 1) && (integrator == 999)
    save(['C:\Users\zfang\Downloads\data\Lorenz96\L96_D' num2str(D) ... 
        '_nu' num2str(nu) '_sig' num2str(sigma) '_dt' num2str(dt) ... 
        '_deltat' num2str(delta_t) '_trapez.mat'],...
        'Data','delta_t');
elseif (savedata == 1) && (AddNoise ~= 1) && (integrator == 999)
    save(['C:\Users\zfang\Downloads\data\Lorenz96\L96_D' num2str(D) ... 
        '_nu' num2str(nu) '_noiseless_dt' num2str(dt) ... 
        '_deltat' num2str(delta_t) '_trapez.mat'],...
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
%------------------------------------

function J = Jacobian(x,eyeD,eyeDleft2,eyeDleft1,eyeDright1,dt)

kernel = eyeDleft1.*(circshift(x,-1,1) - circshift(x,2,1)) + ...
    (eyeDright1 - eyeDleft2).*circshift(x,1,1);

J = dt/2*kernel - (dt/2 + 1)*eyeD;

end
