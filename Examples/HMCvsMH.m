
%% Global setting

% This is the current version, which includes exporting ascii data.

clear, clc;

% Tunable global parameters,
niter = 5e2;
nburn = floor(0.4*niter);
% for HMC,
eps = 1e-2;
L = 50;
m = [1 1];
% for MCMC
std = 0.1;

a = 8; b = 6; c = 5; d = 3;

% [0 0] means starting at the center, [0 0.8] means starting at the edge
S0 = [0 0.8];

% Action and its derivatives
A = @(X,Y) (a*X.^2 + b*Y.^2 - c).^2 + d*Y.^2;
dxA = @(X,Y) 4*a*X.*(a*X.^2 + b*Y.^2 - c);
dyA = @(X,Y) 4*b*Y.*(a*X.^2 + b*Y.^2 - c) + 2*d*Y;

[X,Y] = meshgrid(linspace(-1,1,500),linspace(-1,1,500));
% contour(X,Y,exp(-A(X,Y))); xlabel('x'); ylabel('y','Rotation',0);


%% Hamiltonian Monte Carlo

S_hmc = zeros(niter+1,2);
S_hmc(1,:) = S0;

Accept_hmc = 0;
Action = zeros(niter+1,1);
Action(1) = A(S_hmc(1,1),S_hmc(1,2));

tic;
fprintf('Running... ');
for n = 2:niter+1
    x = S_hmc(n-1,1);
    y = S_hmc(n-1,2);
    
    px0 = normrnd(0,sqrt(m(1)));
    py0 = normrnd(0,sqrt(m(2)));
    
    px = px0 - eps/2*dxA(x,y);
    py = py0 - eps/2*dyA(x,y);
    
    for i = 1:L
        x = x + eps*px/m(1);
        y = y + eps*py/m(2);
        
        if i~=L
            px = px - eps*dxA(x,y);
            py = py - eps*dyA(x,y);
        end
    end
    
    px = px - eps/2*dxA(x,y);
    py = py - eps/2*dyA(x,y);
    
    Action_candidate = A(x,y);
    
    if rand <= exp(Action(n-1)+px0^2/(2*m(1))+py0^2/(2*m(2)) ...
            - Action_candidate-px^2/(2*m(1))-py^2/(2*m(2)))
        S_hmc(n,1) = x;
        S_hmc(n,2) = y;
        Action(n) = Action_candidate;
        Accept_hmc = Accept_hmc + 1;
    else
        S_hmc(n,:) = S_hmc(n-1,:);
        Action(n) = Action(n-1);
    end
end
fprintf('done. ');
toc;

Accept_hmc = Accept_hmc/niter;

subplot(1,2,1);
contour(X,Y,exp(-A(X,Y))); hold all;
plot(S_hmc(nburn+1:end,1),S_hmc(nburn+1:end,2),'r.'); hold off;
xlim([-1 1]); ylim([-1 1]); title('Hamiltonian MC');
xlabel('x'); ylabel('y','Rotation',0);


%% Metropolis-Hastings Monte Carlo

S_mh = zeros(niter+1,2);
S_mh(1,:) = S0;

Accept_mh = 0;
Action = zeros(niter+1,1);
Action(1) = A(S_mh(1,1),S_mh(1,2));

tic;
fprintf('Running... ');
for n = 2:niter+1
    x_c = normrnd(S_mh(n-1,1),std);
    y_c = normrnd(S_mh(n-1,2),std);
    
    Action_candidate = A(x_c,y_c);
    
    if rand <= exp(Action(n-1) - Action_candidate)
        S_mh(n,1) = x_c;
        S_mh(n,2) = y_c;
        Action(n) = Action_candidate;
        Accept_mh = Accept_mh + 1;
    else
        S_mh(n,:) = S_mh(n-1,:);
        Action(n) = Action(n-1);
    end
end
fprintf('done. ');
toc;

Accept_mh = Accept_mh/niter;

subplot(1,2,2);
contour(X,Y,exp(-A(X,Y))); hold all;
plot(S_mh(nburn+1:end,1),S_mh(nburn+1:end,2),'r.'); hold off;
xlim([-1 1]); ylim([-1 1]); title('Random-walk MC');
xlabel('x'); ylabel('y','Rotation',0);


%% Output ascii data

% clear;
% load(['E:\GoogleDrive\0. MLDI\p1-2. PA on Lorenz 96\' ...
%     '8. Formatted data for draft paper\20190324_(HMC_vs_MH)\' ...
%     'HMCvsMH_good_example_clustered.mat']);
% clc;
% 
% [X,Y] = meshgrid(linspace(-1,1,300),linspace(-1,1,300));
% % contour(X,Y,exp(-A(X,Y)));
% 
% contour_data = zeros(90000,3);
% for ind_y = 1:300
%     for ind_x = 1:300
%         ind = (ind_y-1)*300 + ind_x;
%         
%         contour_data(ind,1) = X(ind_y,ind_x);
%         contour_data(ind,2) = Y(ind_y,ind_x);
%         contour_data(ind,3) = exp(-A(X(ind_y,ind_x),Y(ind_y,ind_x)));
%     end
% end
% 
% save('contour_data.dat','contour_data','-ascii','-double');
% 
% HMC_data = S_hmc(nburn+1:end,:);
% save('HMC_data.dat','HMC_data','-ascii','-double');
% 
% Metropolis_data = S_mh(nburn+1:end,:);
% save('Metropolis_data.dat','Metropolis_data','-ascii','-double');
