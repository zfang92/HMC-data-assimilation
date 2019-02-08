
% This program takes the Precision Annealing procedure and uses HMC as the
% optimizer to deal with the state and parameter estimation problem in the
% Lorenz 96 chaotic dynamical system. The data was prepared such that the
% data points are equally spaced and "measurement noise" was added to all 
% the dimensions.

% Self-categorization according to "Development note - HMC Lorenz96.docx":
% RK2 action
% No preannealing

% In particular, this version (v0.1) explores the full PA procedure and
% imposes some stopping criterion within each beta.

clear, clc;
codename = mfilename;

%--------------------------------------------------------------------------
% Tunable global parameters and switches

% Number of dimensions for Lorenz 96
D = 20;
% Number of observed dimensions, corresponds to L in literature
% We always observe the first Dobs dimensions
Dobs = 12;
% Number of training time steps
M = 200;

% Define Rm and the Rf ladder
% If one only wants to test on one beta value, simply change Rf0 to the
% desired value and make betamax equal 1
Rm = 1;
Rf0 = 1e6;                                                    %<----------
alpha = 1.2;                                                   
betamax = 1;                                                 %<----------

% First, try topping after a fixed number of iterations
niter = 1e2;                                                   %<----------

% HMC hyperparameters: integrater step size, integration time, and masses 
% (for X_observed, X_unobserved, f, respectively)
eps = 1e-4;                                                    
L = 50;                                                        
mass = [1e1,1e0,1e0];                                          %<----------

% Temperature schedule for simulated annealing
Te = exp(-1e0*(0:niter-1));
% Te = ones(1,niter); % this can be used to check the acceptance rate

% Switch for plotting Acttion vs. beta at the end. Another can be found at 
% the end of program for plotting Action vs. iteration in a chosen beta
plot_Action_vs_beta = 0;                                       
% Switch for saving Action vs. beta plot at the end
save_Action_vs_beta = 0;                                       

% Switch for saving data at the end
savedata = 0;                                                  

%--------------------------------------------------------------------------
% Data preparation and initialization

% Load Lorenz 96 data
load(['C:/Users/zfang/Downloads/data/Lorenz96/' ...
    'L96_D' num2str(D) '_nu8.17_sig0.4_dt0.001_deltat0.025_RK4.mat']);
% Y is the L-by-M matrix of observations
% We always observe the first Dobs dimensions
Y = Data(1:Dobs,1:M);
% Read time interval from data
dt = delta_t;

% Initialize the state variables in observed dim as states == observations
% Initialize those unobserved dim as ~Uniform(-10,10)
% X should be a D-by-M matrix
X_init = cat(1,Y,20*rand(D-Dobs,M)-10);
% Initialie the forcing paramter as ~Uniform(0,10)
nu_init = 10*rand;

% Assign the above predefined HMC masses and their square roots
% to each dimension. Don't miss the factor 2 in mass_~_sqrt
mass_X = cat(1,mass(1)*ones(Dobs,M),mass(2)*ones(D-Dobs,M));
mass_nu = mass(3);
mass_X_sqrt = cat(1,...
    sqrt(2*mass(1))*ones(Dobs,M),sqrt(2*mass(2))*ones(D-Dobs,M));
mass_nu_sqrt = sqrt(2*mass(3));

% Initialize some vectorized dirac delta functions to be used repetitively
% Below are D-by-D matrices of $\delta_{i-1,j}$, $\delta_{i-2,j}$,
% $\delta_{i+1,j}$, and $\delta_{ij}$, respectively
eyeDleft1 = circshift(eye(D),-1,2);
eyeDleft2 = circshift(eye(D),-2,2);
eyeDright1 = circshift(eye(D),1,2);
eyeD = eye(D);

%--------------------------------------------------------------------------
% Some initializations for HMC

% Define the Rf ladder
Rf = Rf0*(alpha.^linspace(0,betamax-1,betamax));

% Initialize the solutions (which are argmin(Action(:,beta)) for each beta)
% Therefore, the solutions become argmin(Action(:,betamax)) when finished
X_sol = X_init;
nu_sol = nu_init;
% Initialize the final output cell array. In each row, there is one spot
% for X and nu, respectively
q_min = cell(betamax,2);

% Initialize Action matrix, each row stores the Actions within one beta
Action = zeros(betamax,niter+1);
% Initialize the minimized Action for each beta
Action_min = zeros(betamax,1);

% Percentage acceptance and percentage downhill (for the sampler)
Acceptance = zeros(betamax,1);
Downhill = zeros(betamax,1);

%--------------------------------------------------------------------------
% Hamiltonian Monte Carlo core algorithm

% Do annealing for each beta starting from the solution of the previous one
for beta = 1:betamax
    
    % Initialize states
    X0 = X_sol;
    nu0 = nu_sol;
    
    % Evaluate the starting Action under current beta
    [Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,hX] ...
        = fun_getPieces(X0,nu0,dt);
    Action(beta,1) = fun_A(X0,Xleft1,hX,Y,Dobs,M,Rm,Rf(beta));
    Action_min(beta) = Action(beta,1);
    
    fprintf('Start annealing for beta = %d... ',beta); tic;
    for n = 2:niter+1
        % Take the current q as starting point
        X = X0;
        nu = nu0;
        
        % Generate initial momenta from a multivariate normal distribution
        pX0 = cat(1,normrnd(0,sqrt(mass(1)),Dobs,M),...
            normrnd(0,sqrt(mass(2)),D-Dobs,M));
        pnu0 = normrnd(0,sqrt(mass(3)));
        
        % Get necessary pieces
        [Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,hX] ...
            = fun_getPieces(X,nu,dt);
        % Half step for the X momenta
        pX = pX0 - eps/2*fun_dAX(X,Xup1,Xdown2,Xdown1,Xleft1,...
            Zup1,Zdown2,Zdown1,hX,Y,eyeD,eyeDleft1,eyeDleft2,...
            eyeDright1,D,Dobs,M,dt,Rm,Rf(beta));
        % Half step for the nu momentum
        pnu = pnu0 - eps/2*fun_dAnu(Xleft1,Zup1,Zdown2,hX,M,dt,Rf(beta));
        
        % Simulate Hamiltonian dynamics
        for i = 1:L
            % Full step for the state variables
            X = X + eps*pX./mass_X;
            nu = nu + eps*pnu/mass_nu;
            
            % Get necessary pieces
            [Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,hX] ...
                = fun_getPieces(X,nu,dt);
            if i~=L
                % Full step for the X momenta except at end of trajectory
                pX = pX - eps*fun_dAX(X,Xup1,Xdown2,Xdown1,Xleft1,...
                    Zup1,Zdown2,Zdown1,hX,Y,eyeD,eyeDleft1,eyeDleft2,...
                    eyeDright1,D,Dobs,M,dt,Rm,Rf(beta));
                % Full step for the nu momentum except at end of trajectory
                pnu = pnu - eps*fun_dAnu(Xleft1,Zup1,Zdown2,hX,M,dt,Rf(beta));
            end
        end
        
        % Half step for the X momenta
        pX = pX - eps/2*fun_dAX(X,Xup1,Xdown2,Xdown1,Xleft1,...
            Zup1,Zdown2,Zdown1,hX,Y,eyeD,eyeDleft1,eyeDleft2,...
            eyeDright1,D,Dobs,M,dt,Rm,Rf(beta));
        % Half step for the nu momentum
        pnu = pnu - eps/2*fun_dAnu(Xleft1,Zup1,Zdown2,hX,M,dt,Rf(beta));
        
        % Calculate Action when simulation is done
        Action_candidate = fun_A(X,Xleft1,hX,Y,Dobs,M,Rm,Rf(beta));
        % Metropolis-Hastings acceptance/rejection rule
        if rand < exp((Action(beta,n-1) + sumsqr(pX0./mass_X_sqrt) ...
                + (pnu0/mass_nu_sqrt)^2 ...
                - Action_candidate - sumsqr(pX./mass_X_sqrt) ...
                - (pnu/mass_nu_sqrt)^2)/Te(n-1))
            % Accepted states and Action serve as the next starting point
            X0 = X;
            nu0 = nu;
            Action(beta,n) = Action_candidate;
            % Count acceptance rate
            Acceptance(beta) = Acceptance(beta) + 1;
        else
            % If rejected, maintain the same proposal states and Action
            Action(beta,n) = Action(beta,n-1);
        end
        
        % Check if the current proposal yields the lowest Action so far
        if Action(beta,n) < Action_min(beta)
            % If so, update the lowest Action and the corresponding states
            Action_min(beta) = Action(beta,n);
            % And update the corresponding states to the current ones
            X_sol = X0;
            nu_sol = nu0;
            % Count downhill rate
            Downhill(beta) = Downhill(beta) + 1;
        end
    end
    fprintf('Done. '); toc;
    
    % Record argmin(A(:,beta)) for the current beta
    q_min(beta,:) = {X_sol,nu_sol};
    
    % Finalize these percentages
    Acceptance(beta) = Acceptance(beta)/niter;
    Downhill(beta) = Downhill(beta)/niter;
    
end
fprintf('\n');

%--------------------------------------------------------------------------
% Infer new variables and save useful variables

% Reference line (corresonding to no annealing done)
Action_init = zeros(betamax,1);
for beta = 1:betamax
    [Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,hX] ...
        = fun_getPieces(X_init,nu_init,dt);
    Action_init(beta) = fun_A(X_init,Xleft1,hX,Y,Dobs,M,Rm,Rf(beta));
end

ME = zeros(betamax,1); FE = zeros(betamax,1);
for beta = 1:betamax
    % Evaluate (measurement err)/Rm
    ME(beta) = 1/(2*M)*sumsqr(q_min{beta,1}(1:Dobs,:) - Y);
    % Evaluate (model err)/Rf
    [Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,hX] ...
        = fun_getPieces(q_min{beta,1},q_min{beta,2},dt);
    kern2 = (Xleft1 - hX);
    FE(beta) = 1/(2*M)*sumsqr(kern2(:,1:M-1));
end

mass_string = num2str(mass(1));
for j = 2:length(mass)
    mass_string = cat(2,mass_string,['-' num2str(mass(j))]);
end

% save data
if savedata == 1
    save([codename '_(' num2str(D) ')_(' num2str(Dobs) ')_(' ... 
        num2str(M) ')_(' num2str(alpha) ')_(' num2str(Rm) ')_(' ... 
        num2str(niter,'%10.0e') ')_(' num2str(eps,'%10.0e') ')_(' ... 
        num2str(L) ')_(' mass_string ').mat'],...
        'Acceptance','Action','Action_init','Action_min','alpha',...
        'betamax','codename','D','Dobs','Downhill','dt','eps','FE','L',...
        'M','mass','ME','niter','q_min','Rf','Rf0','Rm','Te');
end

%--------------------------------------------------------------------------
% Visualize the whole annealing process

if plot_Action_vs_beta == 1
    % Plot the whole Action vs. Rf process
    figure; ax1 = axes('position',[0.1 0.12 0.87 0.83]);
    loglog(ax1,Rf,Action_init,'k--','LineWidth',1); hold all;
    loglog(ax1,Rf,FE,'LineWidth',1.5); hold all;
    loglog(ax1,Rf,Action_min,'-','LineWidth',2); hold off;
    legend('A (X init)', 'model err./Rf',...
        'annealed A','Location','northwest');
    ylabel('Action'); xlabel('Rf');
    xlim(ax1,[min(Rf) max(Rf)]); grid on;
    
    if save_Action_vs_beta == 1
        % Name the plot as:
        % <code name>_<D>_<Dobs>_<M>_<alpha>_<Rm>_<niter>_<eps>_<L>_<mass>
        saveas(gcf,['0. Action plots\' codename '_(' num2str(D) ')_(' ... 
            num2str(Dobs) ')_(' num2str(M) ')_(' num2str(alpha) ')_(' ... 
            num2str(Rm) ')_(' num2str(niter,'%10.0e') ')_(' ... 
            num2str(eps,'%10.0e') ')_(' num2str(L) ')_(' ...
            mass_string ').fig']);
        saveas(gcf,['0. Action plots\' codename '_(' num2str(D) ')_(' ... 
            num2str(Dobs) ')_(' num2str(M) ')_(' num2str(alpha) ')_(' ... 
            num2str(Rm) ')_(' num2str(niter,'%10.0e') ')_(' ... 
            num2str(eps,'%10.0e') ')_(' num2str(L) ')_(' ...
            mass_string ').png']);
    end
end

%--------------------------------------------------------------------------
% Plot what happened in a particular beta value

beta_choice = 1;                                               %<----------
if beta_choice ~= 0
    % Plot what happened within the chosen beta value
    figure; ax2 = axes('position',[0.1 0.12 0.87 0.83]);
    loglog(ax2,Action(beta_choice,:),'LineWidth',1.5); grid on;
    xlim([1 niter+1]); %ylim([1e0 1.5e5]); %%%
    ylabel('Action'); xlabel('Iteration');
    title(['\beta = ' num2str(beta_choice) ', Rf = ' ...
        num2str(Rf(beta_choice),'%10.0e')]);
end

%--------------------------------------------------------------------------
% Functions to evaluate the Action and its derivatives

function [Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,hX] ...
    = fun_getPieces(X,nu,dt)

Xup1 = circshift(X,-1,1);
Xdown1 = circshift(X,1,1);
Xdown2 = circshift(X,2,1);
Xleft1 = circshift(X,-1,2);

Z = X + dt/2*((Xup1 - Xdown2).*Xdown1 - X + nu);
Zup1 = circshift(Z,-1,1);
Zdown1 = circshift(Z,1,1);
Zdown2 = circshift(Z,2,1);

hX = X + dt*((Zup1 - Zdown2).*Zdown1 - Z + nu);

end
%------------------------------------

function dAX = fun_dAX(X,Xup1,Xdown2,Xdown1,Xleft1,Zup1,Zdown2,Zdown1,...
    hX,Y,eyeD,eyeDleft1,eyeDleft2,eyeDright1,D,Dobs,M,dt,Rm,Rf)

GX = eyeDleft1.*permute(Xup1 - Xdown2,[1 3 2]) + ...
    (eyeDright1 - eyeDleft2).*permute(Xdown1,[1 3 2]) - eyeD;
GZ = eyeDleft1.*permute(Zup1 - Zdown2,[1 3 2]) + ...
    (eyeDright1 - eyeDleft2).*permute(Zdown1,[1 3 2]) - eyeD;

GZGX = zeros(D,D,M);
for k = 1:M
    GZGX(:,:,k) = GZ(:,:,k)*GX(:,:,k);
end

T = eyeD + dt*GZ + dt^2/2*GZGX;
kern3 = -Rf/M*permute(sum(permute(Xleft1-hX,[1 3 2]).*T,1),[2 3 1]);
kern3(:,M) = 0;

kern1 = zeros(D,M);
kern1(1:Dobs,:) = Rm/M*(X(1:Dobs,:) - Y);

kern2 = Rf/M*(X - circshift(hX,1,2));
kern2(:,1) = 0;

dAX = kern1 + kern2 + kern3;

end
%------------------------------------

function dAnu = fun_dAnu(Xleft1,Zup1,Zdown2,hX,M,dt,Rf)

kern = (Xleft1 - hX).*(dt + dt^2/2*(Zup1 - Zdown2 - 1));
dAnu = -Rf/M*sum(sum(kern(:,1:M-1)));

end
%------------------------------------

function Action = fun_A(X,Xleft1,hX,Y,Dobs,M,Rm,Rf)

kern1 = Rm/(2*M)*sumsqr(X(1:Dobs,:) - Y);

kern2 = (Xleft1 - hX);
kern2 = Rf/(2*M)*sumsqr(kern2(:,1:M-1));

Action = kern1 + kern2;

end















