
% clear, clc;
codename = mfilename;

%--------------------------------------------------------------------------
% Tunable global parameters and switches

% Number of dimensions for Lorenz 96
D = 20;
% Number of observed dimensions, corresponds to L in literature
Dobs = 8;
% Must in ascending order without repetition
%------------------------------------
% dim_obs = [1;2;4;6;8;10;12;14;15;17;19;20]; % 60 percent
% dim_obs = [1;3;4;7;9;11;13;15;17;19]; % 50 percent
dim_obs = [1;4;7;9;11;13;16;19]; % 40 percent
% dim_obs = [1;4;7;10;13;16;19]; % 35 percent
%------------------------------------
% Number of training time steps
M = 200;                                                       

% Define Rm and the Rf ladder
Rm = 1;
Rf0 = 1e-1;                                                                
alpha = 1.8;                                                               
betamax = 26;                                                  %<----------

%------------------------------------                          %<----------
% Fix a schedule for HMC parameters

niter = cat(1,...
    1e3*ones(22,1),...
    1e4*ones(4,1));

% HMC hyperparameters: integrator step size, integration time, and masses 
% (for X_observed, X_unobserved, nu, respectively)     
epsilon = cat(1,...
    1e-2*ones(5,1),...
    1e-3*ones(8,1),...
    1e-4*ones(8,1),...
    1e-5*ones(5,1));                           
L = cat(1,...
    150*ones(10,1),...
    50*ones(16,1));
mass = cat(1,...
    [1e0,1e0,1e0].*ones(8,3),...
    [1e-1,1e-1,1e-1].*ones(1,3),...
    [1e0,1e0,1e0].*ones(5,3),...
    [1e1,1e1,1e1].*ones(4,3),...
    [1e2,1e2,1e2].*ones(3,3),...
    [1e1,1e1,1e1].*ones(5,3));                              

% Scaling of the action and its gradients
scaling = 1e6*ones(26,1);                                                 
%------------------------------------

% Each element in niter/nblocks must be an integer
nblocks = 10;
% Determine the percentage of samples used to calculate X_avg for each beta
parts = floor(nblocks/2);

% Switch for plotting action vs. beta at the end. Another can be found at 
% the end of program for plotting action vs. iteration in a chosen beta
plot_action_vs_beta = 0;                                     

% Switch for saving data at the end
savedata = 0;                                                  

%--------------------------------------------------------------------------
% Data preparation and initialization

% Load Lorenz 96 data
gen_nu = '8.17';
gen_noise = 'sig0.4';
gen_dt = '0.025';                                              
gen_delta_t = '0.025';                                         
gen_integrator = 'trapez';                                     
load(['C:\Users\zfang\Downloads\data\Lorenz96\' ...
    'L96_D' num2str(D) '_nu' gen_nu '_' gen_noise '_dt' gen_dt ... 
    '_deltat' gen_delta_t '_' gen_integrator '.mat']);
% Y is the L-by-M matrix of observations
% Specific dimensions are chosen to be the observed dimensions
Y = Data(dim_obs,1:M);
% Read time interval from data
dt = delta_t;

% Specify the unobserved dimensions
dim_unobs = setdiff(1:D,dim_obs)';

% Sanity check, make sure all the PA hyperparameters have the correct size
if (betamax ~= 1) && (betamax ~= 2)
    if (size(niter,1)~=betamax) || (size(epsilon,1)~=betamax) || ...
            (size(L,1)~=betamax) || (size(mass,1)~=betamax) || ...
            (size(scaling,1)~=betamax)
        fprintf('\nAborted. Check precision annealing setting. ');
        clear;
    end
end

% Assign the above predefined HMC masses and their square roots
% to each dimension. Don't miss the factor 2 in mass_~_sqrt
mass_X = cell(betamax,1);
mass_X_sqrt = cell(betamax,1);
mass_nu = zeros(betamax,1);
mass_nu_sqrt = zeros(betamax,1);
for beta = 1:betamax
    mass_X{beta} = zeros(D,M);
    mass_X{beta}(dim_obs,:) = mass(beta,1);
    mass_X{beta}(dim_unobs,:) = mass(beta,2);
    mass_nu(beta) = mass(beta,3);
    
    mass_X_sqrt{beta} = zeros(D,M);
    mass_X_sqrt{beta}(dim_obs,:) = sqrt(2*mass(beta,1));
    mass_X_sqrt{beta}(dim_unobs,:) = sqrt(2*mass(beta,2));
    mass_nu_sqrt(beta) = sqrt(2*mass(beta,3));
end

% Initialize some vectorized dirac delta functions to be used repetitively
% Below are D-by-D matrices of $\delta_{i-1,j}$, $\delta_{i-2,j}$,
% $\delta_{i+1,j}$, and $\delta_{ij}$, respectively
eyeDleft1 = circshift(eye(D),-1,2);
eyeDleft2 = circshift(eye(D),-2,2);
eyeDright1 = circshift(eye(D),1,2);
eyeD = eye(D);

%--------------------------------------------------------------------------
% Some initializations for precision annealing and HMC

% Define the Rf ladder
Rf = Rf0*(alpha.^(0:betamax-1));

% X/nu_init(beta) -- starting point for current beta
X_init = cell(betamax,1);
nu_init = cell(betamax,1);
for beta = 1:betamax
    X_init{beta} = zeros(D,M);
    nu_init{beta} = 0;
end

nu_init{1} = 8;
% Dynamic initialization
X_init{1}(:,1) = 20*rand(D,1) - 10;
X_init{1}(dim_obs,:) = Y;
for k = 1:M-1
    X_init{1}(:,k+1) = X_init{1}(:,k) + ...
        dt*F(X_init{1}(:,k)+dt/2*F(X_init{1}(:,k),nu_init{1}),nu_init{1});
    X_init{1}(dim_obs,k+1) = Y(:,k+1);
end

% X/nu_avg(beta) -- solutions after current beta is finished
X_avg = cell(betamax,1);
nu_avg = cell(betamax,1);
for beta = 1:betamax
    X_avg{beta} = zeros(D,M);
    nu_avg{beta} = 0;
end

% Places to store running sums of the states, with each cell an average
% of around niter/nblocks iterations
X_sum = cell(betamax,nblocks);
nu_sum = cell(betamax,nblocks);
for beta = 1:betamax
    for j = 1:nblocks
        X_sum{beta,j} = zeros(D,M);
        nu_sum{beta,j} = 0;
    end
end

niter_max = max(niter);

% Save X(t_final) and nu along the way
X_f_history = cell(betamax,niter_max);
nu_history = cell(betamax,niter_max);
for beta = 1:betamax
    for j = 1:niter_max
        X_f_history{beta,j} = zeros(D,1);
        nu_history{beta,j} = 0;
    end
end

% Initialize action matrix, each row stores the actions within one beta
action = zeros(betamax,niter_max+1);
% Initialize argmin A(X) for each beta, for reference only
action_min = zeros(betamax,1);
% Initialize A(X_avg), measurement and model errors
action_avg = zeros(betamax,1);
ME_avg = zeros(betamax,1);
FE_avg = zeros(betamax,1);
FE_avg_with_Rf = zeros(betamax,1);

% Percentage acceptance and percentage downhill (for the sampler)
Acceptance = zeros(betamax,1);
Downhill = zeros(betamax,1);

% Initialize the momentum
pX0 = zeros(D,M);

%--------------------------------------------------------------------------
% Precision annealing and Hmailtonian Monte Carlo

% Do annealing for each beta starting from the solution of the previous one
for beta = 1:betamax
    
    % Initialize states
    X0 = X_init{beta};
    nu0 = nu_init{beta};
    
    % Evaluate the starting action under current beta
    [Xu1,Xd1,Xd2,Xleft1,hX] = fun_getPieces(X0,nu0,dt);
    action(beta,1) = fun_A(X0,Xleft1,hX,dim_obs,M,Y,Rm,Rf(beta));
    action_min(beta) = action(beta,1);
    
    % HMC kernel
    %------------------------------------
    fprintf('Start annealing for beta = %d... ',beta); tic;
    for n = 2:niter(beta)+1
        eps = epsilon(beta);
        
        % Take current states as starting points
        X = X0;
        nu = nu0;
        
        % Generate initial momenta from a multivariate normal distribution
        pX0(dim_obs,:) = normrnd(0,sqrt(mass(beta,1)),Dobs,M);
        pX0(dim_unobs,:) = normrnd(0,sqrt(mass(beta,2)),D-Dobs,M);
        pnu0 = normrnd(0,sqrt(mass(beta,3)));
        
        % Get necessary pieces
        [Xu1,Xd1,Xd2,Xleft1,hX] = fun_getPieces(X,nu,dt);
        % Half step for the X momenta
        pX = pX0 - eps/2*fun_dAX(X,Xu1,Xd1,Xd2,Xleft1,hX,...
            eyeD,eyeDleft2,eyeDleft1,eyeDright1,...
            D,dim_obs,M,Y,dt,Rm,Rf(beta),scaling(beta));
        % Half step for the nu momentum
        pnu = pnu0 - ...
            eps/2*fun_dAnu(Xleft1,hX,M,dt,Rf(beta),scaling(beta));
        
        % Simulate Hamiltonian dynamics
        for i = 1:L(beta)
            % Full step for the state variables
            X = X + eps*pX./mass_X{beta};
            nu = nu + eps*pnu/mass_nu(beta);
            
            % Get necessary pieces
            [Xu1,Xd1,Xd2,Xleft1,hX] = fun_getPieces(X,nu,dt);
            if i~=L(beta)
                % Full step for the X momenta except at end of trajectory
                pX = pX - eps*fun_dAX(X,Xu1,Xd1,Xd2,Xleft1,hX,...
                    eyeD,eyeDleft2,eyeDleft1,eyeDright1,...
                    D,dim_obs,M,Y,dt,Rm,Rf(beta),scaling(beta));
                % Full step for the nu momentum except at end of trajectory
                pnu = pnu - ...
                    eps*fun_dAnu(Xleft1,hX,M,dt,Rf(beta),scaling(beta));
            end
        end
        
        % Half step for the X momenta
        pX = pX - eps/2*fun_dAX(X,Xu1,Xd1,Xd2,Xleft1,hX,...
            eyeD,eyeDleft2,eyeDleft1,eyeDright1,...
            D,dim_obs,M,Y,dt,Rm,Rf(beta),scaling(beta));
        % Half step for the nu momentum
        pnu = pnu - ...
            eps/2*fun_dAnu(Xleft1,hX,M,dt,Rf(beta),scaling(beta));
        
        % Calculate action when simulation is done
        action_candidate = fun_A(X,Xleft1,hX,dim_obs,M,Y,Rm,Rf(beta));
        % Metropolis-Hastings acceptance/rejection rule
        if rand < exp(scaling(beta)*action(beta,n-1) ... 
                + sumsqr(pX0./mass_X_sqrt{beta}) ...
                + (pnu0/mass_nu_sqrt(beta))^2 ...
                - scaling(beta)*action_candidate ... 
                - sumsqr(pX./mass_X_sqrt{beta}) ...
                - (pnu/mass_nu_sqrt(beta))^2)
            % Accepted states and action serve as the next starting point
            X0 = X;
            nu0 = nu;
            action(beta,n) = action_candidate;
            % Count acceptance rate
            Acceptance(beta) = Acceptance(beta) + 1;
        else
            % If rejected, maintain the same proposal states and action
            action(beta,n) = action(beta,n-1);
        end
        
        % Assign the running summations to their corresponding cells
        X_sum{beta,ceil((n-1)/(niter(beta)/nblocks))} = ...
            X_sum{beta,ceil((n-1)/(niter(beta)/nblocks))} + X0;
        nu_sum{beta,ceil((n-1)/(niter(beta)/nblocks))} = ...
            nu_sum{beta,ceil((n-1)/(niter(beta)/nblocks))} + nu0;
        
        % Save X(t_final) and nu along the way
        X_f_history{beta,n-1} = X0(:,M);
        nu_history{beta,n-1} = nu0;
        
        % Check if current proposal yields the lowest action so far
        if action(beta,n) < action_min(beta)
            % If so, update the lowest action and the corresponding states
            action_min(beta) = action(beta,n);
            % Count downhill rate
            Downhill(beta) = Downhill(beta) + 1;
        end
    end
    fprintf('Done. '); toc;
    %------------------------------------
    
    % Solution for current bata
    for j = (nblocks-parts+1):nblocks
        X_avg{beta} = X_avg{beta} + X_sum{beta,j};
        nu_avg{beta} = nu_avg{beta} + nu_sum{beta,j};
    end
    X_avg{beta} = X_avg{beta}/((niter(beta)/nblocks)*parts);
    nu_avg{beta} = nu_avg{beta}/((niter(beta)/nblocks)*parts);
    
    % Final action for current beta
    [~,~,~,Xleft1,hX] = fun_getPieces(X_avg{beta},nu_avg{beta},dt);
    action_avg(beta) = ...
        fun_A(X_avg{beta},Xleft1,hX,dim_obs,M,Y,Rm,Rf(beta));
    
    % Measurement and model errors for current beta
    ME_avg(beta) = 1/(2*M)*sumsqr(X_avg{beta}(dim_obs,:) - Y);
    temp = (Xleft1 - hX);
    FE_avg(beta) = 1/(2*M)*sumsqr(temp(:,1:M-1));
    FE_avg_with_Rf(beta) = Rf(beta)*FE_avg(beta);
    
    % starting point for the next beta
    if beta ~= betamax
        X_init{beta+1} = X_avg{beta};
        nu_init{beta+1} = nu_avg{beta};
    end
    
    % Finalize the percentages
    Acceptance(beta) = Acceptance(beta)/niter(beta);
    Downhill(beta) = Downhill(beta)/niter(beta);
    
end
fprintf('\n');

%--------------------------------------------------------------------------
% save data
if savedata == 1
    % assign a temporary name to the saved data
    save([codename '_' date '_' num2str(randi([10000,99999]))],...
        'Acceptance','action','action_avg','action_min','alpha',...
        'betamax','codename','D','dim_obs','Dobs','Downhill','dt',...
        'epsilon','FE_avg','FE_avg_with_Rf','gen_delta_t','gen_dt',...
        'gen_integrator','gen_noise','gen_nu','L','M','mass','ME_avg',...
        'nblocks','niter','nu_avg','nu_history','nu_init','nu_sum',...
        'parts','Rf','Rf0','Rm','scaling',...
        'X_avg','X_f_history','X_init','X_sum','Y');
end

%--------------------------------------------------------------------------
% Visualize the whole annealing process

if plot_action_vs_beta == 1
    % Plot the whole action vs. Rf process
    figure; ax1 = axes('position',[0.1 0.12 0.87 0.83]);
    loglog(ax1,Rf,FE,'LineWidth',1.5); hold all;
    loglog(ax1,Rf,action_min,'-','LineWidth',2); hold off;
    legend('Model err.','Action','Location','northwest');
    ylabel('Action'); xlabel('Rf');
%     xlim(ax1,[min(Rf) max(Rf)]); grid on;
    xlim(ax1,[min(Rf) 1e6]); grid on;
%     xlim(ax1,[1e2 1e6]); ylim(ax1,[1e-3 1e6]); grid on;
end

%--------------------------------------------------------------------------
% Plot what happened in a particular beta value

beta_choice = 0;                                               %<----------
if beta_choice ~= 0
    % Plot what happened within the chosen beta value
    figure; ax2 = axes('position',[0.1 0.12 0.87 0.83]);
    loglog(ax2,action(beta_choice,1:niter(beta_choice)+1),...
        'LineWidth',1.5); grid on;
    xlim([1 niter(beta_choice)+1]);
    ylabel('Action'); xlabel('Iteration');
    title(['\beta = ' num2str(beta_choice) ', Rf = ' ...
        num2str(Rf(beta_choice),'%10.0e')]);
end

















