
check_nu = 0;

RK_pred = 2;
dt_pred = 0.025;

obs_dim = 1;
unobs_dim = 20;

nu_est = q_min{2};
% nu_est = 8.17; % Sanity check
X_est = q_min{1};

% predtime/dt_pred and predtime/dt_train must be integers
predtime = 3;

% Load the ground truth
if check_nu == 1
    gen_noise = 'noiseless';
end
load(['C:/Users/zfang/Downloads/data/Lorenz96/' ...
    'L96_D' num2str(D) '_nu' gen_nu '_' gen_noise '_dt' gen_dt ... 
    '_deltat' gen_delta_t '_' gen_integrator '.mat']);
dt_train = dt;
X_true = Data(:,1:M+predtime/dt_train);

X_pred = zeros(D,predtime/dt_pred);

% Predictions
if RK_pred == 2
    % Second order Runge-Kutta
    
    if check_nu == 0
        X_pred(:,1) = X_est(:,end) + dt_pred*...
            F(X_est(:,end)+dt_pred/2*F(X_est(:,end),nu_est),nu_est);
    elseif check_nu == 1
        X_pred(:,1) = X_true(:,M) + dt_pred*...
            F(X_true(:,M)+dt_pred/2*F(X_true(:,M),nu_est),nu_est);
    end
    
    for k = 1:predtime/dt_pred-1
        X_pred(:,k+1) = X_pred(:,k) + ...
            dt_pred*F(X_pred(:,k)+dt_pred/2*F(X_pred(:,k),nu_est),nu_est);
    end
    
elseif RK_pred == 4
    % Fourth order Runge-Kutta
    
    if check_nu == 0
        u1 = dt_pred*F(X_est(:,end),nu_est);
        u2 = dt_pred*F(X_est(:,end)+u1/2,nu_est);
        u3 = dt_pred*F(X_est(:,end)+u2/2,nu_est);
        u4 = dt_pred*F(X_est(:,end)+u3,nu_est);
        X_pred(:,1) = X_est(:,end) + u1/6 + u2/3 + u3/3 + u4/6;
    elseif check_nu == 1
        u1 = dt_pred*F(X_true(:,M),nu_est);
        u2 = dt_pred*F(X_true(:,M)+u1/2,nu_est);
        u3 = dt_pred*F(X_true(:,M)+u2/2,nu_est);
        u4 = dt_pred*F(X_true(:,M)+u3,nu_est);
        X_pred(:,1) = X_true(:,M) + u1/6 + u2/3 + u3/3 + u4/6;
    end
    
    for k = 1:predtime/dt_pred-1
        u1 = dt_pred*F(X_pred(:,k),nu_est);
        u2 = dt_pred*F(X_pred(:,k)+u1/2,nu_est);
        u3 = dt_pred*F(X_pred(:,k)+u2/2,nu_est);
        u4 = dt_pred*F(X_pred(:,k)+u3,nu_est);
        X_pred(:,k+1) = X_pred(:,k) + u1/6 + u2/3 + u3/3 + u4/6;
    end
    
end

% Time axes
time_true = linspace(dt_train,M*dt_train+predtime,M+predtime/dt_train);
time_est = linspace(dt_train,M*dt_train,M);
time_pred = ...
    linspace(M*dt_train+dt_pred,M*dt_train+predtime,predtime/dt_pred);

% Plot observed dimension
figure;
plot(time_true,X_true(obs_dim,:),'k','LineWidth',1.5); hold all;
plot(time_est,X_est(obs_dim,:),'r','LineWidth',1.0); hold all;
plot(time_pred,X_pred(obs_dim,:),'b','LineWidth',1.0); hold off;
legend({'data','estimate','prediction'},'Orientation','horizontal');
title('Lorenz96 D = 20, L = 12');
xlabel('Time (a.u.)'); ylabel('x_1(t) -- observed');
xticks(0:0.5:8);
xticklabels({'0' '' '1' '' '2' '' '3' '' '4' '' '5' '' '6' '' '7' '' '8'});
ylim([-5 10]); yticks(-5:1.5:10);

% Plot unobserved dimension
figure;
plot(time_true,X_true(unobs_dim,:),'k','LineWidth',1.5); hold all;
plot(time_est,X_est(unobs_dim,:),'r','LineWidth',1.0); hold all;
plot(time_pred,X_pred(unobs_dim,:),'b','LineWidth',1.0); hold off;
legend({'data','estimate','prediction'},'Orientation','horizontal');
title('Lorenz96 D = 20, L = 12');
xlabel('Time (a.u.)'); ylabel('x_{20}(t) -- unobserved');
xticks(0:0.5:8);
xticklabels({'0' '' '1' '' '2' '' '3' '' '4' '' '5' '' '6' '' '7' '' '8'});
ylim([-5 10]); yticks(-5:1.5:10);

% % Output ASCII data files for each dimension
% foldername = ['Lor96_pred_D' num2str(D) '_L' num2str(Dobs) '_(' date ')'];
% mkdir('C:\Users\zfang\Downloads',foldername);
% 
% for a = 1:D
%     if a <= 9
%         file_data = ['x_0' num2str(a) '(t)_data'];
%         file_est = ['x_0' num2str(a) '(t)_est'];
%         file_pred = ['x_0' num2str(a) '(t)_pred'];
%     else
%         file_data = ['x_' num2str(a) '(t)_data'];
%         file_est = ['x_' num2str(a) '(t)_est'];
%         file_pred = ['x_' num2str(a) '(t)_pred'];
%     end
%     
%     temp_true = cat(2,time_true',X_true(a,:)');
%     save(['C:\Users\zfang\Downloads\' foldername '\' file_data '.dat'],...
%         'temp_true','-ascii','-double');
%     
%     temp_est = cat(2,time_est',X_est(a,:)');
%     save(['C:\Users\zfang\Downloads\' foldername '\' file_est '.dat'],...
%         'temp_est','-ascii','-double');
%     
%     temp_pred = cat(2,time_pred',X_pred(a,:)');
%     save(['C:\Users\zfang\Downloads\' foldername '\' file_pred '.dat'],...
%         'temp_pred','-ascii','-double');
% end























