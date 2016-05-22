%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Design and evaluation of sensing matrix design as reported in the paper 'Power-Constrained Sparse Gaussian Linear
%%% Dimensionality Reduction over Noisy Channels' by Amirpasha Shirazinia
%%% and Subhrakanti Dey, IEEE Transactions on Signal Processing, 2015.

%%% Written by: Amirpasha Shirazinia?, Signals & Systems Division, Uppsala
%%% University, Sweden
%%% Email: amirpasha.shirazinia@signal.uu.se
%%% Created: August 2014, Revised: August 2015

%%% The following sample code is for the single-terminal case, and outputs are MSE and
%%% probability for support set recovery versus number of measurements.
%%% It is straightforward to plot them versus power or channel gain.
%%% Please note that the output figuers are not the ones generated in the
%%% original paper, rather give hints how they have been produced.

%%% The program might be slow if you increase 'N' or 'K' since the proposed
%%% method counts 'nchoosek(N,K)' possibilities. In this case, sit back, relax and try to use the
%%% approximate optimization method (codes are given below) by selecting
%%% 'it_max_appx' carefully (See section VI in the paper).

%%% Remember to install CVX (http://cvxr.com/cvx/download) beforehand.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all
%% Inputs
N = 24; % Sparse vector size
K = 2; % Sparsity level
MM = 8:4:20; % Number of measurements
sigma_x_sq = 1; % Variance of each component of the source vector (applicable when OMP_MMSE sparse reconstruction is used)
sigma_w_sq = 0.01; % Variance of (channel) noise
g = 0.5; % channel gain
P_dB = 10; % Total power in dB scale
P = 10.^(P_dB/10); % Total power in normal scale
itr_max = 500; % Number of Monte-Carlo simulations for performance evaluation
all_perms = nchoosek(1:N,K); % All permutations of support set
L = length(all_perms); % Size of support
H = eye(N); % Source-to-sensor matrix is set to an identity transform
% Generate exponential covariance matrix model for the components within
% the support set
rho = 0.5; % Correlation coefficient varying from 0 to 1
Sigma = zeros(K,K);
for i = 1:K
    for j = 1:K
        Sigma(i,j) = rho^abs(i-j);
    end
end
[U_Sigma, V_Sigma] = eig(Sigma); % EVD of the covar. matrix within the support

% Calculate sample covariance matrix of the whole source based on the
% previously-generated exponential covaraince matrix
R_x = Covar_X(N,K,Sigma);


%% Pre-allocations for MSE
MSE_lb_mc = zeros(length(MM),itr_max);
MSE_lb_rand_mc = zeros(length(MM),itr_max);
MSE_ub_mc = zeros(length(MM),itr_max);
MSE_equal_mc = zeros(length(MM),itr_max);
MSE_Gauss_mc = zeros(length(MM),itr_max);

%% Pre-allocations for Probability of support set recovery
supp_err_lb = zeros(length(MM),itr_max);
supp_err_ub = zeros(length(MM),itr_max);
supp_err_equal = zeros(length(MM),itr_max);
supp_err_Gauss = zeros(length(MM),itr_max);

%% Loop for counting measurements
for m = 1:length(MM)
    M = MM(m)
    
    Eig_vec_Q2 = dctmtx(M); % Left Eigenvector (determisitic and common for all designs)
    Eig_vec_Q3 = dctmtx(N); % Right Eigenvector (determisitic)
    
    %% Optimized design of sensing matrices
    
    % Equal power allocation (or tight-frame)
    A_equal = Eig_vec_Q2*[eye(M) zeros(M,N-M)]*Eig_vec_Q3';
    A_equal = sqrt(P/trace(R_x*(H'*(A_equal'*A_equal)*H)))*A_equal; % power re-scaling
   
    
    % Upper-bound minimizing sensing matrix
    A_optimal_ub = A_opt_ub(N,M,R_x,P,g,sigma_w_sq,H,Eig_vec_Q2);
    
    
    % Lower-bound minimizing sensing matrix (proposed scheme)
    % 1. Excat sensing matrix design
    A_optimal_lb = A_opt_lb(N,M,K,P,R_x,Sigma,sigma_w_sq,g,H,Eig_vec_Q2); % Here, Q = A'*A, so we need to extract out A from the actual low-rank Q
    
    
    % 2. Approximate sensing matrix design with low complexity
    
    % it_max_appx = 1500; % Number of iterations for Monte-Carlo simulations (associated with performance evaluation)
    % perms_appx = zeros(K,it_max_appx); % Number of samples required for the approximate method (in Section VI)
    % for it_appx = 1:it_max_appx
    %     perms_appx(:,it_appx) = randsample(N,K); % Find random supports associ. with the number of samples
    % end
    %A_optimal_lb_low = A_opt_lb_low(N,M,K,P,R_x,Sigma,sigma_w_sq,g,H,it_max_appx,perms_appx,Eig_vec_Q2);
     

    %% Performance evaluation based on Monte-Carlo simulations
    for mc = 1:itr_max % Monte-Carlo loop
        % Generate random sparse source and random noise
        supp = randsample(N,K)'; % Uniformaly at rondom support set
        x = zeros(N,1);
        x(supp) = (U_Sigma*V_Sigma^0.5*U_Sigma')*randn(K,1); % Sparse vector generation based on the generated covariance matrix
        noise = sqrt(sigma_w_sq)*randn(M,1); % Additive noise 
        
        %% Performance evaluation of Lower-bound minimizing sensing matrix
        y_lb = g*A_optimal_lb*H*x + noise; % Measurements 
        
        % Sparse source recovery based on MMSE estimation (computationally too expensive)
        %x_hat_lb = MMSE(y_lb,g*A_optimal_lb*H,K,Sigma,sigma_w_sq); 
        
        % Sparse source recovery based on Rand-OMP (computationally less expensive)
        %[x_hat_lb,S] = OMP_MMSE(y_lb,g*A_optimal_lb*H,K,sigma_x_sq,sigma_w_sq);  
        
        % Sparse source recovery based on OMP (the least complex)
        [x_hat_lb,~,~,~, ~] = OMP( g*A_optimal_lb*H, y_lb, K, [], []);
        MSE_lb_mc(m,mc) = norm(x - x_hat_lb)^2;
        supp_err_lb(m,mc) = length(find(find(x) - find(x_hat_lb)));

        
        %% Performance evaluation of Upper-bound minimizing sensing matrix
        y_ub = g*A_optimal_ub*H*x + noise;
        %x_hat_ub = MMSE(y_ub,g*A_optimal_ub*H,K,Sigma,sigma_w_sq);
        %x_hat_ub = OMP_MMSE(y_ub,g*A_optimal_ub*H,K,sigma_x_sq,sigma_w_sq);
        [x_hat_ub,~,~,~, ~] = OMP( g*A_optimal_ub*H, y_ub, K, [], []);
        MSE_ub_mc(m,mc) = norm(x - x_hat_ub)^2;
        supp_err_ub(m,mc) = length(find(find(x) - find(x_hat_ub)));
        

        %% Performance evaluation of Tight frame
        y_equal = g*A_equal*H*x + noise;
        %x_hat_equal = MMSE(y_equal,g*A_equal,K,Sigma,sigma_w_sq);
        %x_hat_equal = OMP_MMSE(y_equal,g*A_equal,K,sigma_x_sq,sigma_w_sq);
        [x_hat_equal,~,~,~, ~] = OMP( g*A_equal*H, y_equal, K, [], []);
        MSE_equal_mc(m,mc) = norm(x - x_hat_equal)^2;
        supp_err_equal(m,mc) = length(find(find(x) - find(x_hat_equal)));
        
        
        %% Performance evaluation of Gaussian sensing matrix
        A_Gauss = randn(M,N);
        A_Gauss = sqrt(P/trace(R_x*H'*(A_Gauss'*A_Gauss)*H))*A_Gauss;
        y_Gauss = g*A_Gauss*H*x + noise;
        %x_hat_Gauss = MMSE(y_Gauss,g*A_Gauss*H,K,Sigma,sigma_w_sq);
        %x_hat_Gauss = OMP_MMSE(y_Gauss,g*A_Gauss*H,K,sigma_x_sq,sigma_w_sq);
        [x_hat_Gauss,~,~,~, ~] = OMP( g*A_Gauss*H, y_Gauss, K, [], []);
        MSE_Gauss_mc(m,mc) = norm(x - x_hat_Gauss)^2;
        supp_err_Gauss(m,mc) = length(find(find(x) - find(x_hat_Gauss)));
    end % end of mc loop
    
end % end of M loop

%% Outputs and plots

% Normalized MSE
MSE_lb = mean(MSE_lb_mc,2)/K;
MSE_ub = mean(MSE_ub_mc,2)/K;
MSE_equal = mean(MSE_equal_mc,2)/K;
MSE_Gauss = mean(MSE_Gauss_mc,2)/K;

% Probability of suppoer set recovery
prob_supp_err_lb = 1 - sum(supp_err_lb,2)/(K*itr_max);
prob_supp_err_ub = 1 - sum(supp_err_ub,2)/(K*itr_max);
prob_supp_err_equal = 1 - sum(supp_err_equal,2)/(K*itr_max);
prob_supp_err_Gauss = 1 - sum(supp_err_Gauss,2)/(K*itr_max);


figure;
plot(MM,10*log10(MSE_lb),'-s',MM,10*log10(MSE_ub),'-o',MM,10*log10(MSE_Gauss),'->',MM,10*log10(MSE_equal),'-d','LineWidth',1.5)
xlabel('Number of measurements (M)')
ylabel('Normalized MSE (dB)')
grid on
legend('LB-minimizing','UB-minimizing','Gaussian','Tight-frame')

figure;
plot(MM,prob_supp_err_lb,'-s',MM,prob_supp_err_ub,'-o',MM,prob_supp_err_Gauss,'->',MM,prob_supp_err_equal,'-d','LineWidth',1.5)
xlabel('Number of measurements (M)')
ylabel('Probability of support recovery')
grid on
legend('LB-minimizing','UB-minimizing','Gaussian','Tight-frame')

