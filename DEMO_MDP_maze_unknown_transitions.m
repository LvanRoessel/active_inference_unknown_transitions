function MDP = DEMO_MDP_maze
% This script reproduces the simulations of the master thesis 
% "Discrete state-space active inference and unknown controlled state 
% transitions" by Luuk van Roessel. It is divided in four parts, each having 
% their own function. 
% 
% The first part functions as a setting menu. The different scenario 
% numbersresult in different test results derived throughout the thesis.
% 
% The second part implements the generative model related to the grid 
% world. Here, standard components of the generative model are defined like 
% the likelihood matrix, transition matrix, possible policies matrix. 
% Furthermore, two types of uncertain transition knowledge are developed: 
% the minor uncertainty implementation and the total uncertainty 
% implementation.
% 
% The third part initiates policy execution using the related 
% “spm_MDP_VB_X_unknown_transitions” script, which chooses actions and 
% updates parameters based on free energy minimisation. 
% 
% The fourth part plots the resuls per specific scenario. 

% This script is a modification of Karl Friston's DEMO_MDP_maze file.
% ($Id: DEMO_MDP_maze.m 7766 2020-01-05 21:37:39Z karl $)

rng('default')          %set random numer generator to default modus
%--------------------------------------------------------------------------
label.factor     = {'where'};
label.modality   = {'what','where'};
label.outcome{1} = {'open','closed'};

trials = 1;         % Number of trials
policies = 500;     % Number of polic

%% scenario Settings
% Here the setting per scenario are defined 
% scenario 1 = No uncertainty results (Figure 5.1)
% scenario 2 = Minimal uncertainty results (Figure 5.3 and 5.4)
% scenario 3 = Minimal uncertainty results while removing incorrect concentration parameters (Figure 5.8)
% scenario 4 = Total uncertainty results (Figures 5.9 and 5.10)
% scenario 5 = Total uncertainty results including alternative learning mechanism (Figure 5.11)
% scenario 6 = Total uncertainty results, including alternative learning mechanism and removing concentration parameters (Figure 5.13)
% scenario 7 = Total uncertainty including goal, alternative learning mechanism and removing conentration parameters (Figure 6.1)

% uncertainty = 1, no uncertainty
% uncertainty = 2, minimal uncertainty
% uncertainty = 3, total uncertainty
% remove_cp = 1, no removing of incorrect concentration parameters
% remove_cp = 2, removing of incorrect concentration parameters
% policies, number of policies per executed trial
% learning = 1, standard learning mechanism
% learning = 2, alternative learning mechanism
% exploitation = 1, no goal task
% exploitation = 2, goal task set

scenario = 1;          %Switch for specific scenario

% scenario settings
if scenario == 1
    uncertainty = 1;
    remove_cp =  1;
    policies = 60;
    learning = 1;
    exploitation = 1;
elseif scenario == 2
    uncertainty = 2;
    remove_cp = 1;
    learning = 1;
    exploitation = 1;
elseif scenario == 3
    uncertainty = 2;
    remove_cp = 2;
    learning = 1;
    exploitation = 1;
elseif scenario == 4
    uncertainty = 3;
    remove_cp = 1;
    learning = 1;
    exploitation = 1;
elseif scenario == 5
    uncertainty = 3;
    remove_cp = 1;
    learning = 2;
    exploitation = 1;
elseif scenario == 6
    uncertainty = 3;
    remove_cp = 2;
    learning = 2;
    exploitation = 1;
elseif scenario == 7
    uncertainty = 3;
    remove_cp = 1;
    learning = 2;
    trials = 40;
    policies = 40;
    exploitation = 2;
end

%% Generative model
% Geometry of the environment
MAZE  = [...
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0];
END   = sub2ind(size(MAZE),1,4);                 % goal or target location, not important for scenarios 1-6
START = sub2ind(size(MAZE),4,2);                 % first or start location, always starting from location 8

% prior beliefs about initial states: D 
%--------------------------------------------------------------------------
D{1}  = zeros(numel(MAZE),1);                    % D vector
Ns    = numel(D{1});                             % number of states

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
A{1}  = [1 - MAZE(:), MAZE(:)]';                  % what outcome, an outcome type defined by the original "DEMO_MDP_maze" script, where states were open or closed. In our case all the states are open states.
A{2}  = eye(Ns,Ns);                               % where outcome, each state has a related observation
Ng    = numel(A);                                 % number of outcome types
for g = 1:Ng
    No(g)  = size(A{g},1);                        % number of observations per outcome
end

% Accurate controlled transitions: B (up, down, left, right, stay)
%--------------------------------------------------------------------------
u    = [1 0; -1 0; 0 1; 0 -1; 0 0];               % allowable actions
nu   = size(u,1);                                 % number of actions
B{1} = zeros(Ns,Ns,nu);                           % initialise the transition matrix
[n,m] = size(MAZE);
for i = 1:n
    for j = 1:m
        
        % allowable transitions from state s to state ss
        %------------------------------------------------------------------
        s     = sub2ind([n,m],i,j);
        for k = 1:nu
            try
                ss = sub2ind([n,m],i + u(k,1),j + u(k,2));
                B{1}(ss,s,k) = 1;
            catch
                B{1}(s, s,k) = 1;
            end
        end
    end
end

% allowable policies (2 moves): V
%--------------------------------------------------------------------------
V     = []; % original
for i = 1:nu
    for j = 1:nu
        V(:,end + 1) = [i;j];
    end
end

% Initialise matrix for preferred observations: C
%--------------------------------------------------------------------------
for g = 1:Ng
    C{g} = zeros(No(g),1);
end

% basic MDP structure
%--------------------------------------------------------------------------
mdp.V = V;                      % allowable policies
mdp.A = A;                      % observation model or likelihood
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states
mdp.Ns = Ns;                    % number of states
mdp.remove_cp = remove_cp;      % remove concentration parameters mechanism switch (1 or 2)
mdp.uncertainty = uncertainty;  % pass information about uncertainty type (1, 2, 3)
mdp.learning = learning;        % pass information about alternative learning mechanism
mdp.exploitation = exploitation;% pass information about pure exploration (1) or exploitation-exploration (2)
mdp.F_671011 = 0;               % pass information about Initialise state accuracy measure

% Creating different types of transition uncertainty (Thesis section 4-3)
if uncertainty > 1
    mdp.b{1} = mdp.B{1};
    mdp.b2{1} = mdp.B{1};
    for i = 1:5
        if uncertainty == 2
            mdp.b{1}(:,:,i)= spm_norm(mdp.b2{1}(:,:,2)+ 1*mdp.b2{1}(:,:,i))/10; % Minimal uncertainty
        elseif uncertainty == 3
            mdp.b{1}(:,:,i) = spm_norm(ones(16,16));                            % Total uncertainty
        end
    end
else
    mdp.B{1}= mdp.B{1};                                                         % No uncertainty
end


%% Test run
mdp.label = label;
mdp       = spm_MDP_check(mdp);
MDP3   = mdp;  

% Set random goals per trial if exploitation task
if exploitation == 2
END = zeros(trials,1);
END = round(rand(trials,1)*16);
end

matrices = trials*policies;     % number of total policies in a test run
reward = 0;                     % initialise reward
tic;                        

% initialise measurements (free energies, knowledge accuracies, reward and visited
% states)
F_result2 = zeros(matrices,1);  
S_avg = zeros(matrices,1);
TN_avg = zeros(matrices,1);
F_671011 = zeros(matrices,1);
visited_states =  zeros(Ns,1);
bnt_storage = zeros(matrices,1);
reward_dev = zeros(trials,1);

% Start test
for i = 1:trials
    %MDP3(end).D{1}(START) = 1;
    
    % Choose between pure exploration/exploitation-exploration task
    if exploitation == 1
        MDP = spm_maze_search(MDP3(end),policies,START,END,128,0);
    elseif exploitation == 2
        MDP = spm_maze_search(MDP3(end),policies,START,END(i),128,1);
    end
    
    if MDP(end).s(end) == END(i)
        reward = reward + 1;        % Accumulate reward if goal state is occupied at trial's end
    end
    
    reward_dev(i) = reward;         % Store accumulated reward per trial to show development
    
    for j = 1:policies
        if MDP(end).uncertainty > 1
            bnt_storage((i*policies-policies)+j) = sum(matrix_difference(MDP(j).B{1},spm_norm(MDP(j).b{1}))); % Calculate the difference between the true transition matrix B and the approximated transition matrix norm(b)
            F_671011((i*policies-policies)+j,:) = MDP(j).F_671011;                                            % Calculate approximated accuracy of the state knowledge as described in Section 4.4.2
            S_avg((i*policies-policies)+j,:) = MDP(j).S_avg;                                                  % Store average salience per policy
            TN_avg((i*policies-policies)+j,:) = MDP(j).TN_avg;                                                % Store average transition novelty per policy      
        end
        F_result2((i*policies-policies)+j,:) = MDP(j).F_result2;                                              % Store average variational free energy
        visited_states(MDP(j).s(1)) = visited_states(MDP(j).s(1)) + 1;                                        % Store visited states (first time step policy)
        visited_states(MDP(j).s(2)) = visited_states(MDP(j).s(2)) + 1;                                        % Store visited states (second time step policy)
    end
    
    MDP3(end).D{1}  = zeros(numel(MAZE),1);                                                         % Remove belief about initial state distribution due to relocation
    if isfield(MDP(end),'b')                                                                        % Pass on final belief about controlled state transitions
        MDP3(end).b = MDP(end).b;
    end
    
    if MDP(end).remove_cp == 1 && MDP(end).exploitation == 2                                        % Remove initial concentration parameters mechanism
        for kk = 1:5
        MDP3(end).b{1}(:,:,kk) = MDP(end).b{1}(:,:,kk).*(spm_norm(MDP(end).b{1}(:,:,kk))>0.06);      % Canceling the concentration parameters that are smaller than 0.06 after normalisation of the column
        end
    end

clear MDP    
end

%% Plotting

if scenario > 1
figure;
plot(bnt_storage);                                                          % plot the development of the transition knowledge
xlabel('Policy number')
ylabel('[nats]')
title('Development of the Transition knowledge')
end

if scenario == 1 || scenario == 6
vs_env = reshape(visited_states,4,4);
figure; imagesc(vs_env); colorbar; xticks([1 2 3 4]); yticks([1 2 3 4]);    % visualise visited states
title('Number of visits per state')
end

if scenario == 2
figure;
plot(F_671011);                                                             % plot the development of the approximated state knowledge
xlabel('Policy number')
ylabel('[nats]')
title('Development of the approximated state knowledge')
end

if scenario == 2 | scenario == 6
figure;
plot(S_avg);                                                                % plot the development of the average salience per policy
xlabel('Policy number')
ylabel('[G_S]')
title('Average Salience')

figure;
plot(TN_avg);                                                               % plot the development of the average transition novelty per policy
xlabel('Policy number')
ylabel('[G_{TN}]')
title('Average Transition Novelty')
end

if scenario == 1 | scenario == 2 | scenario == 6
figure;
plot(F_result2);                                                            % plot the development of the average variational free energy er policy
xlabel('Policy number')
ylabel('[F]')
title('Average variational free energy')
end

if scenario == 7
figure;
bar(reward_dev(1:trials))                                                   % plot the reward accumulated over multiple trials for the exploration-exploitation task
xlabel('Trial number')
ylabel('Accumulated reward')
title('Reward accumulation over trials')
end

toc;

function MDP = spm_maze_search(mdp,N,START,END,alpha,beta)
% FORMAT MDP = spm_maze_search(mdp,N,START,END,alpha,beta
% mdp   - MDP structure
% N     - number of trials (i.e., policies: default 8)
% START - index of intial state (default 1)
% END   - index of target state (default 1)
% alpha - prior concentration parameter for likelihood (default 128)
% beta  - precision of prior preference (default 0)
% peta  - Switch transition uncertainty on (1) or of (else)
% MDP   - MDP structure array

% preliminaries
%--------------------------------------------------------------------------
try, N;     catch, N     = 8;   end
try, START; catch, START = 1;   end
try, END;   catch, END   = 1;   end
try, alpha; catch, alpha = 128; end
try, beta;  catch, beta  = 0;   end

if ~isfield(mdp,'o')
    mdp.o = [];
end
if ~isfield(mdp,'u')
    mdp.u = [];
end
mdp.s = START;

% Evaluate a sequence of moves - recomputing prior preferences at each move
%==========================================================================
for i = 1:N
    
    % Evaluate preferred states on the basis of current beliefs (only if
    % exploration-exploitation task)
    %----------------------------------------------------------------------
    mdp.C{2} = spm_maze_cost(mdp,END)*beta;
    
    % Execute a policy
    %----------------------------------------------------------------------
    MDP(i)   = spm_MDP_VB_X_unknown_transitions(mdp);
    
    % Approximate the state knowledge using the updated transition
    % knowledge
    if MDP(i).uncertainty > 1
        mdp_temp.V = MDP(i).V;
        mdp_temp.a = MDP(i).A;
        mdp_temp.A = MDP(i).A;
        for ii = 1:5
            mdp_temp.B{1}(:,:,ii) = spm_norm(MDP(i).b{1}(:,:,ii));
        end
        MDP(i).F_671011 = test_F(mdp_temp);
        clear mdp_temp
    end
    
    % Remove incorrect concentration parameters after each policy (Page 52 of the thesis)
    if MDP(i).remove_cp == 2 && MDP(i).exploitation == 1
        for kk = 1:5
        MDP(end).b{1}(:,:,kk) = MDP(end).b{1}(:,:,kk).*(spm_norm(MDP(end).b{1}(:,:,kk))>0.06);
        end
    end
    
    
    mdp      = MDP(i);
    mdp.s    = mdp.s(:,end);
    mdp.D{1} = MDP(i).X{1}(:,end);
    mdp.o    = [];
    mdp.u    = [];
    
end

return


function C = spm_maze_cost(MDP,END)
% Evaluate subgoals using graph Laplacian
%==========================================================================
START = MDP.s(1);
if isfield(MDP,'a')
    Q = MDP.a{1};
else
    Q = MDP.A{1};
end
Q   = Q/diag(sum(Q));
Q   = Q(1,:);                                % open states
if isfield(MDP,'b')
    P   = diag(Q)*any(MDP.b{1}.*(MDP.b{1}>0.99),3);
else
    P   = diag(Q)*any(MDP.B{1},3); 
end

ns  = length(Q);                             % number of states
X   = zeros(ns,1);X(START) = 1;              % initial state
Y   = zeros(ns,1);Y(END)   = 1;              % target state


% Preclude transitions to closed states and evaluate graph Laplacian
%--------------------------------------------------------------------------
P   = P - diag(diag(P));
P   = P - diag(sum(P));
P   = expm(P);

% evaluate (negative) cost as a path integral conjunctions
%--------------------------------------------------------------------------
for t = 1:size(MDP.V,1)
    X = P*X;
end
X     = X > exp(-3);
C     = log(X.*(P*Y) + exp(-32));

return


function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);

% Luuks measuring difference between two matrices based on Kullback leibler
% divergence
function md = matrix_difference(p,q)
p   = p + 1e-16;
q   = q + 1e-16;

pl = log(p);    % matrix with log values p
ql = log(q);    % matrix with log values q   

dl = bsxfun(@minus,pl,ql); % difference log values (matrix)

kle = bsxfun(@times,p,dl); % kullback leibler divergence for each element

mdc = sum(kle,1);   % sum of columns kle matrix

md = sum(mdc);  % sum of all values