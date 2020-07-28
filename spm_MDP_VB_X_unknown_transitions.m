function [MDP] = spm_MDP_VB_X(MDP,OPTIONS)
% This script is a modified version of “spm_MDP_VB_X” and is used by 
% “DEMO_MDP_maze_unknown_transitions” during policy executions.  Since 
% slight modifications are applied, the standard format of the script is 
% maintained. Most important modifications compared to the original script 
% can be found at lines:
% 
% 653 - 667, Transition novelty implementation (Section 3.2 in the thesis)
% 806 - 814, Alternative concentration parameter updating mechanism (Section 
%            5.3 in the thesis)

% 
% -------------------------------------------------------------------------
% Here follows the standard script description:
% active inference and learning using variational message passing
% FORMAT [MDP] = spm_MDP_VB_X(MDP,OPTIONS)
%
% Input; MDP(m,n)       - structure array of m models over n epochs
%
% MDP.V(T - 1,P,F)      - P allowable policies (T - 1 moves) over F factors
% MDP.T                 - number of outcomes
%
% MDP.A{G}(O,N1,...,NF) - likelihood of O outcomes given hidden states
% MDP.B{F}(NF,NF,MF)    - transitions among states under MF control states
% MDP.C{G}(O,T)         - (log) prior preferences for outcomes (modality G)
% MDP.D{F}(NF,1)        - prior probabilities over initial states
%
% MDP.b{F}              - concentration parameters for B
%
% optional:
% MDP.s(F,T)            - matrix of true states - for each hidden factor
% MDP.o(G,T)            - matrix of outcomes    - for each outcome modality
% or .O{G}(O,T)         - likelihood matrix     - for each outcome modality
% MDP.u(F,T - 1)        - vector of actions     - for each hidden factor
%
% MDP.alpha             - precision - action selection [512]
% MDP.beta              - precision over precision (Gamma hyperprior - [1])
% MDP.chi               - Occams window for deep updates
% MDP.tau               - time constant for gradient descent [4]
% MDP.eta               - learning rate for model parameters
% MDP.zeta              - Occam's window for polcies [3]
% MDP.erp               - resetting of initial states, to simulate ERPs [4]
%
% Outputs:
%
% MDP.P(M1,...,MF,T)    - probability of emitting action M1,.. over time
% MDP.Q{F}(NF,T,P)      - expected hidden states under each policy
% MDP.X{F}(NF,T)        - and Bayesian model averages over policies
% MDP.R(P,T)            - response: conditional expectations over policies
%
%
% MDP.F           - (P x T) (negative) free energies over time
% MDP.G           - (P x T) (negative) expected free energies over time
% MDP.H           - (1 x T) (negative) total free energy over time
% MDP.Fa          - (1 x 1) (negative) free energy of parameters (a)
% MDP.Fb          - ...
%
% This routine provides solutions of active inference (minimisation of
% variational free energy) using a generative model based upon a Markov
% decision process. The
% model and inference scheme is formulated in discrete space and time. This
% means that the generative model (and process) are  finite state machines
% or hidden Markov models whose dynamics are given by transition
% probabilities among states and the likelihood corresponds to a particular
% outcome conditioned upon hidden states.
%
% When supplied with outcomes, in terms of their likelihood (O) in the
% absence of any policy specification, this scheme will use variational
% message passing to optimise expectations about latent or hidden states
% (and likelihood (A) and prior (B) probabilities). In other words, it will
% invert a hidden Markov model. When  called with policies, it will
% generate outcomes that are used to infer optimal policies for active
% inference.
%
% This implementation equips agents with the prior beliefs that they will
% maximise expected free energy: expected free energy is the free energy of
% future outcomes under the posterior predictive distribution. This can be
% interpreted in several ways - most intuitively as minimising the KL
% divergence between predicted and preferred outcomes (specified as prior
% beliefs) - while simultaneously minimising ambiguity.
%
% This particular scheme is designed for any allowable policies or control
% sequences specified in MDP.V. Constraints on allowable policies can limit
% the numerics or combinatorics considerably. Further, the outcome space
% and hidden states can be defined in terms of factors; corresponding to
% sensory modalities and (functionally) segregated representations,
% respectively. This means, for each factor or subset of hidden states
% there are corresponding control states that determine the transition
% probabilities.
%
% This specification simplifies the generative model, allowing a fairly
% exhaustive model of potential outcomes. In brief, the agent encodes
% beliefs about hidden states in the past (and in the future) conditioned
% on each policy. The conditional expectations determine the (path
% integral) of free energy that then determines the prior over policies.
% This prior is used to create a predictive distribution over outcomes,
% which specifies the next action.
%
% In addition to state estimation and policy selection, the scheme also
% updates model parameters; including the state transition matrices,
% mapping to outcomes and the initial state. This is useful for learning
% the context. Likelihood and prior probabilities can be specified in terms
% of concentration parameters (of a Dirichlet distribution (a,b,c,..). If
% the corresponding (A,B,C,..) are supplied, they will be used to generate
% outcomes; unless called without policies (in hidden Markov model mode).
% In this case, the (A,B,C,..) are treated as posterior estimates.
%
% If supplied with a structure array, this routine will automatically step
% through the implicit sequence of epochs (implicit in the number of
% columns of the array). If the array has multiple rows, each row will be
% treated as a separate model or agent. This enables agents to communicate
% through acting upon a common set of hidden factors, or indeed sharing the
% same outcomes.
%
% See also: spm_MDP, which uses multiple future states and a mean field
% approximation for control states - but allows for different actions at
% all times (as in control problems).
%
% See also: spm_MDP_game_KL, which uses a very similar formulation but just
% maximises the KL divergence between the posterior predictive distribution
% over hidden states and those specified by preferences or prior beliefs.
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_X.m 7766 2020-01-05 21:37:39Z karl $


% deal with a sequence of trials
%==========================================================================

% options
%--------------------------------------------------------------------------
try, OPTIONS.plot;  catch, OPTIONS.plot  = 0; end
try, OPTIONS.gamma; catch, OPTIONS.gamma = 0; end
try, OPTIONS.D;     catch, OPTIONS.D     = 0; end

% check MDP specification
%--------------------------------------------------------------------------
MDP = spm_MDP_check(MDP);


% set up and preliminaries
%==========================================================================

% defaults
%--------------------------------------------------------------------------
try, alpha = MDP(1).alpha; catch, alpha = 512;  end % action precision
try, beta  = MDP(1).beta;  catch, beta  = 1;    end % policy precision
try, zeta  = MDP(1).zeta;  catch, zeta  = 3;    end % Occam window policies
try, eta   = MDP(1).eta;   catch, eta   = 10;    end % learning rate modyfied by luuk
try, tau   = MDP(1).tau;   catch, tau   = 4;    end % update time constant
try, chi   = MDP(1).chi;   catch, chi   = 1/64; end % Occam window updates
try, erp   = MDP(1).erp;   catch, erp   = 4;    end % update reset




% number of updates T & policies V (hidden Markov model with no policies)
%--------------------------------------------------------------------------
[T,V,HMM] = spm_MDP_get_T(MDP);

% initialise model-specific variables
%==========================================================================
Ni    = 16;                                % number of VB iterations
for m = 1:size(MDP,1)
    
    % ensure policy length is less than the number of updates
    %----------------------------------------------------------------------
    if size(V{m},1) > (T - 1)
        V{m} = V{m}(1:(T - 1),:,:);
    end
    
    % numbers of transitions, policies and states
    %----------------------------------------------------------------------
    Ng(m) = numel(MDP(m).A);               % number of outcome factors
    Nf(m) = numel(MDP(m).B);               % number of hidden state factors
    Np(m) = size(V{m},2);                  % number of allowable policies
    for f = 1:Nf(m)
        Ns(m,f) = size(MDP(m).B{f},1);     % number of hidden states
        Nu(m,f) = size(MDP(m).B{f},3);     % number of hidden controls
    end
    for g = 1:Ng(m)
        No(m,g) = size(MDP(m).A{g},1);     % number of outcomes
    end
    
    % parameters of generative model and policies
    %======================================================================
    
    % likelihood model (for a partially observed MDP)
    %----------------------------------------------------------------------
    for g = 1:Ng(m)
        
        % ensure probabilities are normalised  : A
        %------------------------------------------------------------------
        MDP(m).A{g} = spm_norm(MDP(m).A{g});
        
        % parameters (concentration parameters): a
        %------------------------------------------------------------------
        if isfield(MDP,'a')
            A{m,g}  = spm_norm(MDP(m).a{g});
        else
            A{m,g}  = spm_norm(MDP(m).A{g});
        end
        
        % prior concentration paramters for complexity (and novelty)
        %------------------------------------------------------------------
        if isfield(MDP,'a')
            pA{m,g} = MDP(m).a{g};
            wA{m,g} = spm_wnorm(MDP(m).a{g}).*(pA{m,g} > 0);
        end
        
    end
    
    % transition probabilities (priors)
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        for j = 1:Nu(m,f)
            
            % controlable transition probabilities : B
            %--------------------------------------------------------------
            MDP(m).B{f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j));
            
            % parameters (concentration parameters): b
            %--------------------------------------------------------------
            if isfield(MDP,'b') && ~HMM
                sB{m,f}(:,:,j) = spm_norm(MDP(m).b{f}(:,:,j) );
                rB{m,f}(:,:,j) = spm_norm(MDP(m).b{f}(:,:,j)');
            else
                sB{m,f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j) );
                rB{m,f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j)');
            end
            
        end
        
        % prior concentration paramters for complexity
        %------------------------------------------------------------------
        if isfield(MDP,'b')
            pB{m,f} = MDP(m).b{f};
            wB{m,f} = spm_wnorm(MDP(m).b{f}).*(pB{m,f} > 0); % modified by luuk
        end
        
    end
    
    % priors over initial hidden states - concentration parameters
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        if isfield(MDP,'d')
            D{m,f} = spm_norm(MDP(m).d{f});
        elseif isfield(MDP,'D')
            D{m,f} = spm_norm(MDP(m).D{f});
        else
            D{m,f} = spm_norm(ones(Ns(m,f),1));
            MDP(m).D{f} = D{m,f};
        end
        
        % prior concentration paramters for complexity
        %------------------------------------------------------------------
        if isfield(MDP,'d')
            pD{m,f} = MDP(m).d{f};
            wD{m,f} = spm_wnorm(MDP(m).d{f});
        end
    end
    
    % priors over policies - concentration parameters
    %----------------------------------------------------------------------
    if isfield(MDP,'e')
        E{m} = spm_norm(MDP(m).e);
    elseif isfield(MDP,'E')
        E{m} = spm_norm(MDP(m).E);
    else
        E{m} = spm_norm(ones(Np(m),1));
    end
    qE{m}    = spm_log(E{m});
    
    % prior concentration paramters for complexity
    %----------------------------------------------------------------------
    if isfield(MDP,'e')
        pE{m} = MDP(m).e;
    end
    
    % prior preferences (log probabilities) : C
    %----------------------------------------------------------------------
    for g = 1:Ng(m)
        if isfield(MDP,'c')
            C{m,g}  = spm_psi(MDP(m).c{g} + 1/32);
            pC{m,g} = MDP(m).c{g};
        elseif isfield(MDP,'C')
            C{m,g}  = MDP(m).C{g};
        else
            C{m,g}  = zeros(No(m,g),1);
        end
        
        % assume time-invariant preferences, if unspecified
        %------------------------------------------------------------------
        if size(C{m,g},2) == 1
            C{m,g} = repmat(C{m,g},1,T);
            if isfield(MDP,'c')
                MDP(m).c{g} = repmat(MDP(m).c{g},1,T);
                pC{m,g}     = repmat(pC{m,g},1,T);
            end
        end
        C{m,g} = spm_log(spm_softmax(C{m,g}));
    end
    
    % initialise  posterior expectations of hidden states
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        xn{m,f} = zeros(Ni,Ns(m,f),1,1,Np(m)) + 1/Ns(m,f);
        vn{m,f} = zeros(Ni,Ns(m,f),1,1,Np(m));
        x{m,f}  = zeros(Ns(m,f),T,Np(m))      + 1/Ns(m,f);
        X{m,f}  = repmat(D{m,f},1,1);
        for k = 1:Np(m)
            x{m,f}(:,1,k) = D{m,f};
        end
    end
    
    % initialise posteriors over polices and action
    %----------------------------------------------------------------------
    P{m}  = zeros([Nu(m,:),1]);
    un{m} = zeros(Np(m),1);
    u{m}  = zeros(Np(m),1);
    
    
    % if states have not been specified set to 0
    %----------------------------------------------------------------------
    s{m}  = zeros(Nf(m),T);
    try
        i       = find(MDP(m).s);
        s{m}(i) = MDP(m).s(i);
    end
    MDP(m).s    = s{m};
    
    % if outcomes have not been specified set to 0
    %----------------------------------------------------------------------
    o{m}  = zeros(Ng(m),T);
    try
        i = find(MDP(m).o);
        o{m}(i) = MDP(m).o(i);
    end
    MDP(m).o = o{m};
    
    % (indices of) plausible (allowable) policies
    %----------------------------------------------------------------------
    p{m}  = 1:Np(m);
    
    % expected rate parameter (precision of posterior over policies)
    %----------------------------------------------------------------------
    qb{m} = beta;                          % initialise rate parameters
    w{m}  = 1/qb{m};                       % posterior precision (policy)
    
end

% ensure any outcome generating agent is updated first
%--------------------------------------------------------------------------
[M,MDP] = spm_MDP_get_M(MDP,T,Ng);


MDP(m).F_result2 = 0;


MDP(m).S_avg = 0;
MDP(m).TN_avg = 0;
MDP(m).Q_s = zeros(size(V,1),1);
MDP(m).Q_tn = zeros(size(V,1),1);
MDP(m).Q_f = zeros(size(V,1),1);
MDP(m).Q_ln = zeros(size(V,1),1);
MDP(m).Q_ps = zeros(size(V,1),1);



% belief updating over successive time points
%==========================================================================
for t = 1:T
    
    % generate hidden states and outcomes for each agent or model
    %======================================================================
    for m = M(t,:)
        
        if ~HMM % not required for HMM
            
            % sample state, if not specified
            %--------------------------------------------------------------
            for f = 1:Nf(m)
                
                % the next state is generated by action on external states
                %----------------------------------------------------------
                if MDP(m).s(f,t) == 0
                    if t > 1
                        ps = MDP(m).B{f}(:,MDP(m).s(f,t - 1),MDP(m).u(f,t - 1));
                    else
                        ps = spm_norm(MDP(m).D{f});
                    end
                    MDP(m).s(f,t) = find(rand < cumsum(ps),1);
                end
                
            end            
            
            % posterior predictive density over hidden (external) states
            %--------------------------------------------------------------
            for f = 1:Nf(m)
                
                % under selected action (xqq)
                %----------------------------------------------------------
                if t > 1
                    xqq{m,f} = sB{m,f}(:,:,MDP(m).u(f,t - 1))*X{m,f}(:,t - 1);
                else
                    xqq{m,f} = X{m,f}(:,t);
                end
                
                % Bayesian model average (xq)
                %----------------------------------------------------------
                xq{m,f} = X{m,f}(:,t);
                
            end
            
            % sample outcome, if not specified
            %--------------------------------------------------------------
            for g = 1:Ng(m)
                
                % if outcome is not specified
                %----------------------------------------------------------
                if ~MDP(m).o(g,t)
                    
                    % outcome is generated by model n
                    %------------------------------------------------------
                    if MDP(m).n(g,t)

                        n    = MDP(m).n(g,t);
                        if n == m
                            
                            % outcome that minimises free energy (i.e.,
                            % maximises accuracy)
                            %----------------------------------------------
                            F             = spm_dot(spm_log(A{m,g}),xqq(m,:));
                            po            = spm_softmax(F*512);
                            MDP(m).o(g,t) = find(rand < cumsum(po),1);
                            
                        else
                            
                            % outcome from model n
                            %----------------------------------------------
                            MDP(m).o(g,t) = MDP(n).o(g,t);
                            
                        end
                        
                    else

                        % or sample from likelihood given hidden state
                        %--------------------------------------------------
                        ind           = num2cell(MDP(m).s(:,t));
                        po            = MDP(m).A{g}(:,ind{:});
                        MDP(m).o(g,t) = find(rand < cumsum(po),1);
                        
                    end
                end
            end
            
        end % HMM
        
        % get probabilistic outcomes from samples or subordinate level
        %==================================================================
        
        % get outcome likelihood (O{m})
        %------------------------------------------------------------------
        for g = 1:Ng(m)
                % specified as the sampled outcome
                %----------------------------------------------------------
                O{m}{g,t} = sparse(MDP(m).o(g,t),1,1,No(m,g),1);
        end
        
        % Likelihood of hidden states
        %==================================================================
        L{m,t} = 1;
        for g = 1:Ng(m)
            L{m,t} = L{m,t}.*spm_dot(A{m,g},O{m}{g,t});
        end
        
        
        % Variational updates (skip to t = T in HMM mode)
        %==================================================================
        if ~HMM || T == t
            
            % eliminate unlikely policies
            %--------------------------------------------------------------
            if ~isfield(MDP,'U') && t > 1
                F    = log(u{m}(p{m},t - 1));
                p{m} = p{m}((F - max(F)) > -zeta);
            end
            
            % processing time and reset
            %--------------------------------------------------------------
            tstart = tic;
            for f = 1:Nf(m)
                x{m,f} = spm_softmax(spm_log(x{m,f})/erp);
            end
            
            % Variational updates (hidden states) under sequential policies
            %==============================================================
            
            
            % variational message passing (VMP)
            %--------------------------------------------------------------
            S     = size(V{m},1) + 1;   % horizon
            R = S;
            F     = zeros(Np(m),1);
            Q_f =  zeros(Np(m),1);
            for k = p{m}                % loop over plausible policies
                dF    = 1;              % reset criterion for this policy
                for i = 1:Ni            % iterate belief updates
                    F(k)  = 0;          % reset free energy for this policy
                    Q_f(k) = 0;
                    for j = 1:S         % loop over future time points
                        
                        % curent posterior over outcome factors
                        %--------------------------------------------------
                        if j <= t
                            for f = 1:Nf(m)
                                xq{m,f} = full(x{m,f}(:,j,k));
                            end
                        end
                        
                        for f = 1:Nf(m)
                            
                            % hidden states for this time and policy
                            %----------------------------------------------
                            sx = full(x{m,f}(:,j,k));
                            qL = zeros(Ns(m,f),1);
                            v  = zeros(Ns(m,f),1);
                            
                            % evaluate free energy and gradients (v = dFdx)
                            %----------------------------------------------
                            if dF > exp(-8) || i > 4
                                
                                % marginal likelihood over outcome factors
                                %------------------------------------------
                                if j <= t
                                    qL = spm_dot(L{m,j},xq(m,:),f);
                                    qL = spm_log(qL(:));
                                end
                                
                                % entropy
                                %------------------------------------------
                                qx  = spm_log(sx);
                                
                                % emprical priors (forward messages)
                                %------------------------------------------
                                if j < 2
                                    px = spm_log(D{m,f});
                                    v  = v + px + qL - qx;
                                else
                                    px = spm_log(sB{m,f}(:,:,V{m}(j - 1,k,f))*x{m,f}(:,j - 1,k));
                                    v  = v + px + qL - qx;
                                end
                                
                                % emprical priors (backward messages)
                                %------------------------------------------
                                if j < R
                                    px = spm_log(rB{m,f}(:,:,V{m}(j    ,k,f))*x{m,f}(:,j + 1,k));
                                    v  = v + px + qL - qx;
                                end
                                
                                % (negative) free energy
                                %------------------------------------------
                                if j == 1 || j == S
                                    F(k) = F(k) + sx'*0.5*v;
                                else
                                    F(k) = F(k) + sx'*(0.5*v - (Nf(m)-1)*qL/Nf(m));
                                end
                                Q_f(k) = F(k);
                                % update
                                %------------------------------------------
                                v    = v - mean(v);
                                sx   = spm_softmax(qx + v/tau);
                                
                            else
                                F(k) = G(k);
                            end
                            
                            % store update neuronal activity
                            %----------------------------------------------
                            x{m,f}(:,j,k)      = sx;
                            xq{m,f}            = sx;
                            xn{m,f}(i,:,j,t,k) = sx;
                            vn{m,f}(i,:,j,t,k) = v;
                            
                        end
                    end
                    
                    % convergence
                    %------------------------------------------------------
                    if i > 1
                        dF = F(k) - G(k);
                    end
                    G = F;
                    
                end
            end
            
            % accumulate expected free energy of policies (Q)
            %==============================================================
            pu  = 1;                               % empirical prior
            qu  = 1;                               % posterior
            Q   = zeros(Np(m),1);                  % expected free energy
            Q_ss = zeros(Np(m),1);
            Q_ps = zeros(Np(m),1);
            Q_ln = zeros(Np(m),1);
            Q_tn = zeros(Np(m),1);
            if Np(m) > 1
                for k = p{m}
                    
                    % Bayesian surprise about inital conditions
                    %------------------------------------------------------
                    if isfield(MDP,'d')
                        for f = 1:Nf(m)
                            Q(k) = Q(k) - spm_dot(wD{m,f},x{m,f}(:,1,k));
                        end
                    end
                    
                    for j = t:S
                        
                        % get expected states for this policy and time
                        %--------------------------------------------------
                        for f = 1:Nf(m)
                            xq{m,f} = x{m,f}(:,j,k);
                        end
                        
                        % (negative) expected free energy
                        %==================================================
                        
                        % Bayesian surprise about states
                        %--------------------------------------------------
                        Q(k) = Q(k) + spm_MDP_G(A(m,:),xq(m,:));
                        Q_ss(k) = Q_ss(k) + spm_MDP_G(A(m,:),xq(m,:));
                        
                        for g = 1:Ng(m)
                            
                            % prior preferences about outcomes
                            %----------------------------------------------
                            qo   = spm_dot(A{m,g},xq(m,:));
                            Q(k) = Q(k) + qo'*(C{m,g}(:,j));
                            Q_ps(k) = Q_ps(k) + qo'*(C{m,g}(:,j));
                            
                            % Bayesian surprise about likelihood parameters
                            %----------------------------------------------
                            if isfield(MDP,'a')
                                Q(k) = Q(k) - spm_dot(wA{m,g},{qo xq{m,:}});
                                Q_ln(k) = Q_ln(k) - spm_dot(wA{m,g},{qo xq{m,:}});
                                
                            end
                        end
                        
                        %%%%%%%%%%%%%%%%%%MODFIED PART%%%%%%%%%%%%%%%
                        %Bayesian surprise about transition parameters
                        %----------------------------------------------
                        for f = 1:Nf(m)                                                             % For eacht factor, our environment deals with only 1 factor: location
                            if isfield(MDP,'b')                                                     % If prior and posterior transition variables are defined 
                                if j < S                                                            
                                action = V{m}(j,k,f);                                               % Select the action of the current policy
                                qsn = x{m,f}(:,j + 1, k); % Friston's advise described in mail      % Define the belief state distribution for the next time step, given the policy
                                Q(k) = Q(k) - spm_dot(wB{m,f}(:,:,action),{qsn xq{m,:}});           % Calculate transition novelty for a time step (Equation 3-14 in the thesis)
                                Q_tn(k) = Q_tn(k) - spm_dot(wB{m,f}(:,:,action),{qsn xq{m,:}});     % record transition novelty value for later measurements
                                end
                            end
                        end
                        %%%%%%%%%%%%%%%%%%MODFIED PART%%%%%%%%%%%%%%%
                    end
                end
                
                
                % variational updates - policies and precision
                %==========================================================
                
                % previous expected precision
                %----------------------------------------------------------
                if t > 1
                    w{m}(t) = w{m}(t - 1);
                end
                for i = 1:Ni
                    
                    % posterior and prior beliefs about policies
                    %------------------------------------------------------
                    qu = spm_softmax(qE{m}(p{m}) + w{m}(t)*Q(p{m}) + F(p{m}));
                    pu = spm_softmax(qE{m}(p{m}) + w{m}(t)*Q(p{m}));
                    
                    % precision (w) with free energy gradients (v = -dF/dw)
                    %------------------------------------------------------
                    if OPTIONS.gamma
                        w{m}(t) = 1/beta;
                    else
                        eg      = (qu - pu)'*Q(p{m});
                        dFdg    = qb{m} - beta + eg;
                        qb{m}   = qb{m} - dFdg/2;
                        w{m}(t) = 1/qb{m};
                    end
                    
                    % simulated dopamine responses (expected precision)
                    %------------------------------------------------------
                    n             = (t - 1)*Ni + i;
                    wn{m}(n,1)    = w{m}(t);
                    un{m}(p{m},n) = qu;
                    u{m}(p{m},t)  = qu;
                    
                end               
            end % end of loop over multiple policies

            % Bayesian model averaging of hidden states (over policies)
            %--------------------------------------------------------------
            for f = 1:Nf(m)
                for i = 1:S
                    X{m,f}(:,i) = reshape(x{m,f}(:,i,:),Ns(m,f),Np(m))*u{m}(:,t);
                end
            end
            
            % Estimation of contribution of free energy terms
            if t < T
            MDP(m).F_result2 = MDP(m).F_result2 + qu'*-F(p{m});
            MDP(m).S_avg = MDP(m).S_avg + qu'*(-Q_ss(p{m}));
            MDP(m).TN_avg = MDP(m).TN_avg + qu'*(-Q_tn(p{m}));
            end
            % processing (i.e., reaction) time
            %--------------------------------------------------------------
            rt{m}(t)      = toc(tstart);
            
            % record (negative) free energies
            %--------------------------------------------------------------
            MDP(m).F(:,t) = F;
            MDP(m).G(:,t) = Q;
            MDP(m).H(1,t) = qu'*MDP(m).F(p{m},t) - qu'*(spm_log(qu) - spm_log(pu));
            
            
            % check for end of sentence (' ') if in VOX mode
            %--------------------------------------------------------------
            XVOX = 0;
            
            % action selection
            %==============================================================
            if t < T
                
                % marginal posterior over action (for each factor)
                %----------------------------------------------------------
                Pu    = zeros([Nu(m,:),1]);
                for i = 1:Np(m)
                    sub        = num2cell(V{m}(t,i,:));
                    Pu(sub{:}) = Pu(sub{:}) + u{m}(i,t);
                end
                
                % action selection (softmax function of action potential)
                %----------------------------------------------------------
                sub            = repmat({':'},1,Nf(m));
                Pu(:)          = spm_softmax(alpha*log(Pu(:)));
                P{m}(sub{:},t) = Pu;
                
                test_u = spm_softmax(alpha*u{m}(:,t),2);
                % next action - sampled from marginal posterior
                %----------------------------------------------------------
                try
                    MDP(m).u(:,t) = MDP(m).u(:,t);
                catch
                    if MDP(m).learning == 1
                        ind           = find(rand < cumsum(Pu(:)),1);
                        MDP(m).u(:,t) = spm_ind2sub(Nu(m,:),ind);
                    elseif MDP(m).learning == 2
                        test_ind = find(rand < cumsum(test_u(:)),1); %luuk edit
                        test_ind2 = V{m}(t,test_ind,:); %luuk edit
                        MDP(m).u(:,t) = spm_ind2sub(Nu(m,:),test_ind2); %luuk edit
                    end 
                end
            end % end of state and action selection
        end % end of variational updates over time
    end % end of loop over models (agents)
    
    % terminate evidence accumulation
    %----------------------------------------------------------------------
    if t == T
        if T == 1
            MDP(m).u  = zeros(Nf(m),0);
        end
        if ~HMM
            MDP(m).o  = MDP(m).o(:,1:T);        % outcomes at 1,...,T
            MDP(m).s  = MDP(m).s(:,1:T);        % states   at 1,...,T
            MDP(m).u  = MDP(m).u(:,1:T - 1);    % actions  at 1,...,T - 1
        end
        break;
    end
    
end % end of loop over time

% learning - accumulate concentration parameters
%==========================================================================
for m = 1:size(MDP,1)
    
    for t = 1:T
        
        % mapping from hidden states to hidden states: b(u)
        %------------------------------------------------------------------
        if isfield(MDP,'b') && t > 1
            for f = 1:Nf(m)
                if MDP(m).learning == 1
                for k = 1:Np(m)
                    v   = V{m}(t - 1,k,f);
                    db  = u{m}(k,t)*x{m,f}(:,t,k)*x{m,f}(:,t - 1,k)';
                    db  = db.*(MDP(m).b{f}(:,:,v) > 0);
                    MDP(m).b{f}(:,:,v) = MDP(m).b{f}(:,:,v) + db*eta;
                end
                %%%%%%%%%%%%%%%%%%MODFIED PART%%%%%%%%%%%%%%%
                elseif MDP(m).learning == 2
                v = MDP.u(t-1);                                     % Select action in previous time step
                db = x{m,f}(:,t,k)*x{m,f}(:,t - 1,k)';              % combine state with subsequent state
                db = db.*(MDP(m).b{f}(:,:,v) > 0);                  % Check whether b has values for state-subsequent state combination
                MDP(m).b{f}(:,:,v) = MDP(m).b{f}(:,:,v) + db*eta;   % Add state-subsequent state to the original learning matrix (multiplied with learning factor eta)
                end
                %%%%%%%%%%%%%%%%%%MODFIED PART%%%%%%%%%%%%%%%
            end
        end
    end
    
    
    % (negative) free energy of parameters: state specific
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        if isfield(MDP,'b')
            MDP(m).Fb(f) = - spm_KL_dir(MDP(m).b{f},pB{m,f});
        end
    end
    
    % simulated dopamine (or cholinergic) responses
    %----------------------------------------------------------------------
    if Np(m) > 1
        dn{m} = 8*gradient(wn{m}) + wn{m}/8;
    else
        dn{m} = [];
        wn{m} = [];
    end
    
    % Bayesian model averaging of expected hidden states over policies
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        if ~XVOX
            Xn{m,f} = zeros(Ni,Ns(m,f),T,T);
            Vn{m,f} = zeros(Ni,Ns(m,f),T,T);
        end
        for i = 1:T
            for k = 1:Np(m)
                Xn{m,f}(:,:,1:T,i) = Xn{m,f}(:,:,1:T,i) + xn{m,f}(:,:,1:T,i,k)*u{m}(k,i);
                Vn{m,f}(:,:,1:T,i) = Vn{m,f}(:,:,1:T,i) + vn{m,f}(:,:,1:T,i,k)*u{m}(k,i);
            end
        end
    end
    
    
    % assemble results and place in NDP structure
    %----------------------------------------------------------------------
    MDP(m).T  = T;            % number of belief updates
    MDP(m).O  = O{m};         % outcomes
    MDP(m).P  = P{m};         % probability of action at time 1,...,T - 1
    MDP(m).R  = u{m};         % conditional expectations over policies
    MDP(m).Q  = x(m,:);       % conditional expectations over N states
    MDP(m).X  = X(m,:);       % Bayesian model averages over T outcomes
    MDP(m).C  = C(m,:);       % utility
    
    
    MDP(m).w  = w{m};         % posterior expectations of precision (policy)
    MDP(m).vn = Vn(m,:);      % simulated neuronal prediction error
    MDP(m).xn = Xn(m,:);      % simulated neuronal encoding of hidden states
    MDP(m).un = un{m};        % simulated neuronal encoding of policies
    MDP(m).wn = wn{m};        % simulated neuronal encoding of precision
    MDP(m).dn = dn{m};        % simulated dopamine responses (deconvolved)
    MDP(m).rt = rt{m};        % simulated reaction time (seconds)
    
end



% auxillary functions
%==========================================================================

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);

function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);

function A  = spm_wnorm(A)
% summation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A   = A + 1e-16;
A   = bsxfun(@minus,1./sum(A,1),1./A)/2;

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
%--------------------------------------------------------------------------
n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end

return


function [T,V,HMM] = spm_MDP_get_T(MDP)
% FORMAT [T,V,HMM] = spm_MDP_get_T(MDP)
% returns number of policies, policy cell array and HMM flag
% MDP(m) - structure array of m MPDs
% T      - number of trials or updates
% V(m)   - indices of policies for m-th MDP
% HMM    - flag indicating a hidden Markov model
%
% This subroutine returns the policy matrix as a cell array (for each
% model) and the maximum number of updates. If outcomes are specified
% probabilistically in the field MDP(m).O, and there is only one policy,
% the partially observed MDP reduces to a hidden Markov model.
%__________________________________________________________________________

for m = 1:size(MDP,1)
    
    % check for policies: hidden Markov model, with a single policy
    %----------------------------------------------------------------------
    if isfield(MDP(m),'U')
        HMM = size(MDP(m).U,1) < 2;
    elseif isfield(MDP(m),'V')
        HMM = size(MDP(m).V,2) < 2;
    else
        HMM = 1;
    end
    
    if isfield(MDP(m),'O') && ~any(MDP(m).o(:)) && HMM
        
        % probabilistic outcomes - assume hidden Markov model (HMM)
        %------------------------------------------------------------------
        T(m) = size(MDP(m).O{1},2);         % HMM mode
        V{m} = ones(T - 1,1);               % single 'policy'
        HMM  = 1;
        
    elseif isfield(MDP(m),'U')
        
        % called with repeatable actions (U,T)
        %------------------------------------------------------------------
        T(m) = MDP(m).T;                    % number of updates
        V{m}(1,:,:) = MDP(m).U;             % allowable actions (1,Np,Nf)
        HMM  = 0;
        
    elseif isfield(MDP(m),'V')
        
        % full sequential policies (V)
        %------------------------------------------------------------------
        V{m} = MDP(m).V;                    % allowable policies (T - 1,Np,Nf)
        T(m) = size(MDP(m).V,1) + 1;        % number of transitions
        HMM  = 0;
        
    else
        sprintf('Please specify MDP(%d).U, MDP(%d).V or MDP(%d).O',m), return
    end
    
end

% number of time steps
%--------------------------------------------------------------------------
T = max(T);

return


function [M,MDP] = spm_MDP_get_M(MDP,T,Ng)
% FORMAT [M,MDP] = spm_MDP_get_M(MDP,T,Ng)
% returns an update matrix for multiple models
% MDP(m) - structure array of m MPDs
% T      - number of trials or updates
% Ng(m)  - number of output modalities for m-th MDP
%
% M      - update matrix for multiple models
% MDP(m) - structure array of m MPDs
%
% In some applications, the outcomes are generated by a particular model
% (to maximise free energy, based upon the posterior predictive density).
% The generating model is specified in the matrix MDP(m).n, with a row for
% each outcome modality, such that each row lists the index of the model
% responsible for generating outcomes.
%__________________________________________________________________________

    
for m = 1:size(MDP,1)
    
    % check size of outcome generating agent, as specified by MDP(m).n
    %----------------------------------------------------------------------
    if ~isfield(MDP(m),'n')
        MDP(m).n = zeros(Ng(m),T);
    end
    if size(MDP(m).n,1) < Ng(m)
        MDP(m).n = repmat(MDP(m).n(1,:),Ng(m),1);
    end
    if size(MDP(m).n,1) < T
        MDP(m).n = repmat(MDP(m).n(:,1),1,T);
    end
    
    % mode of generating model (most frequent over outcome modalities)
    %----------------------------------------------------------------------
    n(m,:) = mode(MDP(m).n.*(MDP(m).n > 0),1);
    
end

% reorder list of model indices for each update
%--------------------------------------------------------------------------
n     = mode(n,1);
for t = 1:T
    if n(t) > 0
        M(t,:) = circshift((1:size(MDP,1)),[0 (1 - n(t))]);
    else
        M(t,:) = 1;
    end
end


return

function MDP = spm_MDP_update(MDP,OUT)
% FORMAT MDP = spm_MDP_update(MDP,OUT)
% moves Dirichlet parameters from OUT to MDP
% MDP - structure array (new)
% OUT - structure array (old)
%__________________________________________________________________________

% check for concentration parameters at this level
%--------------------------------------------------------------------------
try,  MDP.a = OUT.a; end
try,  MDP.b = OUT.b; end
try,  MDP.c = OUT.c; end
try,  MDP.d = OUT.d; end
try,  MDP.e = OUT.e; end

% check for concentration parameters at nested levels
%--------------------------------------------------------------------------
try,  MDP.MDP(1).a = OUT.mdp(end).a; end
try,  MDP.MDP(1).b = OUT.mdp(end).b; end
try,  MDP.MDP(1).c = OUT.mdp(end).c; end
try,  MDP.MDP(1).d = OUT.mdp(end).d; end
try,  MDP.MDP(1).e = OUT.mdp(end).e; end

return




function md = matrix_difference(p,q)
p   = p + 1e-16;
q   = q + 1e-16;

pl = log(p);    % matrix with log values p
ql = log(q);    % matrix with log values q   

dl = bsxfun(@minus,pl,ql); % difference log values (matrix)

kle = bsxfun(@times,p,dl); % kullback leibler divergence for each element

mdc = sum(kle,1);   % sum of columns kle matrix

md = sum(mdc);  % sum of all values
