function [F_671011] = calculate_F671011(MDP) 
%%Measure the effect of salience
%I decided that F is a good way to measure salience since it is based on
%the prediction error in state distribution. The only problem is that not
%every state is visited, therefore, after the trials (in which salience
%must try to optimise s, because it is looking for information gain), we
%place the robot in 4 places in the environment and calculate the
%variational free energy.............


% Needed: 

try, tau   = MDP(1).tau;   catch, tau   = 4;    end
HMM = 0;
Ni    = 16;
T = 3; %dacht ik
t = 1;
m = 1;
Nf = 1;
Ns = 16;
Ng = 2;
V{m} = MDP(m).V; 
Np(m) = size(V{m},2);
Nu = 5;
nbs = 4;
o_new = ones(2,1,4);
o_new(2,1,1) = 7;
o_new(2,1,2) = 6;
o_new(2,1,3) = 11;
o_new(2,1,4) = 10;
Ftotal = zeros(25,4);
load('x_7');
load('x_6');
load('x_11');
load('x_10');
actual{1}.x_actual = x_7;
actual{2}.x_actual = x_6;
actual{3}.x_actual = x_11;
actual{4}.x_actual = x_10;
S_d = zeros(4,1);


    for g = 1:Ng(m)
        No(m,g) = size(MDP(m).A{g},1);     % number of outcomes
    end

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
        
        
    end
    
    % transition probabilities (priors)
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        for j = 1:Nu(m,f)
            
            % controlable transition probabilities : B
            %--------------------------------------------------------------
            %MDP(m).B{f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j));
            
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
    end

    d7 = zeros(16,1) + 1e-16;
    d7(7) = 1;
    d7 = spm_norm(d7);
    d6 = zeros(16,1) + 1e-16;
    d6(6) = 1;
    d6 = spm_norm(d6);
    d11 = zeros(16,1) + 1e-16;
    d11(11) = 1;
    d11 = spm_norm(d11);
    d10 = zeros(16,1) + 1e-16;
    d10(10) = 1;
    d10 = spm_norm(d10);
    
    D{m,f,1} = d7;
    D{m,f,2} = d6;
    D{m,f,3} = d11;
    D{m,f,4} = d10;

for ms = 1:4


%        if isfield(MDP,'d')
%            D{m,f} = spm_norm(MDP(m).d{f});
%        elseif isfield(MDP,'D')
%            D{m,f} = spm_norm(MDP(m).D{f});
%        else
           % D{m,f} = spm_norm(ones(Ns(m,f),1));
%            MDP(m).D{f} = D{m,f};
%        end


    % initialise  posterior expectations of hidden states
    %----------------------------------------------------------------------
    for f = 1:Nf(m)
        %xn{m,f} = zeros(Ni,Ns(m,f),1,1,Np(m)) + 1/Ns(m,f);
        %vn{m,f} = zeros(Ni,Ns(m,f),1,1,Np(m));
        x{m,f}  = zeros(Ns(m,f),T,Np(m))      + 1/Ns(m,f);
        %X{m,f}  = repmat(D{m,f},1,1);
        for k = 1:Np(m)
            x{m,f}(:,1,k) = D{m,f,ms};
        end
    end


Np(m) = size(V{m},2);                  % number of allowable policies

    % (indices of) plausible (allowable) policies
    %----------------------------------------------------------------------
    p{m}  = 1:Np(m);
    
    
        % get outcome likelihood (O{m})
        %------------------------------------------------------------------
        for g = 1:Ng(m)
            
            % specified as a likelihood or observation
            %--------------------------------------------------------------
            if HMM
                
                % specified as a likelihood(HMM)
                %----------------------------------------------------------
                O{m}{g,t} = MDP(m).O{g}(:,t);
                
            else
                
                % specified as the sampled outcome
                %----------------------------------------------------------
                O{m}{g,t} = sparse(o_new(g,t,ms),1,1,No(m,g),1);
            end
            
        end
                 
    
    
    
    
    
    
    
    
    
        L{m,t} = 1;
        for g = 1:Ng(m)
            L{m,t} = L{m,t}.*spm_dot(A{m,g},O{m}{g,t});
        end
    
    
            S     = size(V{m},1) + 1; 
            R = S;
            F     = zeros(Np(m),1);
            for k = p{m}                % loop over plausible policies
                dF    = 1;              % reset criterion for this policy
                for i = 1:Ni            % iterate belief updates
                    F(k)  = 0;          % reset free energy for this policy
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
                            %if dF > exp(-8) || i > 4
                                
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
                                    px = spm_log(D{m,f,ms});
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
                                % update
                                %------------------------------------------
                                v    = v - mean(v);
                                sx   = spm_softmax(qx + v/tau);
                                
                            %else
                             %   F(k) = G(k);
                            %end
                            
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
            %Ftotal(:,ms) = F(:);
            S_d(ms) = matrix_difference(actual{1,ms}.x_actual{1}(:,:,:),x{1}(:,:,:));
            
end

%F_1 = sum(Ftotal,1);
F_671011 = sum(S_d);



function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);

function md = matrix_difference(p,q)
p   = p + 1e-16;
q   = q + 1e-16;

pl = log(p);    % matrix with log values p
ql = log(q);    % matrix with log values q   

dl = bsxfun(@minus,pl,ql); % difference log values (matrix)

kle = bsxfun(@times,p,dl); % kullback leibler divergence for each element

mdc = sum(kle,1);   % sum of columns kle matrix

mdt = sum(mdc);% sum of all values

md = sum(mdt);

