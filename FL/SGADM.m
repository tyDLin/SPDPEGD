% Stochastic GADM proposed by Xiang Gao.
function outputs = SGADM(samples, labels, opts)
time_min_run = opts.min_t;
F = opts.F; mu = opts.mu; beta = opts.beta; gamma = opts.gamma; C = opts.L1;
max_it = opts.max_it; checki = opts.checki;
if isfield(opts,'scaling')
    scaling = opts.scaling;
else
    scaling = 1;
end

% Initialization
t = cputime; time_solving = 0; betaFT = beta*F'; 
xs = []; times = []; iters = []; Num_i  = 0; [d,N] = size(samples);  rnd_pm = randperm(N);
x = zeros(d,1); lambda1 = zeros(d,1); lambda2 = zeros(d-1,1); xbar = zeros(d, 1);

done = false; k = 0;
while ~done
    % Randomly choose a sample
    idx = rnd_pm(mod(k,N)+1);
    sample = samples(:,idx);
    label  = labels(idx);
    
    % SGADM
    eta = scaling/(sqrt(k+1)+C);
    
    y = wthresh(x + lambda1/beta,'s', gamma/beta);
    z = wthresh(F*x + lambda2/beta,'s', mu/beta);
    t_exp  = exp(-label*sample'*x);
    if isnan(t_exp) || isinf(t_exp)
        d = 0;
    else
        d = 1/(1+t_exp);
    end
    x = x - eta*( (d-1)*label*sample + lambda1 + F'*lambda2 + beta*(x - y) + betaFT*(F*x -z));
    lambda1 = lambda1 - beta*(y - x);
    lambda2 = lambda2 - beta*(z - F*x);
    xbar = (k*xbar + x)/(k+1); 
    
    % Trace the log scaled results
    Num_i  = Num_i+1;
    if (Num_i >= checki)
        xs = [xs xbar];
        iters = [iters k];
        time_solving = cputime - t;
        times = [times time_solving];
        Num_i    = 0;
    end    
    
    % Stopping Criteriont
    k = k + 1;
    if k>max_it && time_solving>time_min_run; done = true; end;
end

trace = []; 
trace.checki = checki; 
trace.xs     = xs;
trace.times  = times;
trace.iters  = iters;

outputs.x      = xbar;
outputs.iter   = k; 
outputs.trace  = trace;
end
