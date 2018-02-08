%Algorithm EGADM
function outputs = EGADM(samples, labels, opts)
time_min_run = opts.min_t;
F = opts.F; mu = opts.mu; gamma = opts.gamma; beta = 1/opts.L2;
checki = opts.checki; max_it = opts.epochs;

% Initialization
t = cputime; time_solving = 0;
xs = []; times = []; iters = []; [d, N]= size(samples); 
x = zeros(d,1); lambda1 = zeros(d,1);  lambda2 = zeros(d-1,1); xbar = zeros(d, 1); 

done = false; k = 0;
while ~done
    
    % EGADM
    y = wthresh(x + lambda1/beta,'s', gamma/beta);
    z = wthresh(F*x + lambda2/beta,'s', mu/beta);
    
    gradient = zeros(d, 1);
    for i=1:N
        temp  = exp(-labels(i)*samples(:,i)'*x);
        if isnan(temp) || isinf(temp)
            gradient = gradient -labels(i)*samples(:,i);
        else
            gradient = gradient -labels(i)*samples(:,i)*temp/(1+temp);
        end
    end
    gradient = gradient/N; 
    
    x_hat = x - beta *(gradient + lambda1 + F'*lambda2);  
    lambda1_hat = lambda1 - beta*(y - x);
    lambda2_hat = lambda2 - beta*(z - F*x);
    
    gradient = zeros(d, 1);
    for i=1:N
        temp = exp(-labels(i)*samples(:,i)'*x_hat);
        if isnan(temp) || isinf(temp)
            gradient = gradient -labels(i)*samples(:,i);
        else
            gradient = gradient -labels(i)*samples(:,i)*temp/(1+temp);
        end
    end
    gradient = gradient/N;
    
    x = x - beta*(gradient + lambda1_hat + F'*lambda2_hat);
    lambda1 = lambda1 - beta*(y - x_hat);
    lambda2 = lambda2 - beta*(z - F*x_hat);
    
    xbar = (k*xbar + x_hat)/(k+1);
    
    % Trace the log scaled results
    xs = [xs xbar];
    iters        = [iters k+1];
    time_solving = cputime - t;
    times        = [times time_solving];    
    
    % Stopping Criterion
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

