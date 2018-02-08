% Graph-Guided Logistic Regression Task
clear;close all;

funfcn_batch = {@EGADM};
funfcn_stoc  = {@SGADM,@STOC_ADMM,@RDA_ADMM,@OPG_ADMM,@Ada_SADMMdiag,@Ada_SADMMfull,@PEGSADM_Uniform,@PEGSADM_Nonuniform};
funfcns      = [funfcn_batch, funfcn_stoc];
datasets     = {'classic', 'hitech', 'k1b', 'la12', 'la1', 'la2', 'ng3sim', 'ohscal', 'reviews', 'sports'};

n_epochs     = 5;
for idx_datasets = 1:length(datasets)
    dataset_name = datasets{idx_datasets};
    fprintf('\nNow processing (%d/%d) dataset: %s\n',idx_datasets,length(datasets),dataset_name);
     try
        data_path = 'E:\Documents\Datasets\mat_datasets\LIBSVM\';
        load([data_path dataset_name '.mat']);
    catch
        data_path = 'E:\Documents\Datasets\mat_datasets\docdatasets\';
        dataset   = load([data_path dataset_name '.mat']);
    end
    
    % Samples
    K_fold = 5;
    if( exist(['idx_cv_' dataset_name '_' num2str(K_fold) '.mat'],'file') ==0 )
        idx_cv = crossvalind('Kfold',length(labels),K_fold);
        save(['idx_cv_' dataset_name '_' num2str(K_fold) '.mat'], 'idx_cv');
    else
        idx_cv = load(['idx_cv_' dataset_name '_' num2str(K_fold) '.mat'], 'idx_cv');
        idx_cv = idx_cv.idx_cv;
    end
    
    [d, N] = size(samples);
    %Graphical Matrix generation
    if( exist(['temp_F_' dataset_name '.mat'],'file') ==0 )
        S  = cov(samples');
        rho = 0.005; % weighting parameter and the parameters can be tuned
        opts.mxitr = 500; opts.mu0 = 1e-1;  opts.muf = 1e-3; opts.rmu = 1/4;
        opts.tol_gap = 1e-1; opts.tol_frel = 1e-7; opts.tol_Xrel = 1e-7; opts.tol_Yrel = 1e-7;
        opts.numDG = 10; opts.record = 0;opts.sigma = 1e-10;
        out = SICS_ALM(S,rho,opts);
        X = out.X; X(abs(X) > 2.5e-3) = 1; X(abs(X) < 2.5e-3) = 0; F = -tril(X,-1) + triu(X,1);
        save(['temp_F_' dataset_name '.mat'], 'F','S');
    else
        load(['temp_F_' dataset_name '.mat'],'F','S');
    end
    
    for idx_fold = 1:K_fold
        idx_test     = (idx_cv == idx_fold);
        idx_train    = ~idx_test;
        s_train      = samples(:,idx_train);
        l_train      = labels(idx_train);
        s_test       = samples(:,idx_test);
        l_test       = labels(idx_test);
        num_train    = length(idx_train);
        
        for idx_method = 1:length(funfcns)
            trace_accuracy = [];
            trace_test_loss= [];
            trace_obj_val  = [];
            trace_passes   = [];
            trace_time     = [];
            if( exist(['results_GGRLR_' func2str(funfcns{idx_method}) '_' dataset_name '.mat'],'file') )
                tdata = load(['results_GGRLR_' func2str(funfcns{idx_method}) '_' dataset_name '.mat'], 'trace_passes', 'trace_time', 'trace_accuracy', 'trace_obj_val', 'trace_test_loss');
                trace_accuracy = tdata.trace_accuracy(:,1:idx_fold-1);
                trace_test_loss= tdata.trace_test_loss(:,1:idx_fold-1);
                trace_obj_val  = tdata.trace_obj_val(:,1:idx_fold-1);
                trace_passes   = tdata.trace_passes(:,1:idx_fold-1);
                trace_time     = tdata.trace_time(:,1:idx_fold-1);
            end
            %Parameters setup
            %Parameters of model
            opts         = struct();
            opts.mu      = 1e-5; %parameter of graph-guided term
            opts.min_t   = 100;   %10sec at least
            
            %Parameters of algorithms
            opts.F       = F;     %The graph structure
            opts.beta    = 1;     %parameter of STOC-ADMM to balance augmented lagrange term
            opts.gamma   = 1e-2;  %Regularized Logistic Regression term
            opts.epochs  = n_epochs;     %maximum effective passes
            opts.max_it  = (1-1/K_fold)*length(labels)*opts.epochs;
            opts.checkp  = 0.01;  %save the solution very 1% of data processed
            opts.checki  = 10;%floor(opts.max_it * opts.checkp);
            opts.a       = 1;     %parameter of Ada_SADMM
            
            tempVar = zeros(size(samples,2),1);
            for idx_s = 1:size(samples,2)
                tempVar(idx_s) = samples(:,idx_s)'*samples(:,idx_s);
            end
            opts.L = 0.25*max(tempVar);
            eigFTF = eigs(opts.F'*opts.F, 1);
            opts.L1 = opts.beta*eigFTF + opts.L;
            opts.L2 = sqrt(max(2*opts.L*opts.L+eigFTF, 2*eigFTF));
            opts.L3 = max(8*opts.beta*eigFTF, sqrt(8*opts.L*opts.L + opts.beta*eigFTF))+opts.gamma;
            
            opts.eta = opts.beta*eigFTF;%parameter of RDA-ADMM and OPG-ADMM
            
            
            par_settings = [2.^(-7:1:2) 5:1:16 17:5:32 2.^(6:1:10)];
            switch func2str(funfcns{idx_method})
                case 'PEGSADM_Uniform'
                    switch datasets{idx_datasets}
                        case 'a9a'
                            idx_opt = 10;
                            opts.scaling = par_settings(idx_opt);
                        case '20news_100word'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                        case 'mushrooms'
                            idx_opt = 13;
                            opts.scaling = par_settings(idx_opt);
                        case 'w8a'
                            idx_opt = 14;
                            opts.scaling = par_settings(idx_opt);
                        case 'splice'
                            idx_opt = 14;
                            opts.scaling = par_settings(idx_opt) + 0.025;
                        case 'svmguide3'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                    end
                case 'PEGSADM_Nonuniform'
                    switch datasets{idx_datasets}
                        case 'a9a'
                            idx_opt = 11;
                            opts.scaling = par_settings(idx_opt);
                        case '20news_100word'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                        case 'mushrooms'
                            idx_opt = 13;
                            opts.scaling = par_settings(idx_opt);
                        case 'w8a'
                            idx_opt = 14;
                            opts.scaling = par_settings(idx_opt);
                        case 'splice'
                            idx_opt = 10;
                            opts.scaling = par_settings(idx_opt) + 0.022;
                        case 'svmguide3'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                    end
                case 'SGADM'
                    switch datasets{idx_datasets}
                        case 'a9a'
                            idx_opt = 7;
                            opts.scaling = par_settings(idx_opt);
                        case '20news_100word'
                            idx_opt = 6;
                            opts.scaling = par_settings(idx_opt);
                        case 'mushrooms'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                        case 'w8a'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                        case 'splice'
                            idx_opt = 7;
                            opts.scaling = par_settings(idx_opt);
                        case 'svmguide3'
                            idx_opt = 5;
                            opts.scaling = par_settings(idx_opt);
                    end
                case 'STOC_ADMM'
                    switch datasets{idx_datasets}
                        case 'a9a'
                            idx_opt = 4;
                            opts.scaling = par_settings(idx_opt);
                        case '20news_100word'
                            idx_opt = 6;
                            opts.scaling = par_settings(idx_opt);
                        case 'mushrooms'
                            idx_opt = 3;
                            opts.scaling = par_settings(idx_opt);
                        case 'w8a'
                            idx_opt = 2;
                            opts.scaling = par_settings(idx_opt);
                        case 'splice'
                            idx_opt = 3;
                            opts.scaling = par_settings(idx_opt);
                        case 'svmguide3'
                            idx_opt = 6;
                            opts.scaling = par_settings(idx_opt);
                    end
                case 'RDA_ADMM'
                    switch datasets{idx_datasets}
                        case 'a9a'
                            idx_opt = 10;
                            opts.scaling = par_settings(idx_opt);
                        case '20news_100word'
                            idx_opt = 23;
                            opts.scaling = par_settings(idx_opt);
                        case 'mushrooms'
                            idx_opt = 10;
                            opts.scaling = par_settings(idx_opt);
                        case 'w8a'
                            idx_opt = 9;
                            opts.scaling = par_settings(idx_opt);
                        case 'splice'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt);
                        case 'svmguide3'
                            idx_opt = 19;
                            opts.scaling = par_settings(idx_opt);
                    end
                case 'OPG_ADMM'
                    switch datasets{idx_datasets}
                        case 'a9a'
                            idx_opt = 12;
                            opts.scaling = par_settings(idx_opt);
                        case '20news_100word'
                            idx_opt = 23;
                            opts.scaling = par_settings(idx_opt);
                        case 'mushrooms'
                            idx_opt = 10;
                            opts.scaling = par_settings(idx_opt);
                        case 'w8a'
                            idx_opt = 10;
                            opts.scaling = par_settings(idx_opt);
                        case 'splice'
                            idx_opt = 8;
                            opts.scaling = par_settings(idx_opt) ;
                        case 'svmguide3'
                            idx_opt = 16;
                            opts.scaling = par_settings(idx_opt);
                    end
                    
                case 'Ada_SADMMdiag'
                    switch datasets{idx_datasets}
                        case 'splice'
                            idx_opt = 9;
                            opts.scaling = par_settings(idx_opt);
                        otherwise
                            idx_opt = 9;
                            opts.scaling = par_settings(idx_opt);
                    end
                case 'Ada_SADMMfull'
                    switch datasets{idx_datasets}
                        case 'splice'
                            idx_opt = 9;
                            opts.scaling = par_settings(idx_opt);
                        otherwise
                            idx_opt = 9;
                            opts.scaling = par_settings(idx_opt);
                    end
            end
            %Trainning
            t = cputime;
            outputs       = funfcns{idx_method}(s_train, l_train, opts);
            time_solving  = cputime - t;
            time_per_iter = time_solving/outputs.iter;
            fprintf('Method(%d/%d) %s, (%d/%d) folds, time_per_iter:%s\n',idx_method,length(funfcns),func2str(funfcns{idx_method}), idx_fold, K_fold, num2str(time_per_iter));
            
            num_traces = length(outputs.trace.times);
            idx_max = find(outputs.trace.times(:) > opts.min_t,1);
            if isempty(idx_max)
                idx_max = max(outputs.trace.times(:));
            end
            if idx_max >= 1000
                idx_sel = floor(1:idx_max*0.001:idx_max);
            else
                idx_sel = 1:1:idx_max;
            end
            for idx_trace = 1:length(idx_sel)
                x = outputs.trace.xs(:,idx_sel(idx_trace));
                trace_accuracy(idx_trace,idx_fold)  = get_accuracy(s_test, l_test, x);
                trace_time(idx_trace,idx_fold)      = outputs.trace.times(idx_sel(idx_trace));
                trace_passes(idx_trace,idx_fold)    = outputs.trace.iters(idx_sel(idx_trace))/num_train;
                trace_test_loss(idx_trace,idx_fold) = get_test_loss(s_test,l_test,x);
                trace_obj_val(idx_trace,idx_fold)   = get_obj_val(s_train, l_train, x, opts.F, opts.mu, opts.gamma);
            end
	    parsave_r(['results_GGRLR_' func2str(funfcns{idx_method}) '_' dataset_name '.mat'], trace_passes, trace_time, trace_accuracy, trace_obj_val, trace_test_loss);
        end
    end
end

draw_results_time_cv
