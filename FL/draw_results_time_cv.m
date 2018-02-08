clear; close all;

colors  = {'k','y','b','g','m','c','c','r','k'};
markers = {'^','+','o','+','o','>','<','v','o','p','h'};
LStyles = {'-','-','-','-','-','-','-','-',':'};
funfcn_batch = {@EGADM};
funfcn_stoc  = {@SGADM,@STOC_ADMM,@RDA_ADMM,@OPG_ADMM,@Ada_SADMMdiag,@Ada_SADMMfull,@PEGSADM};
funfcns      = [funfcn_batch, funfcn_stoc]; 
funfcns_names = []; for idx_fn = 1:length(funfcns); funfcns_names{idx_fn} = strrep(func2str(funfcns{idx_fn}), '_','-'); end
datasets     = {'splice','svmguide3','mushrooms','a9a','w8a'};

opts.epochs  = 5;
opts.min_time= 10;
en_subplot   = 0;

nruns_batch  = 1;
showp = 0.1;

size_font_title  = 14; size_font_legend = 12; size_font_data   = 14;
if en_subplot
    size_axis   = 12; size_axis_label   = 14;
else
    size_axis   = 24; size_axis_label   = 28;
end

if en_subplot
    figure(1);
    scrsz = get(0,'ScreenSize');
    set(gcf,'Position',[1 1 scrsz(3) scrsz(4)]);
    x_start = 0.098; x_end = 0.91; num_sub = length(datasets);
    for idx_sub = 1:num_sub
        annotation(gcf,'textbox',[(x_start + (idx_sub*2-1)*(x_end - x_start)/(2*num_sub)) 0.89 0.02 0.06],...
            'VerticalAlignment','middle',...
            'String',strrep(datasets{idx_sub},'_','-'),...
            'HorizontalAlignment','center',...
            'FontSize',14,...
            'FitBoxToText','off',...
            'EdgeColor','none');
    end
end

for idx_dataset = 1:length(datasets)
    dataset_name = datasets{idx_dataset};     
    switch datasets{idx_dataset}
        case 'splice'
            opts.min_time= 2;
        case 'svmguide3'
            opts.min_time= 2;
        case 'mushrooms'
            opts.min_time= 10;
        case '20news_100word'
            opts.min_time= 100;
        case 'a9a'
            opts.min_time= 10;
        case 'w8a'
            opts.min_time= 10;
    end
        
    % Data
    trace_accuracy   =[];    trace_time       =[];    trace_passes     =[];    trace_obj_val    =[];    trace_test_loss   =[];
    for idx_method = 1:length(funfcns)
        stoc_data = load(['results_FL_' func2str(funfcns{idx_method}) '_' dataset_name '.mat'],'trace_passes','trace_time','trace_accuracy','trace_obj_val','trace_test_loss');
        num_runs = size(stoc_data.trace_time,2);
        for idx_runs = 1:num_runs
            idx_max = find(opts.min_time <= stoc_data.trace_time(:,idx_runs),1);
            if idx_max < floor(1/showp)
                if floor(1/showp) <= find(stoc_data.trace_time(:,idx_runs) == max(stoc_data.trace_time(:,idx_runs)))
                    idx_max = floor(1/showp);
                else
                    idx_max = find(stoc_data.trace_time(:,idx_runs) == max(stoc_data.trace_time(:,idx_runs)));
                end
            end 
            if isempty(idx_max)
                idx_max = find(stoc_data.trace_time(:,idx_runs) == max(stoc_data.trace_time(:,idx_runs)),1);
            end
            idx_sel = floor(1:idx_max*showp:idx_max);
            for idx_trace = 1:length(idx_sel)
                trace_accuracy(idx_trace,idx_method,idx_runs)   = stoc_data.trace_accuracy(idx_sel(idx_trace),idx_runs);
                trace_time(idx_trace,idx_method,idx_runs)       = stoc_data.trace_time(idx_sel(idx_trace),idx_runs);
                trace_passes(idx_trace,idx_method,idx_runs)     = stoc_data.trace_passes(idx_sel(idx_trace),idx_runs);
                trace_obj_val(idx_trace,idx_method,idx_runs)    = stoc_data.trace_obj_val(idx_sel(idx_trace),idx_runs);
                trace_test_loss(idx_trace,idx_method,idx_runs)  = stoc_data.trace_test_loss(idx_sel(idx_trace),idx_runs);
            end
            while idx_trace<floor(1/showp)
                idx_trace = idx_trace+1;
                trace_accuracy(idx_trace,idx_method,idx_runs)   = trace_accuracy(idx_trace-1,idx_method,idx_runs);
                trace_time(idx_trace,idx_method,idx_runs)       = trace_time(idx_trace-1,idx_method,idx_runs);
                trace_passes(idx_trace,idx_method,idx_runs)     = trace_passes(idx_trace-1,idx_method,idx_runs);
                trace_obj_val(idx_trace,idx_method,idx_runs)    = trace_obj_val(idx_trace-1,idx_method,idx_runs);
                trace_test_loss(idx_trace,idx_method,idx_runs)  = trace_test_loss(idx_trace-1,idx_method,idx_runs);                
            end
        end
    end
    trace_accuracy_avg   =[];    trace_accuracy_std   =[];
    trace_time_avg       =[];    trace_time_std       =[];
    trace_passes_avg     =[];    trace_passes_std     =[];
    trace_obj_val_avg    =[];    trace_obj_val_std    =[];
    trace_test_loss_avg  =[];    trace_test_loss_std  =[];
    for idx_method = 1:length(funfcns)
        num_traces = length(trace_passes(:,idx_method,1));
        for idx_trace = 1:num_traces
            trace_accuracy_avg(idx_trace,idx_method) = mean(trace_accuracy(idx_trace,idx_method,:));
            trace_accuracy_std(idx_trace,idx_method) = std(trace_accuracy(idx_trace,idx_method,:));
            trace_time_avg(idx_trace,idx_method)     = mean(trace_time(idx_trace,idx_method,:));
            trace_time_std(idx_trace,idx_method)     = std(trace_time(idx_trace,idx_method,:));
            trace_passes_avg(idx_trace,idx_method)   = mean(trace_passes(idx_trace,idx_method,:));
            trace_passes_std(idx_trace,idx_method)   = std(trace_passes(idx_trace,idx_method,:));
            trace_obj_val_avg(idx_trace,idx_method)  = mean(trace_obj_val(idx_trace,idx_method,:));
            trace_obj_val_std(idx_trace,idx_method)  = std(trace_obj_val(idx_trace,idx_method,:));
            trace_test_loss_avg(idx_trace,idx_method)= mean(trace_test_loss(idx_trace,idx_method,:));
            trace_test_loss_std(idx_trace,idx_method)= std(trace_test_loss(idx_trace,idx_method,:));
        end
    end    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Accuracy vs Time %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(1)
    if en_subplot;subplot(3,num_sub,2*num_sub+idx_dataset);end
    min_y = inf;         max_y = -inf;
    for idx_method = 1:length(funfcns)
        errorbar(trace_time_avg(:,idx_method),trace_accuracy_avg(:,idx_method),trace_accuracy_std(:,idx_method), ...
            colors{idx_method},'Marker',markers{idx_method},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        temp_min_y = min(trace_accuracy_avg(:,idx_method));
        temp_max_y = max(trace_accuracy_avg(:,idx_method));
        if temp_min_y < min_y;min_y = temp_min_y;end
        if temp_max_y > max_y;max_y = temp_max_y;end
    end
    
    hold off;  
    xlim([0 opts.min_time]);
    set(gca,'fontsize',size_axis);
    xlabel('Time','FontSize',size_axis_label);
    if ~en_subplot;
        saveas(gca, [dataset_name '_accuracy_vs_time.eps'],'psc2');
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Loss vs Time %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(1)
    if en_subplot;subplot(3,num_sub,num_sub+idx_dataset);end
    min_y = inf;         max_y = -inf;
    for idx_method = 1:length(funfcns)
        errorbar(trace_time_avg(:,idx_method),trace_test_loss_avg(:,idx_method),trace_test_loss_std(:,idx_method), ...
            colors{idx_method},'Marker',markers{idx_method},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        temp_min_y = min(trace_test_loss_avg(:,idx_method));
        temp_max_y = max(trace_test_loss_avg(:,idx_method));
        if temp_min_y < min_y;min_y = temp_min_y;end
        if temp_max_y > max_y;max_y = temp_max_y;end
    end
    hold off;
    xlim([0 opts.min_time]);
    set(gca,'fontsize',size_axis);
    
    xlabel('Time','FontSize',size_axis_label);
    if ~en_subplot;
        saveas(gca, [dataset_name '_loss_vs_time.eps'],'psc2');
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Objective Value vs Time %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(1)
    if en_subplot;subplot(3,num_sub,idx_dataset);end
    min_y = inf;         max_y = -inf;
    for idx_method = 1:length(funfcns)
        errorbar(trace_time_avg(:,idx_method),trace_obj_val_avg(:,idx_method),trace_obj_val_std(:,idx_method), ...
            colors{idx_method},'Marker',markers{idx_method},'LineWidth',1.3,'LineStyle',LStyles{idx_method});     hold on;
        temp_min_y = min(trace_obj_val_avg(:,idx_method));
        temp_max_y = max(trace_obj_val_avg(:,idx_method));
        if temp_min_y < min_y;min_y = temp_min_y;end
        if temp_max_y > max_y;max_y = temp_max_y;end
    end
    hold off;
    xlim([0 opts.min_time]);
    set(gca,'fontsize',size_axis);
    xlabel('Time','FontSize',size_axis_label);
    if ~en_subplot;
        saveas(gca, [dataset_name '_obj_vs_time.eps'],'psc2');
    end
end

if en_subplot
    figure(1);
    x= [0.09,0.93];   y= [0.90,0.90];            annotation(gcf,'arrow',x,y);
    x= [0.108,0.108]; y= [0.95,0.06];            annotation(gcf,'arrow',x,y);
    suptitle('The Comparsion of All Methods on Fused Logistic Regression');    
    
    figure(1)
    h_legend=legend('EGADM', 'SGADM', 'SADMM', 'RDA-ADMM', 'OPG-ADMM', 'SADMMdiag', 'SADMMfull', 'SPDPEG');
    set(h_legend,'FontSize',size_font_legend,'Position',[0.22,0,0.6,0.06],'Box', 'off','Orientation','horizontal');
    
    annotation(gcf,'textarrow',[0.062125 0.065],...
        [0.852362707535126 0.152362707535126],'TextRotation',90,...
        'String','Objective Value',...
        'LineStyle','none',...
        'HeadStyle','none',...
        'FontSize',14);
    
    annotation(gcf,'textarrow',[0.075875 0.079375],...
        [0.557905491698599 0.177905491698599],'TextRotation',90,...
        'String','Test Loss',...
        'LineStyle','none',...
        'HeadStyle','none',...
        'FontSize',14);
    
    annotation(gcf,'textarrow',[0.050875 0.054375],...
    [0.314942528735633 0.214942528735633],'TextRotation',90,...
    'String','Prediction Accuracy',...
    'LineStyle','none',...
    'HeadStyle','none',...
    'FontSize',14);
end
