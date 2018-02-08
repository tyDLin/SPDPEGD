% Save variables in parallel env
function parsave_r(fname,trace_passes, trace_time, trace_accuracy, trace_obj_val, trace_test_loss)
save(fname,'trace_passes','trace_time','trace_accuracy','trace_obj_val','trace_test_loss');
end