function obj_val = get_obj_val(s_train,l_train,x, F, lambda, gamma)

obj_val = sum(log(1+exp(-l_train.*(s_train'*x))))/length(l_train);
obj_val = obj_val + lambda*sum(sum(abs(F*x)));
obj_val = obj_val + gamma*sum(abs(x));
end