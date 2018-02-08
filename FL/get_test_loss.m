function test_loss = get_test_loss(s_test,l_test,x)
test_loss = sum(log(1+exp(-l_test.*(s_test'*x))))/length(l_test);
end