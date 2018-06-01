function show_result_relu(model_name)
[~,~,label] = show_result_batch('results',model_name);
U = unique(label(:,size(label,2)/2+1:end),'rows');
fprintf('There are %d labels\n', size(U,1));
