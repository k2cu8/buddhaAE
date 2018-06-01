function y = read_batch_result(prefix, model_name,suffix)

y_files = dir(strcat(prefix,model_name, suffix));
y = [];
%read y
for i = 1:size(y_files,1)
    fld_name = y_files(i).folder;
    f_name = y_files(i).name;
    y_ = csvread(fullfile(fld_name,f_name));
    y = [y;y_];
end





