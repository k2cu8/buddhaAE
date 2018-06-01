function [y,z,relu] = show_result_batch(folder, model_name)
y = read_batch_result(folder,strcat('/output_y_',model_name),'_*.csv');
relu = read_batch_result(folder,strcat('/output_relu_',model_name),'_*.csv');
z = read_batch_result(folder,strcat('/output_z_',model_name),'_*.csv');

figure;
scatter3(y(:,1),y(:,2),y(:,3),5,'filled');
axis equal

