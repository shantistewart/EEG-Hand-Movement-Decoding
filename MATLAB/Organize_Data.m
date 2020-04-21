% This script will just take all the files competition files
% and have all of the first 2 trials turned into .txt data
% to be analyzed in Python

file_path = './Competition_2b/';

for i=1:9
    
    for j=1:2
        
        general_file = sprintf('B0%d0%dT.gdf', i, j);
        file_to_open = strcat(file_path, general_file);
        [s,hdr] = sload(file_to_open);
        %hdr = sopen(file_to_open,'r');
        %[s, hdr] = sread(hdr);
        hdr = sclose(hdr);
        
        output_folder = './Data_txt/';
        output_file = sprintf('Data%d_%d_T.txt',i,j);
        output_file_name = strcat(output_folder,output_file);
        writematrix(s(:,1:3), output_file_name, 'Delimiter', 'tab')
        
    end
    
end

