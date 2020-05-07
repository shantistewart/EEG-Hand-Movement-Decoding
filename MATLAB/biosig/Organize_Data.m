% This script will just take all the files competition files
% and have all of the first 2 trials turned into .txt data
% to be analyzed in Python

file_path = './Competition_2b/';
left_event = hex2dec('301');
right_event = hex2dec('302');

% iterate through 9 data files
for k=1:9
    
    % iterate through first 2 session files
    for j=1:2
        
        general_file = sprintf('B0%d0%dT.gdf', k, j);
        file_to_open = strcat(file_path, general_file);
        [s,hdr] = sload(file_to_open);
        %hdr = sopen(file_to_open,'r');
        %[s, hdr] = sread(hdr);
        hdr = sclose(hdr);
        output_folder = './Data_txt/';
        
        num_left_sig = 0;
        num_right_sig = 0;
        
        for i=1:length(hdr.EVENT.TYP);
            if hdr.EVENT.TYP(i) == left_event;
                left_start = hdr.EVENT.POS(i) + hdr.EVENT.DUR(i);
                % Gather 3.5 seconds of data (just like other EEG papers)
                left_end = left_start + 875;
                if (i ~= length(hdr.EVENT.TYP)) && (left_end >= hdr.EVENT.POS(i+1));
                    "error"
                    break;
                else
                    num_left_sig = num_left_sig + 1;
                    left_sig = s(left_start:left_end, 1:3);
                    % Write data to file
                            
                    output_file = sprintf('Data_Left_%d_%d_%d.txt',k,j,num_left_sig);
                    output_file_name = strcat(output_folder,output_file);
                    writematrix(left_sig, output_file_name, 'Delimiter', 'tab');
                    
                end
        
            elseif hdr.EVENT.TYP(i) == right_event;
                right_start = hdr.EVENT.POS(i) + hdr.EVENT.DUR(i);
                right_end = right_start + 875;
        
                if (i ~= length(hdr.EVENT.TYP)) && (right_end >= hdr.EVENT.POS(i+1));
                    "error"
                    break;
                else
                    num_right_sig = num_right_sig + 1;
                    right_sig = s(right_start:right_end, 1:3);
                    % Write signal data to file
                    % File name convention: %d + %d + %d: patient + session + trial
                    output_file = sprintf('Data_Right_%d_%d_%d.txt',k,j,num_right_sig);
                    output_file_name = strcat(output_folder,output_file);
                    writematrix(right_sig, output_file_name, 'Delimiter', 'tab');
        
                end
                % right hand event
                % Gather 3.5 seconds of data following this event
        
            end
        end
        
    end
    
end
