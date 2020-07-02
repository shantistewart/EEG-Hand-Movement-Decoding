left_event = hex2dec('301');
right_event = hex2dec('302');

% So we want to iterate through the entire
% hdr.EVENT.TYP and fine instances where
% a 0x0301 (left) or 0x0302 (right) 

for i=1:length(hdr.EVENT.TYP);
    if hdr.EVENT.TYP(i) == left_event;
        left_start = hdr.EVENT.POS(i) + hdr.EVENT.DUR(i);
        % Gather 2.5 seconds of data (just like other EEG papers)
        left_end = left_start + 875;
        if (i ~= length(hdr.EVENT.TYP)) && (left_end >= hdr.EVENT.POS(i+1));
            "error"
            break;
        else
            left_sig = s(left_start:left_end, 1:3);
            % DO STUFF WITH DATA
        end
        % left hand event
        % Gather 3.5 seconds of data following this event
        
    elseif hdr.EVENT.TYP(i) == right_event;
        right_start = hdr.EVENT.POS(i) + hdr.EVENT.DUR(i);
        right_end = right_start + 875;
        
        if (i ~= length(hdr.EVENT.TYP)) && (right_end >= hdr.EVENT.POS(i+1));
            "error"
            break;
        else
            right_sig = s(right_start:right_end, 1:3);
            % DO STUFF WITH DATA
        end
        % right hand event
        % Gather 3.5 seconds of data following this event
        
    end
end
