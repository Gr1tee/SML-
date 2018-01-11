clear all;
clc;
tic;

files = dir(fullfile('E:\\zlh\\研一上\\课程\\WEB搜索\\web\\size_ren','*.png'));
filelength = length(files);

for image_idx = 1:filelength
    image_RGB = imread(strcat('E:\\zlh\\研一上\\课程\\WEB搜索\\web\\size_ren\\',files(image_idx).name));
    image_YBR = rgb2ycbcr(im2double(image_RGB));
    [rows,cols,channels]=size(image_YBR);
    N_row = floor((rows-2)/6);
    N_col = floor((cols-2)/6);
    for row_idx = 1:N_row
        for col_idx = 1:N_col
    %         this_blocktmp1 = image_YBR(6*(N_idx-1)+1:1:6*N_idx+2,6*(N_idx-1)+1:1:6*N_idx+2,:);
    %         this_blocktmp2 = reshape(this_blocktmp1,[],3).';
    %         this_block = reshape(this_blocktmp2,1,[]);
            this_block = reshape(reshape(image_YBR(6*(row_idx-1)+1:1:6*row_idx+2, 6*(col_idx-1)+1:1:6*col_idx+2,:),[],3).',1,[]);
            this_DCT_mid = dct(this_block);
            this_DCT = this_DCT_mid(1:30);
            N_counter = N_col*(row_idx-1)+1*(col_idx-1)+1;
            resultofdct(image_idx,N_counter,:) = this_DCT;
        end
    end
    
end