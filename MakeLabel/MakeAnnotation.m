function MakeAnnotation(CLASS_NUM,CLASS,image_path)

IMG = imread(image_path);
fn = strsplit(image_path,'/');
fn = strsplit(fn{1,length(fn)},'.');
disp(fn)
[IMG_ROW, IMG_COLUMN] = size(IMG);
IMG_COLUMN = IMG_COLUMN /3 ;

processed = getProcessedImage(IMG);
Segmentation = uint8(processed{1,1});
Boundaries = processed{1,2};

%%% GTcls
GTcls.Boundaries = cell(CLASS_NUM,1);
for i = 1:CLASS_NUM
    if i == CLASS
        disp(['CLASS :' num2str(i)]);
        GTcls.Boundaries{i,1} = Boundaries;
    else
        GTcls.Boundaries{i,1} = sparse(IMG_ROW,IMG_COLUMN);
    end
end
GTcls.Segmentation = Segmentation;
GTcls.CategoriesPresent = CLASS;

%%% GTinst
GTinst.Segmentation = double(Segmentation);
GTinst.Boundaries = cell(1,1);
GTinst.Boundaries{1,1} = Boundaries;
GTinst.Categories = CLASS;

%figure
%image(IMG)
%figure
%spy(GTcls.Boundaries{1,1})
%figure
%spy(GTcls.Segmentation)

save('out/cls/'+string(fn{1,1})+'.mat','GTcls');
save('out/inst/'+string(fn{1,1})+'.mat','GTinst');

end