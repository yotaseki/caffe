clear

list = dir('in');
for i = 3:length(list)
    fn = string(list(i).folder)+'/'+string(list(i).name)
    MakeAnnotation(1,1,fn{1,1});
end
