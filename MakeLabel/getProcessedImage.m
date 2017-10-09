function C = getProcessedImage( IMAGE )
    C = cell(1,2);
    channel = 3;
    [rows, cols] = size(IMAGE);
    cols = cols/channel;
    Seg = uint8(zeros(rows,cols));    
    %{
    Seg.i = [];
    Seg.j = [];
    Seg.s = [];
    %}
    Bnd.i = [];
    Bnd.j = [];
    Bnd.s = [];
    for y = 1:rows
        for x = 1:cols
            if IMAGE(y,x,2)==255 & IMAGE(y,x,1)==255 & IMAGE(y,x,3)==255
                Bnd.i = [Bnd.i, y];
                Bnd.j = [Bnd.j, x];
                Bnd.s = [Bnd.s, 1];
            elseif IMAGE(y,x,1)==255 & IMAGE(y,x,3)==255
                Seg(y,x) = 1;
                %{
                Seg.i = [Seg.i, y];
                Seg.j = [Seg.j, x];
                Seg.s = [Seg.s, 1];
                %}
            end
        end
    end
    %C{1,1} = sparse(Seg.i,Seg.j,Seg.s,rows, cols);
    C{1,1} = Seg;
    C{1,2} = sparse(Bnd.i,Bnd.j,Bnd.s,rows, cols);
end